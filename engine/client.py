import asyncio
import dataclasses
import huggingface_hub
import json
import queue
import os
import pathlib
import pydantic
import safetensors
import threading
import time
import torch
import tqdm
import transformers
import uuid


MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
MAX_CONCURRENT_ACTIVE=64

_START_PROFILER = object()
_STOP_PROFILER = object()

class ChatCompletionRequest(pydantic.BaseModel):
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0

@dataclasses.dataclass
class WorkItem:
    request_id: str
    req: ChatCompletionRequest
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop

def sample_next(logits, temperature=4.0, top_p=1.0):
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature

    if top_p < 1.0:
        s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(s_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        s_logits = s_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, s_idx, s_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

class Client:
    def __init__(self, max_queue_size=1024):
        self.inbox = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(
            target=self._engine_main,
            name="engine-thread",
            daemon=True
        )
        self.ready = False
        self._profiler = None
        self._profiler_dir = os.environ.get("ENGINE_PROFILER_DIR")

    def _drain_new_requests(self, waiting):
        item = self.inbox.get()
        if item is None:
            return False
        if item is _START_PROFILER:
            self.start_profile()
            return True
        if item is _STOP_PROFILER:
            self.stop_profile()
            return True
        waiting.append(item)
        while True:
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                return True
            if item is None:
                return False
            if item is _START_PROFILER:
                self.start_profile()
                continue
            if item is _STOP_PROFILER:
                self.stop_profile()
                continue
            waiting.append(item)

    def _set_result_threadsafe(self, item: WorkItem, result):
        def _set_result() -> None:
            if not item.future.done():
                item.future.set_result(result)

        item.loop.call_soon_threadsafe(_set_result)

    def _set_exception_threadsafe(self, item: WorkItem, exc):
        def _set_exc() -> None:
            if not item.future.done():
                item.future.set_exception(exc)

        item.loop.call_soon_threadsafe(_set_exc)


    def infer(self, batch: list[WorkItem]):

        stop_ids = {self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|im_end|>")}

        finished = []
        with torch.profiler.record_function(f"infer[n={len(batch)}]"):
            for item in batch:
                with torch.profiler.record_function("tokenize"):
                    input_ids = self.tokenizer.apply_chat_template(
                            item.req.messages,
                            add_generation_prompt=True,
                            enable_thinking=False,
                            return_tensors="pt"
                    ).input_ids.to(device=self.model.device)
                seq = input_ids
                finish_reason = "length"                 # hit the token budget unless we break out
                for _ in range(item.req.max_tokens):
                    with torch.inference_mode():
                        with torch.profiler.record_function("forward"):
                            logits = self.model(seq, use_cache=False).logits
                        with torch.profiler.record_function("sample"):
                            next_id = sample_next(logits[:, -1, :], temperature=item.req.temperature, top_p=item.req.top_p)
                            if next_id.item() in stop_ids:
                                finish_reason = "stop"
                                break
                        seq = torch.cat([seq, next_id], dim=1)
                gen_ids = seq[0, input_ids.shape[1]:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                prompt_tokens = int(input_ids.shape[1])
                completion_tokens = int(gen_ids.shape[0])

                finished.append(
                        (
                            item,
                            {
                                "request_id": item.request_id,
                                "text": text,
                                "finish_reason": finish_reason,
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                    },
                                },
                            )
                        )
        return finished

    def _engine_main(self):
        self._load()
        self._warmup()
        self.ready = True
        waiting = []
        active = []

        while True:
            keep_going = self._drain_new_requests(waiting)
            if not keep_going:
                break
            while waiting and len(active) < MAX_CONCURRENT_ACTIVE:
                active.append(waiting.pop(0))
            if not active:
                continue

            batch = active

            try:
                finished = self.infer(batch)
                for item, result in finished:
                    self._set_result_threadsafe(item, result)
            except BaseException as exc:
                for item in batch:
                    self._set_exception_threadsafe(item, exc)
            active = []

    def start(self):
        if self.thread.is_alive():
            return
        self.thread.start()

    def _load(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_dir = huggingface_hub.snapshot_download(MODEL_ID)
        cfg = transformers.AutoConfig.from_pretrained(model_dir)
        cls = getattr(transformers, cfg.architectures[0])   # Qwen3_5MoeForConditionalGeneration
        with torch.device("cuda"):
            self.model = cls._from_config(cfg, dtype=torch.bfloat16)
        self.model.eval()

        idx = os.path.join(model_dir, "model.safetensors.index.json")
        weight_map = json.load(open(idx))["weight_map"]
        shards = sorted(set(weight_map.values()))

        gpu_tensors = dict(self.model.state_dict())
        loaded_names = set()
        with tqdm.tqdm(total=len(gpu_tensors), desc="Loading weights", unit="tensor") as pbar:
            for shard in shards:
                with safetensors.safe_open(os.path.join(model_dir, shard), framework="pt", device="cuda") as f:
                    for name in f.keys():
                        if name not in gpu_tensors:
                            continue
                        gpu_tensors[name].data.copy_(f.get_tensor(name))
                        loaded_names.add(name)
                        pbar.update(1)
                pbar.set_postfix_str(shard[-25:])
        missing_names = set(gpu_tensors) - loaded_names 
        assert not missing_names, f"never filled {missing_names}"


    def _warmup(self):
        enc = self.tokenizer(["hello"], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            self.model.generate(**enc, max_new_tokens=4, do_sample=False)

    def submit(self, req: ChatCompletionRequest):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = uuid.uuid4().hex

        item = WorkItem(
            request_id=request_id,
            req=req,
            future=future,
            loop=loop,
        )

        self.inbox.put_nowait(item)
        return request_id, future

    def queue_start_profile(self):
        if self._profiler_dir is None:
            raise RuntimeError("profiling disabled; set ENGINE_PROFILER_DIR")
        self.inbox.put_nowait(_START_PROFILER)

    def start_profile(self):
        if self._profiler is not None:
            print("Profiler already in progress")
            return
        self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False
        )
        self._profiler.start()

    def queue_stop_profile(self):
        if self._profiler_dir is None:
            raise RuntimeError("profiling disabled; set ENGINE_PROFILER_DIR")
        self.inbox.put_nowait(_STOP_PROFILER)

    def stop_profile(self):
        if self._profiler_dir is None:
            print("profiling disabled; set ENGINE_PROFILER_DIR")
            return
        if self._profiler is None:
            print("Profiler not running")
            return
        self._profiler.stop()
        pathlib.Path(self._profiler_dir).mkdir(parents=True, exist_ok=True)
        out = os.path.join(self._profiler_dir, f"trace-{int(time.time())}.json.gz")
        self._profiler.export_chrome_trace(out) # view in ui.perfetto.dev / chrome://tracing
        print(f"trace: {out}")
        self._profiler = None

    def cancel_id(self, req_id):
        pass

    async def shutdown(self):
        pass
