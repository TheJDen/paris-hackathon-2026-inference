import asyncio
from concurrent.futures import ThreadPoolExecutor
import engine.batching
import engine.caching
import engine.kernels.moe_experts
import engine.loading
import engine.model_running
import engine.patches.attn
import engine.patches.gdn
import engine.records
import os
import pathlib
import queue
import threading
import time
import torch
import tqdm
import transformers
import transformers.integrations.moe
import uuid


MAX_CONCURRENT_ACTIVE=64
MAX_LEN = 2560

_START_PROFILER = object()
_STOP_PROFILER = object()

class AsyncEngine:
    def __init__(self, max_queue_size=1024, load_fn=engine.loading.load):
        self.inbox = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(
            target=self._engine_main,
            name="engine-thread",
            daemon=True
        )
        self.ready = False
        self._profiler = None
        self._profiler_dir = os.environ.get("ENGINE_PROFILER_DIR")
        self.load_fn = load_fn
        self.batching_mode = os.environ.get("BATCHING_MODE", "continuous")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(engine.loading.MODEL_ID)
        self._tok_pool = ThreadPoolExecutor(max_workers=4)
        self.stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}

    def _set_result_threadsafe(self, item: engine.records.WorkItem, result):
        def _set_result() -> None:
            if not item.future.done():
                item.future.set_result(result)

        item.loop.call_soon_threadsafe(_set_result)

    def _set_exception_threadsafe(self, item: engine.records.WorkItem, exc):
        def _set_exc() -> None:
            if not item.future.done():
                item.future.set_exception(exc)

        item.loop.call_soon_threadsafe(_set_exc)

    def _tokenize(self, messages) -> torch.Tensor:
        return self.tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt"
        ).input_ids


    def _engine_main(self):
        self.model = model = self.load_fn()
        transformers.integrations.moe.ExpertsInterface.register("triton_grouped", engine.kernels.moe_experts.fbgemm_grouped_experts_forward)
        self.model.config._experts_implementation = "triton_grouped"
        if self.batching_mode == "continuous":
            engine.patches.attn.patch_attention()
            engine.patches.gdn.patch_gdn()
            self.slotcache = engine.caching.SlotCache(
                model.config,
                MAX_CONCURRENT_ACTIVE,
                MAX_LEN,
                model.device
            )
            self.model_runner = engine.model_running.ModelRunner(model, self.slotcache)
            self.batcher = engine.batching.ContinuousBatcher(
                self.model_runner,
                self.slotcache,
                self.stop_ids,
            )
        else:
            self.batcher = engine.batching.StaticBatcher(model, self.stop_ids)
        self._warmup()
        self.ready = True
        self._run_batches()

    def drain_inbox(self, block: bool, window: float):
        items = []
        first = True
        deadline = None
        while True:
            if block and first:
                item = self.inbox.get()
                deadline = time.monotonic() + window
            else:
                remaining = (deadline - time.monotonic()) if deadline is not None else 0.0
                try:
                    item = self.inbox.get(timeout=remaining) if remaining > 0.0 else self.inbox.get_nowait()
                except queue.Empty:
                    break
            first = False
            if item is None:
                return None
            elif item is _START_PROFILER:
                self.start_profile()
            elif item is _STOP_PROFILER:
                self.stop_profile()
            else:
                items.append(item)
        return items

    def _run_batches(self):
        while True:
            block = not self.batcher.has_work()
            items = self.drain_inbox(block=block, window=self.batcher.collect_window)
            if items is None:
                break
            self.batcher.add(items)

            try:
                for item, raw_result in self.batcher.step():
                    self._set_result_threadsafe(item, raw_result)
            except Exception as exc:
                for item in self.batcher.abort_items():
                    self._set_exception_threadsafe(item, exc)

    def start(self):
        if self.thread.is_alive():
            return
        self.thread.start()

    def _warmup(self):
        for n in tqdm.tqdm([1, 2, 4, 8, 16, 32, MAX_CONCURRENT_ACTIVE], desc="Warmup shapes"):
            self.batcher.add([self._warmup_item() for _ in range(n)])
            while self.batcher.has_work():
                for _ in self.batcher.step():
                    pass

    def _warmup_item(self):
        req = engine.records.CompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=2
        )
        return engine.records.WorkItem(
            req=req,
            future=None,
            loop=None,
            enqueue_ts=0.0,
            input_ids=self._tokenize(req.messages)
        )

    async def generate(self, req: engine.records.CompletionRequest):
        loop = asyncio.get_running_loop()
        input_ids = await loop.run_in_executor(self._tok_pool, self._tokenize, req.messages)
        request_id, future = self._submit(req, input_ids)
        raw = await future
        text, num_toks = await loop.run_in_executor(self._tok_pool, self._detokenize, raw.generated)
        return engine.records.Completion(
            text,
            raw.prompt_len,
            num_toks,
            raw.finish_reason,
            request_id
        )

    def _submit(self, req, input_ids):
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = engine.records.WorkItem(
            req=req,
            future=future,
            loop=loop,
            enqueue_ts=time.monotonic(),
            input_ids=input_ids
        )
        self.inbox.put_nowait(item)
        return request_id, future

    def _detokenize(self, token_ids) -> tuple[str, int]:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, len(self.tokenizer.encode(text, add_special_tokens=False))

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

