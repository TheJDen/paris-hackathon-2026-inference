import asyncio
import concurrent
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch
import transformers
import transformers.integrations.moe

import engine.batching
import engine.caching
import engine.kernels.moe_experts
import engine.loading
import engine.model_running
import engine.patches.attn
import engine.patches.gdn
import engine.records

MAX_CONCURRENT_ACTIVE=64
MAX_LEN = 2560

class AsyncEngine:
    def __init__(self, max_queue_size=1024, load_fn=engine.loading.load):
        self.inbox = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(
            target=self._engine_main,
            name="engine-thread",
            daemon=True
        )
        self.ready = False
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
        ).input_ids[0]


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
        self.batcher.warmup()
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
            if isinstance(item, tuple) and item[0] == "call": # call arb fn on engine thread
                item[1]()
                continue
            items.append(item)
        return items

    def run_on_engine_thread(self, fn):
        fut = concurrent.futures.Future()
        def task():
            try:
                fut.set_result(fn())
            except Exception as e:
                fut.set_exception(e)
        self.inbox.put_nowait(("call", task))
        return fut.result()

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
