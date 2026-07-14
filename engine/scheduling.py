import asyncio
import dataclasses

import engine.batching
import queue
import os
import pathlib
import statistics
import threading
import time
import torch
import uuid
import engine.model


MAX_CONCURRENT_ACTIVE=64

_START_PROFILER = object()
_STOP_PROFILER = object()


@dataclasses.dataclass
class WorkItem:
    request_id: str
    req: engine.model.CompletionRequest
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    enqueue_ts: float

class Scheduler:
    def __init__(self, max_queue_size=1024, load_fn=engine.model.load):
        self.inbox = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(
            target=self._engine_main,
            name="engine-thread",
            daemon=True
        )
        self.ready = False
        self._profiler = None
        self._profiler_dir = os.environ.get("ENGINE_PROFILER_DIR")
        self._debug = os.environ.get("DEBUG") == "1"
        self.load_fn = load_fn

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

    def deliver(self, item: WorkItem, completion: engine.model.Completion):
        return (
                item, 
                {
                    "request_id": item.request_id,
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "usage": {
                        "prompt_tokens": completion.prompt_tokens,
                        "completion_tokens": completion.completion_tokens,
                        "total_tokens": completion.prompt_tokens + completion.completion_tokens,
                        },
                    }
                )


    def infer(self, items: list[WorkItem]):
        requests = [item.req for item in items]
        batcher = engine.batching.StaticBatcher(
                self.model,
                self.tokenizer,
                self.stop_ids,
                requests
                )
        n = num_active = len(items)
        cum_active = 0
        finsish_steps = []
        step = 0
        while not batcher.is_done():
            completions_by_id = batcher.step()
            step += 1
            num_active -= len(completions_by_id)
            cum_active += num_active
            finsish_steps.extend([step] * len(completions_by_id))
            yield from (self.deliver(items[i], completion) for i, completion in completions_by_id.items())

        if self._debug and step:
            eff_batch = cum_active / step
            print(f"[util] steps={step} eff_batch={eff_batch:.1f} occ={eff_batch / n}")
            print(f"[finish step] p50={statistics.median(finsish_steps)}")
            


    def _collect(self, window_s=0.02):
        batch = []
        first_ts = None
        while len(batch) < MAX_CONCURRENT_ACTIVE:
            if not batch:
                item = self.inbox.get()
            else:
                remaining = first_ts + window_s - time.monotonic()
                if remaining < 0.0:
                    break
                try:
                    item = self.inbox.get(timeout=remaining)
                except queue.Empty:
                    break
            if item is None:
                return None
            elif item is _START_PROFILER:
                self.start_profile()
                continue
            elif item is _STOP_PROFILER:
                self.stop_profile()
                continue
            if not batch:
                first_ts = time.monotonic()
            batch.append(item)
        return batch

    def _engine_main(self):
        self.model, self.tokenizer, self.stop_ids = self.load_fn()
        self._warmup()
        self.ready = True

        while True:
            batch = self._collect()
            if batch is None:
                break

            if self._debug:
                now = time.monotonic()
                waits = [(now - it.enqueue_ts) * 1000 for it in batch]
                print(f"[batch] n={len(batch):>2} wait_ms: p50={statistics.median(waits):.1f} max={max(waits):.1f}")

            try:
                finished = self.infer(batch)
                for item, result in finished:
                    self._set_result_threadsafe(item, result)
            except Exception as exc:
                for item in batch:
                    self._set_exception_threadsafe(item, exc)

    def start(self):
        if self.thread.is_alive():
            return
        self.thread.start()


    def _warmup(self):
        for n in [1, 2, 4, 8, 16, 32, MAX_CONCURRENT_ACTIVE]:
            reqs = [engine.model.CompletionRequest(
                messages=[{"role": "user", "content": "hi"}], max_tokens=2)
                    for _ in range(n)]
            with torch.inference_mode():
                batcher = engine.batching.StaticBatcher(self.model, self.tokenizer, self.stop_ids, reqs)
                for _ in range(2):
                    batcher.step()

    def submit(self, req: engine.model.CompletionRequest):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = uuid.uuid4().hex

        item = WorkItem(
            request_id=request_id,
            req=req,
            future=future,
            loop=loop,
            enqueue_ts=time.monotonic()
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

