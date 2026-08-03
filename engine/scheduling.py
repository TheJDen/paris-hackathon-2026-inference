import asyncio
import engine.caching
import collections
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
import engine.patches.attn
import engine.patches.gdn


MAX_CONCURRENT_ACTIVE=64
MAX_LEN = 2560

_START_PROFILER = object()
_STOP_PROFILER = object()


@dataclasses.dataclass
class WorkItem:
    request_id: str
    req: engine.model.CompletionRequest
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    enqueue_ts: float

@dataclasses.dataclass(kw_only=True)
class SeqState:
    item: WorkItem
    stop_ids: set[int]
    max_tokens: int
    prompt_len: int
    generated: list[int] = dataclasses.field(default_factory=list)

    def advance(self, tok: int) -> str | None:
        if tok in self.stop_ids:
            return "stop"
        self.generated.append(tok)
        return "length" if len(self.generated) >= self.max_tokens else None


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
        self.batching_mode = os.environ.get("BATCHING_MODE", "continuous")

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
        batcher = engine.batching.StaticBatcher(self.model_bundle, requests)
        n = num_active = len(items)
        cum_active = 0
        finish_steps = []
        step = 0
        while not batcher.is_done():
            completions_by_id = batcher.step()
            step += 1
            num_active -= len(completions_by_id)
            cum_active += num_active
            finish_steps.extend([step] * len(completions_by_id))
            yield from (self.deliver(items[i], completion) for i, completion in completions_by_id.items())

        if self._debug and step:
            eff_batch = cum_active / step
            print(f"[util] steps={step} eff_batch={eff_batch:.1f} occ={eff_batch / n}")
            print(f"[finish step] p50={statistics.median(finish_steps)}")
            


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
        self.model_bundle = self.load_fn()
        if self.batching_mode == "continuous":
            engine.patches.attn.patch_attention()
            engine.patches.gdn.patch_gdn()
            self.slotcache = engine.caching.SlotCache(
                self.model_bundle.model.config,
                MAX_CONCURRENT_ACTIVE,
                MAX_LEN,
                self.model_bundle.model.device
            )
            self.running = {}
            self.ready = True
            self._run_continuous_batches()
        else:
            self._warmup()
            self.ready = True
            self._run_static_batches()

    def _run_static_batches(self):
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

    def drain_inbox(self, waiting: collections.deque[WorkItem], block: bool):
        first = True
        while True:
            try:
                item = self.inbox.get() if (block and first) else self.inbox.get_nowait()
            except queue.Empty:
                return True
            first = False
            if item is None:
                return False
            elif item is _START_PROFILER:
                self.start_profile()
            elif item is _STOP_PROFILER:
                self.stop_profile()
            else:
                waiting.append(item)

    def _prefill(self, items: list[WorkItem]):
        input_ids = [self.model_bundle.tokenizer.apply_chat_template(
                [item.req.messages],
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt"
        ).to(self.model_bundle.model.device).input_ids for item in items]
        slots = [self.slotcache.alloc() for _ in range(len(items))]
        with torch.inference_mode():
            for i, item in enumerate(items):
                slot = torch.tensor([slots[i]], dtype=torch.int32, device=self.model_bundle.model.device)
                with torch.profiler.record_function("prefill"):
                    h = self.model_bundle.model.model(
                        input_ids[i],
                        past_key_values=None,
                        use_cache=False,
                        slotcache=self.slotcache,
                        slots=slot,
                        decoding=False,
                    ).last_hidden_state
                self.slotcache.lens[slot] = input_ids[i].shape[1]
                logits = self.model_bundle.model.lm_head(h[:, -1, :])

                self.running[slots[i]] = SeqState(
                    item=item,
                    prompt_len=input_ids[i].shape[1], 
                    stop_ids=self.model_bundle.stop_ids,
                    max_tokens=item.req.max_tokens
                )
                with torch.profiler.record_function("sample"):
                    next_tok = int(engine.model.sample_next(
                        logits,
                        temperature=torch.tensor([item.req.temperature], device=self.model_bundle.model.device),
                        top_p=torch.tensor([item.req.top_p], device=self.model_bundle.model.device),
                    ).item())
                stop_reason = self.running[slots[i]].advance(next_tok)
                if stop_reason is not None:
                    yield self._finish(slots[i], stop_reason)

    def _decode_step(self):
        slot_list = list(self.running)
        slots = torch.tensor(slot_list, dtype=torch.int32, device=self.model_bundle.model.device)
        tok = torch.tensor([[self.running[slot].generated[-1]] for slot in slot_list], device=self.model_bundle.model.device)
        temp = torch.tensor([self.running[slot].item.req.temperature for slot in slot_list], device=self.model_bundle.model.device)
        top_p = torch.tensor([self.running[slot].item.req.top_p for slot in slot_list], device=self.model_bundle.model.device)
        with torch.inference_mode():
            with torch.profiler.record_function("decode"):
                h = self.model_bundle.model.model(
                    tok,
                    past_key_values=None,
                    use_cache=False,
                    slotcache=self.slotcache,
                    position_ids=self.slotcache.lens[slots].unsqueeze(1),
                    slots=slots,
                    decoding=True,
                ).last_hidden_state
                logits = self.model_bundle.model.lm_head(h[:, -1, :])
            with torch.profiler.record_function("sample"):
                next_tok = engine.model.sample_next(
                    logits,
                    temperature=temp,
                    top_p=top_p,
                ).squeeze(1).tolist()
            self.slotcache.lens[slots] += 1
            for i, slot in enumerate(slot_list):
                stop_reason = self.running[slot].advance(next_tok[i])
                if stop_reason is not None:
                    yield self._finish(slot, stop_reason)

    def _complete(self, seq_state: SeqState, stop_reason: str):
        text = self.model_bundle.tokenizer.decode(seq_state.generated, skip_special_tokens=True)
        return engine.model.Completion(
            text,
            seq_state.prompt_len,
            len(self.model_bundle.tokenizer.encode(text, add_special_tokens=False)),
            stop_reason
        )

    def _finish(self, slot: int, stop_reason: str):
        completion = self._complete(self.running[slot], stop_reason)
        self.slotcache.release(slot)
        return self.deliver(self.running.pop(slot).item, completion)

    def _run_continuous_batches(self):
        waiting = collections.deque()
        t_prefill = t_decode = 0.0
        cum_occ = peak_occ = 0
        step = tokens = 0
        last_report = time.monotonic()
        while True:
            block = not self.running and not waiting
            if not self.drain_inbox(waiting, block=block):
                break
            prefill_items = [waiting.popleft() for _ in range(min(len(waiting), self.slotcache.num_free_slots()))]
            if prefill_items:
                t0 = time.monotonic()
                for item, result in self._prefill(prefill_items):
                    self._set_result_threadsafe(item, result)
                tp = time.monotonic() - t0
                t_prefill += tp
                if self._debug:
                    print(f"[prefill] n={len(prefill_items)} took={tp*1e3}ms running={len(self.running)} wait={len(waiting)}")
            if self.running:
                n = len(self.running)
                t0 = time.monotonic()
                for item, result in self._decode_step():
                    self._set_result_threadsafe(item, result)
                t_decode += time.monotonic() - t0
                cum_occ += n
                peak_occ = max(peak_occ, n)
                step += 1
                tokens += n
            if self._debug and cum_occ and time.monotonic() - last_report > 2:
                wall = t_prefill + t_decode
                print(f"[cont] running={len(self.running)} wait={len(waiting)} occ_mean={cum_occ/step}/{MAX_CONCURRENT_ACTIVE} "
                      f"peak={peak_occ} prefill%={t_prefill/wall} decode_tok/s={tokens/t_decode} eff_tok/s={tokens/wall}")



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
                batcher = engine.batching.StaticBatcher(self.model_bundle, reqs)
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

