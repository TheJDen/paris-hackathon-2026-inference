"""Top-level Engine.

Phase 2a: continuous batching, no `model.generate`.

The engine bridges the asyncio FastAPI handlers and a synchronous
single-threaded scheduler that drives `runner.prefill` / `runner.decode`
on a slot-pool KV cache. HuggingFace `transformers` is used **only** as
a weight loader and an `nn.Module` library — `model.generate()` is gone,
HF's `DynamicCache` is gone, HF's left-padded batched generate is gone.

Lifecycle:

  1. `Engine.build(...)` loads the HF model + builds the ModelRunner +
     Scheduler.
  2. `Engine.start()` spawns a single dedicated thread (`_engine_executor`,
     1 worker) and runs the scheduler's drain loop on it forever.
  3. `Engine.generate(...)` wraps the user request as a Sequence, drops
     it on the scheduler's waiting queue, and awaits the Sequence's
     future. The scheduler resolves the future when the sequence
     finishes.

The single-thread executor is what makes torch.profiler happy across
batches (CUPTI binds callbacks to the thread that first imports them).
It also avoids any cross-thread torch nonsense in the model forward.

Phase 0 stub mode is preserved for laptop API smoke tests.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from typing import Literal

from engine.runtime.metrics import metrics
from engine.runtime.profiling import (
    annotate_hf_model_for_profiling,
    enable_torch_profiler,
    export_torch_profile,
    time_region,
    torch_profiler_enabled,
    torch_profiler_tag,
)
from engine.runtime.prefix_cache import PrefixCache
from engine.runtime.scheduler import Scheduler, SchedulerConfig
from engine.runtime.sequence import (
    GenerationResult,
    SamplingParams,
    Sequence,
    SequenceStatus,
    make_request_id,
)
from engine.tokenizer.chat_template import ChatTokenizer, THINK_OPEN


log = logging.getLogger(__name__)


FinishReason = Literal["stop", "length"]


class Engine:
    """Engine boundary used by the FastAPI server.

    Construct via `Engine.build(...)`. The `stub` constructor flag returns
    an instance that bypasses model loading entirely; useful for the
    laptop API smoke tests in Phase 0.
    """

    def __init__(
        self,
        model_name: str,
        *,
        stub: bool = False,
        tp: int = 1,
        max_batch: int = 64,
        max_model_len: int = 4096,
        device: str = "cuda:0",
        attn_impl: str = "sdpa",
        max_prefill_batch: int = 8,
        batch_window_ms: float = 5.0,  # legacy CLI arg, ignored by Phase 2a
        profile_torch_after_batches: int = 0,
        profile_torch_min_batch_size: int = 1,
        profile_torch_tag: str = "run",
        enable_cuda_graphs: bool = True,
        torch_compile: bool = True,
    ) -> None:
        self.model_name = model_name
        self.stub = stub
        self.tp = tp
        self.max_batch = max_batch
        self.max_model_len = max_model_len
        self.device = device
        self.attn_impl = attn_impl
        self.max_prefill_batch = max_prefill_batch

        # One-shot torch.profiler capture: skip the first N batches as
        # warmup, then capture the FIRST batch (decode step) after that
        # whose batch size is at least `profile_torch_min_batch_size`.
        self._profile_after_batches = int(profile_torch_after_batches)
        self._profile_min_batch_size = max(1, int(profile_torch_min_batch_size))
        self._profile_tag = str(profile_torch_tag)
        self._profile_done = False
        self._batch_index = 0  # incremented per decode step
        self._enable_cuda_graphs = bool(enable_cuda_graphs)
        self._torch_compile = bool(torch_compile)

        self.tokenizer: ChatTokenizer | None = None
        self.runner = None
        self.scheduler: Scheduler | None = None
        self.prefix_cache: PrefixCache | None = None

        # Single-thread executor so torch.profiler thread-affinity holds.
        self._gen_executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Cross-thread wakeup for the scheduler drain loop.
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._drain_future: concurrent.futures.Future | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        if not stub:
            self.tokenizer = ChatTokenizer(model_name)
            self._gen_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="engine-gen"
            )

        metrics.set_max_batch_capacity(self.max_batch)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, model_name: str, **kwargs) -> "Engine":
        eng = cls(model_name, **kwargs)
        if not eng.stub:
            eng._load()
        return eng

    def _load(self) -> None:
        """Heavy load: HF model + slot-pool runner + scheduler."""
        from engine.model.qwen3_next import load_model
        from engine.runtime.model_runner import ModelRunner

        with time_region("engine.load_model"):
            loaded = load_model(
                self.model_name,
                device=self.device,
                attn_impl=self.attn_impl,
            )
        self._loaded = loaded

        # If profiling is armed, install the per-layer record_function
        # hooks BEFORE the first forward so the chrome trace is readable.
        if torch_profiler_enabled() or self._profile_after_batches > 0:
            n = annotate_hf_model_for_profiling(loaded.model)
            log.info("installed %d profile hooks on HF model layers", n)

        with time_region("engine.build_runner"):
            self.runner = ModelRunner(
                hf_model=loaded.model,
                text_config=loaded.text_config,
                device=loaded.device,
                num_slots=self.max_batch,
                max_seq_len=self.max_model_len,
                enable_cuda_graphs=self._enable_cuda_graphs,
                compile=self._torch_compile,
            )
            log.info(
                "model runner ready: cache_mem=%.2f GB", self.runner.cache_memory_gb()
            )

        self.scheduler = Scheduler(
            runner=self.runner,
            config=SchedulerConfig(
                num_slots=self.max_batch,
                max_seq_len=self.max_model_len,
                max_decode_batch=self.max_batch,
                max_prefill_batch=self.max_prefill_batch,
            ),
        )

        # Prefix cache: TEMPORARILY DISABLED — interacts badly with batched
        # prefill (tensor shape mismatch from the 3-token shift). 3 tokens of
        # savings is negligible, will re-enable after debugging.
        self.prefix_cache = None
        if False:  # disabled
            self.prefix_cache = PrefixCache(
                cache=self.runner.cache,
                num_slots=self.max_batch,
            )
        # Remove the reserved slot from the scheduler's free pool so it is
        # never assigned to a regular request.
        if self.prefix_cache is not None:
            reserved = self.prefix_cache.RESERVED_SLOT
            if reserved in self.scheduler.free_slots:
                self.scheduler.free_slots.remove(reserved)
                log.info("prefix_cache: removed slot %d from scheduler free pool", reserved)

        # Warm up: one-time prefill of the prefix into the reserved slot.
        # We need the tokenizer's raw encode function (no chat template).
        assert self._loaded is not None

        def _raw_encode(messages_or_text, /) -> list[int]:
            """Encode either a messages list (via chat template) or a plain string."""
            if isinstance(messages_or_text, str):
                return self._loaded.tokenizer.encode(
                    messages_or_text, add_special_tokens=False
                )
            # messages list → apply chat template, then encode
            from engine.tokenizer.chat_template import ChatTokenizer as _CT
            # Re-use the existing tokenizer wrapper's tok
            text = self.tokenizer.tok.apply_chat_template(
                messages_or_text,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ) if hasattr(self.tokenizer.tok, "apply_chat_template") else ""
            return self._loaded.tokenizer.encode(text, add_special_tokens=False)

        if self.prefix_cache is not None:
            try:
                self.prefix_cache.warm_up(
                    runner=self.runner,
                    tokenizer_encode_fn=_raw_encode,
                )
            except Exception as exc:
                log.exception("prefix_cache warm_up failed (%s); continuing without it", exc)
                self.prefix_cache = None

        # Pass the prefix cache reference to the scheduler so admit can use it.
        if self.prefix_cache is not None:
            self.scheduler.prefix_cache = self.prefix_cache
            log.info(
                "prefix_cache: ready — prefix_len=%d tokens, reserved_slot=%d",
                self.prefix_cache.prefix_len,
                self.prefix_cache.RESERVED_SLOT,
            )

    async def start(self) -> None:
        """Spawn the dedicated engine thread + drain loop. Idempotent."""
        if self.stub or self._drain_future is not None:
            return
        assert self._gen_executor is not None
        self._loop = asyncio.get_event_loop()
        self._drain_future = self._gen_executor.submit(self._drain_blocking)

    async def stop(self) -> None:
        if self._drain_future is not None:
            self._stop_event.set()
            self._wake_event.set()
            try:
                # Don't block forever — the drain loop checks stop every step
                await asyncio.get_event_loop().run_in_executor(
                    None, self._drain_future.result, 5.0
                )
            except (concurrent.futures.TimeoutError, Exception):
                pass
            self._drain_future = None
        if self._gen_executor is not None:
            self._gen_executor.shutdown(wait=False)
            self._gen_executor = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        """Submit a request and await its result."""
        t0 = time.perf_counter()
        success = False
        result: GenerationResult | None = None
        try:
            with time_region("engine.generate"):
                if self.stub:
                    result = self._stub_generate(messages, max_tokens)
                    success = True
                    return result

                # Make sure the drain loop is running.
                if self._drain_future is None:
                    await self.start()

                seq = await self._build_sequence(
                    messages, max_tokens, temperature, top_p
                )
                self.scheduler.add_request(seq)
                self._wake_event.set()

                # Await the future; the scheduler resolves it from the
                # engine thread.
                result = await seq.future
            success = True
            return result
        finally:
            metrics.record_request(
                latency_s=time.perf_counter() - t0,
                prompt_tokens=getattr(result, "prompt_tokens", 0) if result else 0,
                completion_tokens=getattr(result, "completion_tokens", 0) if result else 0,
                success=success,
            )

    # ------------------------------------------------------------------ #
    # request building
    # ------------------------------------------------------------------ #

    async def _build_sequence(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Sequence:
        assert self.tokenizer is not None
        with time_region("engine.render_prompt"):
            prompt_text = self.tokenizer.render_prompt(messages)
        with time_region("engine.encode_prompt"):
            prompt_token_ids = self._loaded.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].tolist()

        eos_ids = []
        eos = self._loaded.tokenizer.eos_token_id
        if isinstance(eos, list):
            eos_ids.extend(int(x) for x in eos)
        elif eos is not None:
            eos_ids.append(int(eos))

        loop = asyncio.get_event_loop()
        future: asyncio.Future[GenerationResult] = loop.create_future()

        seq = Sequence(
            request_id=make_request_id(),
            prompt_token_ids=prompt_token_ids,
            sampling=SamplingParams(
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                eos_token_ids=tuple(eos_ids),
            ),
            future=future,
        )
        # Stash a reference to the loop on the seq so the engine thread
        # can resolve futures via call_soon_threadsafe.
        seq._loop = loop  # type: ignore[attr-defined]
        seq._engine_self = self  # type: ignore[attr-defined]
        return seq

    # ------------------------------------------------------------------ #
    # engine thread — scheduler drain loop
    # ------------------------------------------------------------------ #

    def _drain_blocking(self) -> None:
        """Run on the dedicated engine thread. Drives the scheduler."""
        log.info("engine drain loop started")
        try:
            while not self._stop_event.is_set():
                if not self.scheduler.has_work():
                    # Sleep until a request comes in (or stop).
                    self._wake_event.wait(timeout=0.1)
                    self._wake_event.clear()
                    continue

                # Step. Returns finished sequences.
                finished = self.scheduler.step()
                self._batch_index += 1

                # Resolve futures for finished sequences.
                if finished:
                    for seq in finished:
                        self._resolve(seq)
        except Exception as e:
            log.exception("engine drain loop crashed: %s", e)
        finally:
            log.info("engine drain loop exited")

    def _resolve(self, seq: Sequence) -> None:
        """Materialize the GenerationResult and resolve the user's future."""
        assert self._loaded is not None
        if seq.status is SequenceStatus.FAILED:
            return  # exception already set

        try:
            with time_region("engine.decode_output_text"):
                text = self._loaded.tokenizer.decode(
                    seq.output_token_ids, skip_special_tokens=True
                )
            if THINK_OPEN in text:
                text = ChatTokenizer.strip_think(text)

            finish_reason = seq.status.finish_reason  # "stop" or "length"
            result = GenerationResult(
                text=text,
                output_token_ids=list(seq.output_token_ids),
                prompt_tokens=seq.prompt_len,
                completion_tokens=seq.output_len,
                finish_reason=finish_reason,
            )
        except Exception as e:
            log.exception("resolve failed for %s: %s", seq.request_id, e)
            result = e

        loop: asyncio.AbstractEventLoop | None = getattr(seq, "_loop", None)
        if loop is None:
            return
        if isinstance(result, Exception):
            loop.call_soon_threadsafe(seq.future.set_exception, result)
        else:
            loop.call_soon_threadsafe(seq.future.set_result, result)

    # ------------------------------------------------------------------ #
    # stub mode (laptop, no GPU)
    # ------------------------------------------------------------------ #

    def _stub_generate(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> GenerationResult:
        with time_region("engine.stub.render_prompt"):
            if self.tokenizer is not None:
                prompt_tokens = self.tokenizer.count_prompt_tokens(messages)
            else:
                prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)

        text = "ok"
        completion_tokens = min(max(1, len(text.split())), max_tokens)
        return GenerationResult(
            text=text,
            output_token_ids=[],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )

    # ------------------------------------------------------------------ #
    # helpers used by the server
    # ------------------------------------------------------------------ #

    @staticmethod
    def make_request_id() -> str:
        return make_request_id()

    @staticmethod
    def now_unix() -> int:
        return int(time.time())
