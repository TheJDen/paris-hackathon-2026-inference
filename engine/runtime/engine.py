"""Top-level Engine.

Phase 0: a stub Engine that does NOT load weights — it returns canned text
so we can verify the server, OpenAI shape, chat template, and metrics
plumbing on the laptop without a GPU.

Phase 1 (current): real generate via HuggingFace `Qwen3_5MoeForCausalLM` with
**static microbatching**. A background batcher coroutine drains a request
queue, gathers up to `max_batch` requests within `batch_window_s`, left-pads
them to a common length, and runs one batched `model.generate(...)` call.
Per-request futures are resolved as the batch completes. The asyncio.Lock
of the prior version is gone — concurrent FastAPI handlers actually batch
through one model forward.

Limitations of this Phase 1 batcher (all addressed in Phase 2):
- Static (not in-flight) batching: a batch is locked when generation starts;
  late arrivals wait for the next batch.
- Generation length is `max(max_tokens)` across the batch — short outputs
  pay for the longest one. Mitigated by trimming each output at first EOS
  before returning.
- Uses HF's KV cache, not a paged cache. Memory ceiling will be the limit.
- TP=1: all batching happens on one GPU.

The `Engine.generate(...)` interface is the stable boundary the FastAPI
server depends on; later phases swap out the internals without touching it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from engine.runtime.metrics import metrics
from engine.runtime.profiling import time_region
from engine.tokenizer.chat_template import ChatTokenizer, THINK_OPEN


log = logging.getLogger(__name__)


FinishReason = Literal["stop", "length"]


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason


@dataclass
class _BatchRequest:
    """One in-flight request waiting in the batcher queue."""

    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float
    top_p: float
    future: "asyncio.Future[GenerationResult]"
    enqueued_at: float = field(default_factory=time.perf_counter)


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
        batch_window_ms: float = 5.0,
    ) -> None:
        self.model_name = model_name
        self.stub = stub
        self.tp = tp
        self.max_batch = max_batch
        self.max_model_len = max_model_len
        self.device = device
        self.attn_impl = attn_impl
        self.batch_window_s = batch_window_ms / 1000.0

        self.tokenizer: ChatTokenizer | None = None
        self._loaded = None  # LoadedModel from engine.model.qwen3_next
        self._req_queue: asyncio.Queue[_BatchRequest] | None = None
        self._batcher_task: asyncio.Task | None = None

        if not stub:
            self.tokenizer = ChatTokenizer(model_name)

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
        """Heavy load of the HF reference model. Phase 1: TP=1 only."""
        from engine.model.qwen3_next import load_model

        with time_region("engine.load_model"):
            self._loaded = load_model(
                self.model_name,
                device=self.device,
                attn_impl=self.attn_impl,
            )
        # Make sure padding works for batched generate. Many tokenizers leave
        # pad_token unset; HF generate then refuses to batch.
        tok = self._loaded.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # Left padding is required for HF generate (so the right edge of every
        # row is the position the model is decoding from).
        tok.padding_side = "left"

    async def start(self) -> None:
        """Spawn the background batcher coroutine. Idempotent."""
        if self.stub or self._batcher_task is not None:
            return
        self._req_queue = asyncio.Queue()
        self._batcher_task = asyncio.create_task(self._batcher_loop(), name="engine-batcher")

    async def stop(self) -> None:
        if self._batcher_task is not None:
            self._batcher_task.cancel()
            try:
                await self._batcher_task
            except (asyncio.CancelledError, Exception):
                pass
            self._batcher_task = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        """Submit a generation request and await the result."""
        t0 = time.perf_counter()
        success = False
        result: GenerationResult | None = None
        try:
            with time_region("engine.generate"):
                if self.stub:
                    result = self._stub_generate(messages, max_tokens)
                else:
                    if self._req_queue is None:
                        await self.start()
                    assert self._req_queue is not None
                    fut: asyncio.Future[GenerationResult] = asyncio.get_event_loop().create_future()
                    req = _BatchRequest(
                        messages=messages,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        future=fut,
                    )
                    await self._req_queue.put(req)
                    result = await fut
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
    # static microbatching
    # ------------------------------------------------------------------ #

    async def _batcher_loop(self) -> None:
        """Drain the request queue and run batched generates."""
        assert self._req_queue is not None
        log.info(
            "batcher loop started: max_batch=%d window_ms=%.1f",
            self.max_batch, self.batch_window_s * 1000,
        )
        try:
            while True:
                first = await self._req_queue.get()
                batch: list[_BatchRequest] = [first]
                deadline = time.monotonic() + self.batch_window_s
                while len(batch) < self.max_batch:
                    timeout = max(0.0, deadline - time.monotonic())
                    if timeout == 0.0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self._req_queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                    batch.append(nxt)

                # Greedy and sampled requests can't share a batch — split.
                greedy = [r for r in batch if r.temperature <= 0.0]
                sampled = [r for r in batch if r.temperature > 0.0]
                # Process greedy first (typical case for the eval), then sampled.
                if greedy:
                    await self._process_batch(greedy, do_sample=False)
                if sampled:
                    await self._process_batch(sampled, do_sample=True)
        except asyncio.CancelledError:
            log.info("batcher loop cancelled")
            raise
        except Exception as e:
            log.exception("batcher loop crashed: %s", e)
            raise

    async def _process_batch(self, batch: list[_BatchRequest], do_sample: bool) -> None:
        """Tokenize, pad, run one batched HF generate, route results back."""
        assert self._loaded is not None and self.tokenizer is not None
        loaded = self._loaded
        tok = loaded.tokenizer
        n = len(batch)

        try:
            with time_region("engine.batch.render"):
                rendered = [self.tokenizer.render_prompt(r.messages) for r in batch]

            with time_region("engine.batch.tokenize"):
                # left-padded batch tensor + attention_mask
                enc = tok(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                input_ids = enc["input_ids"].to(loaded.device)
                attention_mask = enc["attention_mask"].to(loaded.device)

            # Per-request prompt length = number of non-pad tokens on each row.
            prompt_lens = attention_mask.sum(dim=-1).tolist()
            # Batch generation length = the largest requested max_tokens.
            max_new = max(r.max_tokens for r in batch)
            padded_prompt_len = int(input_ids.shape[-1])

            gen_kwargs: dict[str, object] = {
                "max_new_tokens": int(max_new),
                "do_sample": do_sample,
                "pad_token_id": tok.pad_token_id,
                "eos_token_id": tok.eos_token_id,
            }
            if do_sample:
                # All sampled requests in a batch share the first request's
                # temperature/top_p — Phase 1 doesn't try to mix sampling
                # configs in one batch. The eval uses temperature=0 anyway.
                gen_kwargs["temperature"] = float(batch[0].temperature)
                gen_kwargs["top_p"] = float(batch[0].top_p)

            metrics.record_batch(
                running=n,
                waiting=self._req_queue.qsize() if self._req_queue else 0,
                batch_size=n,
            )

            t0 = time.perf_counter()
            output_ids = await asyncio.to_thread(
                self._generate_blocking,
                input_ids,
                attention_mask,
                gen_kwargs,
            )
            wall_s = time.perf_counter() - t0
            metrics.record_step(wall_s, kind="prefill")

            # Slice each row's continuation, trim at first EOS, decode.
            with time_region("engine.batch.decode"):
                eos_id = tok.eos_token_id
                pad_id = tok.pad_token_id
                # eos_token_id may be a list (multi-eos models). Build a set.
                if isinstance(eos_id, (list, tuple)):
                    eos_ids = set(int(x) for x in eos_id)
                else:
                    eos_ids = {int(eos_id)} if eos_id is not None else set()
                eos_ids.add(int(pad_id))

                results: list[GenerationResult] = []
                for i, req in enumerate(batch):
                    cont = output_ids[i, padded_prompt_len:]
                    cont_ids = cont.tolist()
                    # Trim at first EOS/pad.
                    end = len(cont_ids)
                    for j, tid in enumerate(cont_ids):
                        if tid in eos_ids:
                            end = j
                            break
                    cont_trimmed = cont_ids[:end]
                    text = tok.decode(cont_trimmed, skip_special_tokens=True)
                    if THINK_OPEN in text:
                        text = ChatTokenizer.strip_think(text)
                    finish_reason: FinishReason = (
                        "length" if end >= int(req.max_tokens) else "stop"
                    )
                    results.append(
                        GenerationResult(
                            text=text,
                            prompt_tokens=int(prompt_lens[i]),
                            completion_tokens=int(end),
                            finish_reason=finish_reason,
                        )
                    )

            # Throughput accounting — the primary signal we iterate against.
            total_prompt = sum(r.prompt_tokens for r in results)
            total_completion = sum(r.completion_tokens for r in results)
            metrics.record_batch_throughput(
                batch_size=n,
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                wall_s=wall_s,
            )

            for req, res in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(res)

        except Exception as e:
            log.exception("batch processing failed: %s", e)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    def _generate_blocking(self, input_ids, attention_mask, gen_kwargs):
        """Synchronous HF generate, called from a worker thread."""
        import torch

        loaded = self._loaded
        with time_region("engine.model.generate"):
            with torch.inference_mode():
                return loaded.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

    # ------------------------------------------------------------------ #
    # stub
    # ------------------------------------------------------------------ #

    def _stub_generate(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> GenerationResult:
        """Return canned text without touching a model."""
        with time_region("engine.stub.render_prompt"):
            if self.tokenizer is not None:
                prompt_tokens = self.tokenizer.count_prompt_tokens(messages)
            else:
                prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)

        text = "ok"
        completion_tokens = min(max(1, len(text.split())), max_tokens)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )

    # ------------------------------------------------------------------ #
    # helpers used by the server
    # ------------------------------------------------------------------ #

    @staticmethod
    def make_request_id() -> str:
        return f"chatcmpl-{uuid.uuid4().hex[:24]}"

    @staticmethod
    def now_unix() -> int:
        return int(time.time())
