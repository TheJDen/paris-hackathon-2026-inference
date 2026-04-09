"""Continuous-batching scheduler.

Phase 2a, version 0:

  * `waiting`: deque of Sequences that have not been prefilled yet.
  * `running`: dict slot_idx → Sequence currently in decode.
  * Free slots are tracked as a set.

Each `step()` does ONE of:

  1. **Admit-and-prefill** — if any slot is free and `waiting` is non-empty,
     pop a Sequence, assign it the free slot, run `runner.prefill(...)`,
     append the first generated token, and put it in `running`.

  2. **Decode** — if no admissions are pending OR we've already done one
     prefill this step, gather all running Sequences, run
     `runner.decode(...)` over their last tokens, append the new tokens,
     check finish conditions, free finished slots.

This is the simplest correct continuous-batching loop. Notable
limitations of v0 (each addressed in a follow-up commit):

  * Prefill admits *one* sequence per step. Phase 2b admits multiple
    via prefill chunking + co-running with decode.
  * Decode batch is whatever's currently running. No bucketing.
  * Greedy sampling only. temp/top_p coming next.
  * No cross-step CUDA graphs.

The point of v0 is to **prove the architecture**: HF used only as a
weight loader / nn.Module library, no `model.generate`, our own KV cache,
our own loop. Perf wins land iteratively on top.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from dataclasses import dataclass

from engine.runtime.model_runner import ModelRunner
from engine.runtime.prefix_cache import PrefixCache
from engine.runtime.profiling import time_region
from engine.runtime.sequence import (
    GenerationResult,
    Sequence,
    SequenceStatus,
)
from engine.runtime.metrics import metrics


log = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    num_slots: int = 64
    max_seq_len: int = 4096
    # Hard caps to keep one step bounded.
    max_decode_batch: int = 64
    # How many sequences to admit and prefill in a single step. Admitting
    # more sequences per step amortises scheduler overhead over the batch and
    # fills the decode pool faster at high concurrency. Default 8 is a
    # conservative starting point; increase toward 16–32 if GPU memory allows.
    # TODO: expose via CLI / server config (currently hardcoded in SchedulerConfig).
    max_prefill_batch: int = 8
    # Soft co-scheduling knob: how many steps of decode we let happen between
    # prefill admissions, when there are sequences waiting AND decoding.
    decode_steps_per_prefill: int = 1
    # Phase 2a chunked-prefill wiring: when True, scheduler.step() fuses
    # decode + prefill into ONE runner.mixed_step() forward pass so decode
    # never stalls for an independent prefill batch. Disable to fall back
    # to the classic prefill-OR-decode path.
    use_mixed_step: bool = False  # disabled — first measurement showed 447 vs 816 baseline regression
    # When mixed_step is active, cap how many waiting sequences are admitted
    # into the same forward pass as the running decode batch. Higher = faster
    # ramp-up but slower per-step time (prefill rows pad max_L upward).
    max_prefill_admit_per_step: int = 8


class Scheduler:
    """Drives prefill + decode through the ModelRunner."""

    def __init__(self, runner: ModelRunner, config: SchedulerConfig) -> None:
        self.runner = runner
        self.config = config
        self.num_slots = config.num_slots

        self.waiting: collections.deque[Sequence] = collections.deque()
        self.running: dict[int, Sequence] = {}  # slot_idx -> Sequence
        self.free_slots: collections.deque[int] = collections.deque(range(config.num_slots))

        self._step_count = 0
        self._steps_since_prefill = 0

        # Prefix cache — set by Engine after warm_up().  None means disabled.
        self.prefix_cache: PrefixCache | None = None

        # Telemetry counters.
        self._prefix_hits: int = 0
        self._prefix_misses: int = 0

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def add_request(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def num_running(self) -> int:
        return len(self.running)

    def num_waiting(self) -> int:
        return len(self.waiting)

    # ------------------------------------------------------------------ #
    # one engine step
    # ------------------------------------------------------------------ #

    def step(self) -> list[Sequence]:
        """Run one prefill OR one decode step. Returns finished sequences."""
        with time_region("scheduler.step"):
            self._step_count += 1
            finished: list[Sequence] = []

            # Phase 2a chunked-prefill path: fuse decode + prefill into a
            # single runner.mixed_step() forward pass so decode never stalls
            # for an independent prefill batch.
            if self.config.use_mixed_step:
                try:
                    result = self._mixed_step_unified(finished)
                    if result is not None:
                        return result
                except Exception as e:
                    log.exception(
                        "mixed_step unified path failed, falling back to "
                        "legacy prefill/decode split: %s", e,
                    )
                    # Fall through to legacy path below.

            # 1. Try to admit a new sequence if a slot is free.
            should_prefill = (
                self.waiting
                and self.free_slots
                and (
                    not self.running
                    or self._steps_since_prefill >= self.config.decode_steps_per_prefill
                )
            )

            if should_prefill:
                with time_region("scheduler.admit_prefill"):
                    # Length bucketing DISABLED — small length variance was
                    # splitting sequences across buckets and starving the batch.
                    # Simple FIFO admit: take up to max_prefill_batch from the
                    # head of the waiting queue.
                    batch_seqs: list[Sequence] = []
                    max_admit = min(self.config.max_prefill_batch, len(self.free_slots))
                    for _ in range(max_admit):
                        if not self.waiting:
                            break
                        batch_seqs.append(self.waiting.popleft())

                    prefill_t0 = time.perf_counter()
                    for seq in batch_seqs:
                        slot_id = self.free_slots.popleft()
                        seq.slot_idx = slot_id
                        seq.status = SequenceStatus.PREFILLING
                        seq.prefill_started_at = prefill_t0

                    # --- Prefix caching: copy shared prefix K/V into each slot ---
                    # For sequences whose prompt starts with the cached prefix,
                    # copy prefix K/V from the reserved slot into the new slot,
                    # then pass only the suffix tokens to prefill_batch with the
                    # appropriate position offset.
                    prefix_offsets: list[int] = [0] * len(batch_seqs)
                    suffix_token_ids: list[list[int]] = [s.prompt_token_ids for s in batch_seqs]
                    if self.prefix_cache is not None and self.prefix_cache.ready:
                        pc = self.prefix_cache
                        for i, seq in enumerate(batch_seqs):
                            if pc.matches(seq.prompt_token_ids):
                                pc.apply_to_slot(seq.slot_idx)
                                prefix_offsets[i] = pc.prefix_len
                                suffix_token_ids[i] = seq.prompt_token_ids[pc.prefix_len:]
                                self._prefix_hits += 1
                                log.debug(
                                    "prefix_cache hit: seq=%s slot=%d prefix_len=%d suffix_len=%d",
                                    seq.request_id, seq.slot_idx, pc.prefix_len, len(suffix_token_ids[i]),
                                )
                            else:
                                self._prefix_misses += 1

                    try:
                        first_tokens = self.runner.prefill_batch(
                            [s.slot_idx for s in batch_seqs],
                            suffix_token_ids,
                            [s.sampling for s in batch_seqs],
                            prefix_offsets=prefix_offsets,
                        )
                    except Exception as e:
                        log.exception(
                            "prefill_batch failed for %d seqs: %s",
                            len(batch_seqs), e,
                        )
                        for seq in batch_seqs:
                            seq.status = SequenceStatus.FAILED
                            if not seq.future.done():
                                seq.future.set_exception(e)
                            self.free_slots.append(seq.slot_idx)
                        return [s for s in batch_seqs]

                    prefill_done_t = time.perf_counter()
                    for seq, first_token in zip(batch_seqs, first_tokens):
                        seq.append_output_token(first_token)
                        seq.prefill_done_at = prefill_done_t
                        if seq.maybe_finish():
                            finished.append(seq)
                            self._free_slot(seq)
                        else:
                            seq.status = SequenceStatus.RUNNING
                            self.running[seq.slot_idx] = seq
                    self._steps_since_prefill = 0

            # 2. Decode all running sequences (one token each).
            if self.running and not should_prefill:
                with time_region("scheduler.decode_step"):
                    self._steps_since_prefill += 1
                    # Single pass over self.running: build all four lists
                    # together instead of sorted() + 3 list comprehensions +
                    # per-item dict lookups. Dict insertion order is stable
                    # and the runner keys into slot_ids[] explicitly, so
                    # ordering is fine.
                    slot_ids: list[int] = []
                    seqs: list[Sequence] = []
                    last_tokens: list[int] = []
                    cache_lengths: list[int] = []
                    for sid, s in self.running.items():
                        slot_ids.append(sid)
                        seqs.append(s)
                        last_tokens.append(s.output_token_ids[-1])
                        cache_lengths.append(s.total_len - 1)
                    # cache_lengths is the absolute position of the next
                    # token. Right now slot has prompt + output_len tokens
                    # cached, the new token will land at index total_len-1
                    # ... wait, after prefill we already appended the first
                    # token to outputs but the cache only holds the prompt
                    # (L positions), so the new token lands at position L.
                    # Let's recompute: the cache has L_cached = prompt_len
                    # AFTER prefill, then each subsequent decode step writes
                    # the previously-sampled token AT position L_cached, then
                    # samples the next one (which we don't write yet).
                    # So when we run decode for `last_tokens[i]`, we want
                    # to write it at position prompt_len + (output_len - 1)
                    # which equals total_len - 1. That's what's above.

                    samplings = [s.sampling for s in seqs]
                    try:
                        next_tokens = self.runner.decode(
                            slot_ids, last_tokens, cache_lengths, samplings
                        )
                    except Exception as e:
                        log.exception("decode step failed: %s", e)
                        for s in seqs:
                            s.status = SequenceStatus.FAILED
                            if not s.future.done():
                                s.future.set_exception(e)
                            finished.append(s)
                            self._free_slot(s)
                        return finished

                    for s, tok in zip(seqs, next_tokens):
                        s.append_output_token(tok)
                        if s.maybe_finish():
                            finished.append(s)
                            self._free_slot(s)

            metrics.record_batch(
                running=len(self.running),
                waiting=len(self.waiting),
                batch_size=max(1, len(self.running)),
            )

            return finished

    # ------------------------------------------------------------------ #
    # Phase 2a: chunked-prefill unified step
    # ------------------------------------------------------------------ #

    def _mixed_step_unified(self, finished: list[Sequence]) -> list[Sequence] | None:
        """One forward pass that fuses running decode with newly-admitted prefills.

        Returns the finished list on success (caller returns it immediately).
        Returns None if there's nothing to do this step so the caller can
        fall through. Raises on runner failure — caller catches and falls
        back to the legacy prefill/decode split path.
        """
        cfg = self.config

        # --- Pick decode rows: everything currently running. ---
        decode_slot_ids_order = sorted(self.running.keys())
        decode_seqs: list[Sequence] = [self.running[s] for s in decode_slot_ids_order]
        D = len(decode_seqs)

        # --- Decide how many waiting seqs to admit into THIS forward. ---
        # Heuristic: if the running decode batch is already hot (>= 75% of
        # max_decode_batch), skip prefill admission — admitting now would
        # balloon max_L and drag the decode batch. Prefill fires next step.
        admit_budget = 0
        if self.waiting and self.free_slots:
            batch_hot = D >= int(cfg.max_decode_batch * 0.75)
            if not batch_hot:
                admit_budget = min(
                    cfg.max_prefill_admit_per_step,
                    len(self.free_slots),
                    len(self.waiting),
                )

        # Nothing at all to do.
        if D == 0 and admit_budget == 0:
            return None

        # --- Admit prefill seqs (mirror legacy admit_prefill bookkeeping). ---
        batch_seqs: list[Sequence] = []
        if admit_budget > 0:
            with time_region("scheduler.admit_prefill"):
                for _ in range(admit_budget):
                    if not self.waiting:
                        break
                    batch_seqs.append(self.waiting.popleft())
                prefill_t0 = time.perf_counter()
                for seq in batch_seqs:
                    slot_id = self.free_slots.popleft()
                    seq.slot_idx = slot_id
                    seq.status = SequenceStatus.PREFILLING
                    seq.prefill_started_at = prefill_t0

        # --- Prefix cache: copy shared prefix K/V into each admitted slot. ---
        prefill_offsets: list[int] = [0] * len(batch_seqs)
        suffix_token_ids: list[list[int]] = [s.prompt_token_ids for s in batch_seqs]
        if batch_seqs and self.prefix_cache is not None and self.prefix_cache.ready:
            pc = self.prefix_cache
            for i, seq in enumerate(batch_seqs):
                if pc.matches(seq.prompt_token_ids):
                    pc.apply_to_slot(seq.slot_idx)
                    prefill_offsets[i] = pc.prefix_len
                    suffix_token_ids[i] = seq.prompt_token_ids[pc.prefix_len:]
                    self._prefix_hits += 1
                    log.debug(
                        "prefix_cache hit: seq=%s slot=%d prefix_len=%d suffix_len=%d",
                        seq.request_id, seq.slot_idx, pc.prefix_len, len(suffix_token_ids[i]),
                    )
                else:
                    self._prefix_misses += 1

        # --- Build decode-side inputs. ---
        decode_slot_ids = list(decode_slot_ids_order)
        decode_last_tokens = [s.output_token_ids[-1] for s in decode_seqs]
        decode_cache_lengths = [s.total_len - 1 for s in decode_seqs]
        decode_samplings = [s.sampling for s in decode_seqs]

        prefill_slot_ids = [s.slot_idx for s in batch_seqs]
        prefill_samplings = [s.sampling for s in batch_seqs]

        # --- Fused forward pass. ---
        try:
            with time_region("scheduler.mixed_step"):
                decode_next, prefill_first = self.runner.mixed_step(
                    decode_slot_ids=decode_slot_ids,
                    decode_last_tokens=decode_last_tokens,
                    decode_cache_lengths=decode_cache_lengths,
                    decode_samplings=decode_samplings,
                    prefill_slot_ids=prefill_slot_ids,
                    prefill_token_ids_list=suffix_token_ids,
                    prefill_samplings=prefill_samplings,
                    prefill_offsets=prefill_offsets if batch_seqs else None,
                )
        except Exception as e:
            log.exception("runner.mixed_step failed (D=%d P=%d): %s", D, len(batch_seqs), e)
            # Fail admitted prefill seqs cleanly (their slots were already
            # allocated, so return them to the free pool).
            for seq in batch_seqs:
                seq.status = SequenceStatus.FAILED
                if not seq.future.done():
                    seq.future.set_exception(e)
                self.free_slots.append(seq.slot_idx)
                finished.append(seq)
            # Re-raise so the caller falls back to the legacy split path
            # for the running decode batch on subsequent steps... but only
            # if we'd lose decode work. If D==0, we already drained the
            # failure above; return finished so step() doesn't double-run.
            if D == 0:
                return finished
            raise

        # --- Post-process decode rows. ---
        for s, tok in zip(decode_seqs, decode_next):
            s.append_output_token(tok)
            if s.maybe_finish():
                finished.append(s)
                self._free_slot(s)

        # --- Post-process freshly-prefilled rows. ---
        if batch_seqs:
            prefill_done_t = time.perf_counter()
            for seq, first_token in zip(batch_seqs, prefill_first):
                seq.append_output_token(first_token)
                seq.prefill_done_at = prefill_done_t
                if seq.maybe_finish():
                    finished.append(seq)
                    self._free_slot(seq)
                else:
                    seq.status = SequenceStatus.RUNNING
                    self.running[seq.slot_idx] = seq
            self._steps_since_prefill = 0
        else:
            self._steps_since_prefill += 1

        metrics.record_batch(
            running=len(self.running),
            waiting=len(self.waiting),
            batch_size=max(1, len(self.running)),
        )
        return finished

    def _free_slot(self, seq: Sequence) -> None:
        slot_id = seq.slot_idx
        if slot_id in self.running:
            del self.running[slot_id]
        self.runner.cache.reset_slot(slot_id)
        self.free_slots.append(slot_id)

    # ------------------------------------------------------------------ #
    # async driver — runs on the dedicated engine thread via run_in_executor
    # ------------------------------------------------------------------ #

    def drain_loop_blocking(self, stop_event: "asyncio.Event | None" = None) -> None:
        """Run step() in a tight loop until everything is done. Blocking.

        Designed to be called from the engine's single-thread executor so
        torch.profiler thread-affinity is happy.
        """
        # Caller is responsible for adding requests / waking the loop.
        # Loop just runs until there's no work AND the stop event is set.
        while True:
            if not self.has_work():
                if stop_event is not None and stop_event.is_set():
                    return
                # Brief yield so the asyncio thread can hand us new requests.
                time.sleep(0.001)
                continue
            self.step()
