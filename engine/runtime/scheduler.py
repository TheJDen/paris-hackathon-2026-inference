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
    # How many sequences to admit and prefill together in one forward pass.
    max_prefill_batch: int = 32
    # Soft co-scheduling knob: how many steps of decode we let happen between
    # prefill admissions, when there are sequences waiting AND decoding.
    decode_steps_per_prefill: int = 1


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
                    n_admit = min(
                        len(self.waiting),
                        len(self.free_slots),
                        self.config.max_prefill_batch,
                    )
                    to_prefill: list[Sequence] = []
                    for _ in range(n_admit):
                        seq = self.waiting.popleft()
                        slot_id = self.free_slots.popleft()
                        seq.slot_idx = slot_id
                        seq.status = SequenceStatus.PREFILLING
                        seq.prefill_started_at = time.perf_counter()
                        to_prefill.append(seq)

                    try:
                        first_tokens = self.runner.prefill_batch(
                            [s.slot_idx for s in to_prefill],
                            [s.prompt_token_ids for s in to_prefill],
                            [s.sampling for s in to_prefill],
                        )
                    except Exception as e:
                        log.exception("prefill_batch failed (%d seqs): %s", len(to_prefill), e)
                        for seq in to_prefill:
                            seq.status = SequenceStatus.FAILED
                            if not seq.future.done():
                                seq.future.set_exception(e)
                            self.free_slots.append(seq.slot_idx)
                        return to_prefill

                    for seq, first_token in zip(to_prefill, first_tokens):
                        seq.append_output_token(first_token)
                        seq.prefill_done_at = time.perf_counter()
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
                    slot_ids = sorted(self.running.keys())
                    seqs = [self.running[s] for s in slot_ids]
                    last_tokens = [s.output_token_ids[-1] for s in seqs]
                    cache_lengths = [s.total_len - 1 for s in seqs]
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

                    try:
                        next_tokens = self.runner.decode(
                            slot_ids, last_tokens, cache_lengths,
                            [s.sampling for s in seqs],
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
