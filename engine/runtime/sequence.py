"""Per-request state for the continuous-batching scheduler.

A `Sequence` is the unit of work the scheduler shuffles between waiting,
running, and finished sets. It owns:

* the prompt tokens (in)
* the generated tokens so far (out)
* a slot index (assigned by the scheduler when prefill begins; it's the
  index into the slot-pool KV cache + DeltaNet state cache)
* sampling parameters
* an `asyncio.Future` the engine resolves with the final result

The Sequence object lives entirely on the CPU side. The actual KV / state
tensors live on GPU and are addressed by `slot_idx`.

This file is pure data — no torch import, no GPU work — so the scheduler
can be reasoned about without dragging in CUDA.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SequenceStatus(Enum):
    """Lifecycle of one in-flight request."""

    WAITING = "waiting"        # in the scheduler's waiting queue, no slot yet
    PREFILLING = "prefilling"  # being prefilled this step
    RUNNING = "running"        # has a slot, generating one token per decode step
    FINISHED_STOP = "finished_stop"      # hit EOS
    FINISHED_LENGTH = "finished_length"  # hit max_tokens
    FAILED = "failed"          # generation crashed somewhere

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOP,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FAILED,
        )

    @property
    def finish_reason(self) -> str:
        if self is SequenceStatus.FINISHED_LENGTH:
            return "length"
        return "stop"


@dataclass
class SamplingParams:
    """What the user asked for, distilled to what the sampler needs."""

    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    eos_token_ids: tuple[int, ...] = ()

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@dataclass
class Sequence:
    """One in-flight request."""

    request_id: str
    prompt_token_ids: list[int]
    sampling: SamplingParams
    future: "asyncio.Future"  # resolved by the scheduler with a Result

    status: SequenceStatus = SequenceStatus.WAITING
    slot_idx: int = -1                          # set when scheduler assigns a slot
    output_token_ids: list[int] = field(default_factory=list)
    enqueued_at: float = field(default_factory=time.perf_counter)
    prefill_started_at: float | None = None
    prefill_done_at: float | None = None
    finished_at: float | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def cache_seq_len(self) -> int:
        """Number of tokens whose K/V (or recurrent state) is in the cache.

        After prefill: prompt_len.
        After each decode step that produces output[i]: prompt_len + i + 1
        (because the *previous* token's state was written when generating it).
        """
        return self.prompt_len + self.output_len

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def hit_eos(self) -> bool:
        if not self.output_token_ids:
            return False
        return int(self.output_token_ids[-1]) in self.sampling.eos_token_ids

    def hit_max_tokens(self) -> bool:
        return self.output_len >= self.sampling.max_tokens

    def maybe_finish(self) -> bool:
        """Mark the sequence finished if it hit a stop condition. Returns True if so."""
        if self.hit_eos():
            self.status = SequenceStatus.FINISHED_STOP
            self.finished_at = time.perf_counter()
            return True
        if self.hit_max_tokens():
            self.status = SequenceStatus.FINISHED_LENGTH
            self.finished_at = time.perf_counter()
            return True
        return False


@dataclass
class GenerationResult:
    """What `Engine.generate(...)` resolves the caller's future with."""

    text: str
    output_token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" or "length"


def make_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"
