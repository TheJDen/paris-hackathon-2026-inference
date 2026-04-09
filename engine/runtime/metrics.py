"""Always-on lightweight metrics for the engine.

The numbers we look at *first* before reaching for torch.profiler. The
focus is **throughput** — we're scored on weighted output tok/s across
concurrency levels, with c=64 worth 8x the score of c=1. Every counter
here exists to answer "what's our tok/s right now and what just changed
when we touched something."

What's tracked:
  * `prompt_tok_per_s` / `completion_tok_per_s` — split rather than
    summed because they have very different costs (prefill is parallel
    in the seq dim, decode is the autoregressive bottleneck).
  * Rolling-window batch tok/s — last-N-batch median + p99 so we can
    see steady-state throughput as opposed to lifetime average.
  * Batch fill ratio — how full our microbatcher is filling each batch
    out of `max_batch`. If this is < 50% the scheduler / batch_window
    is leaving throughput on the floor.
  * Batch size histogram and queue depth — same idea, surfaces
    underutilization at high concurrency.

Phase 1+ will populate the GPU memory / KV slot / MoE routing fields
once those subsystems exist.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Histogram:
    """Tiny reservoir for percentile estimation. O(log n) inserts."""

    capacity: int = 4096
    samples: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if len(self.samples) >= self.capacity:
            # Drop the oldest sample to keep size bounded.
            self.samples.pop(0)
        bisect.insort(self.samples, value)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        idx = min(len(self.samples) - 1, int(round((p / 100) * (len(self.samples) - 1))))
        return self.samples[idx]


class Metrics:
    """Process-wide metric collector. Use the module-level `metrics` singleton."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()

        # Lifetime counters
        self.requests_total = 0
        self.requests_failed = 0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0

        # Latency histograms (seconds) — kept around as a sanity signal but
        # NOT what we iterate against; throughput rules the loop.
        self.request_latency = _Histogram()
        self.step_latency = _Histogram()
        self.prefill_latency = _Histogram()
        self.decode_latency = _Histogram()

        # ---- throughput-focused state ----
        # Per-batch tok/s reservoir for rolling median / p99.
        self.batch_throughput = _Histogram(capacity=256)
        # Aggregate tokens processed in the last 60s, for "right now" tok/s.
        self._recent_tokens: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._recent_window_s = 60.0
        # max_batch ceiling so we can compute batch fill ratio.
        self.max_batch_capacity = 1
        # Per-batch fill ratio reservoir.
        self.batch_fill = _Histogram(capacity=256)

        # Concurrency snapshots (set by the scheduler each step)
        self.running_count = 0
        self.waiting_count = 0
        self.batch_size_recent: deque[int] = deque(maxlen=256)

        # GPU memory (bytes), populated by Phase 1+ workers
        self.gpu_mem_used: dict[int, int] = {}

        # Cache occupancies (Phase 2+)
        self.kv_slots_used = 0
        self.kv_slots_total = 0
        self.state_slots_used = 0
        self.state_slots_total = 0

        # MoE routing histogram, layer -> per-expert token counts (Phase 2+)
        self.moe_routing: dict[int, list[int]] = {}

        # Background flusher
        self._flusher: threading.Thread | None = None
        self._stop = threading.Event()

    # ---------- recording ----------

    def record_request(
        self,
        latency_s: float,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool,
    ) -> None:
        with self._lock:
            self.requests_total += 1
            if not success:
                self.requests_failed += 1
                return
            self.prompt_tokens_total += prompt_tokens
            self.completion_tokens_total += completion_tokens
            self.request_latency.add(latency_s)

    def record_step(self, latency_s: float, kind: str = "decode") -> None:
        with self._lock:
            self.step_latency.add(latency_s)
            if kind == "prefill":
                self.prefill_latency.add(latency_s)
            else:
                self.decode_latency.add(latency_s)

    def record_batch(self, running: int, waiting: int, batch_size: int) -> None:
        with self._lock:
            self.running_count = running
            self.waiting_count = waiting
            self.batch_size_recent.append(batch_size)

    def record_batch_throughput(
        self,
        batch_size: int,
        prompt_tokens: int,
        completion_tokens: int,
        wall_s: float,
    ) -> None:
        """Record one batched generate call's tok/s, fill ratio, and tokens.

        This is the **primary** throughput signal. The engine should call
        this once per batched forward pass with the actual token counts
        and wall time. Roll-up percentiles end up in `snapshot()`.
        """
        if wall_s <= 0:
            return
        total = max(0, int(prompt_tokens + completion_tokens))
        tps = total / wall_s
        with self._lock:
            self.batch_throughput.add(tps)
            cap = max(1, int(self.max_batch_capacity))
            self.batch_fill.add(min(1.0, batch_size / cap))
            now = time.perf_counter()
            self._recent_tokens.append((now, total))
            # Trim the rolling window.
            cutoff = now - self._recent_window_s
            while self._recent_tokens and self._recent_tokens[0][0] < cutoff:
                self._recent_tokens.popleft()

    def set_max_batch_capacity(self, cap: int) -> None:
        with self._lock:
            self.max_batch_capacity = max(1, int(cap))

    # ---------- snapshot ----------

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            elapsed = max(time.perf_counter() - self._start_time, 1e-6)
            avg_batch = (
                sum(self.batch_size_recent) / len(self.batch_size_recent)
                if self.batch_size_recent
                else 0.0
            )
            # Rolling-window tok/s
            now = time.perf_counter()
            cutoff = now - self._recent_window_s
            while self._recent_tokens and self._recent_tokens[0][0] < cutoff:
                self._recent_tokens.popleft()
            window_tokens = sum(t for _, t in self._recent_tokens)
            window_dt = (
                now - self._recent_tokens[0][0]
                if self._recent_tokens
                else 0.0
            )
            recent_tps = window_tokens / window_dt if window_dt > 0 else 0.0

            avg_fill = (
                sum(self.batch_fill.samples) / len(self.batch_fill.samples)
                if self.batch_fill.samples
                else 0.0
            )

            return {
                "uptime_s": round(elapsed, 1),
                "requests_total": self.requests_total,
                "requests_failed": self.requests_failed,
                # ---- throughput, the primary signal ----
                "tok_per_s_recent": round(recent_tps, 1),
                "tok_per_s_lifetime": round(
                    (self.prompt_tokens_total + self.completion_tokens_total) / elapsed, 1
                ),
                "prompt_tok_per_s_lifetime": round(self.prompt_tokens_total / elapsed, 1),
                "completion_tok_per_s_lifetime": round(self.completion_tokens_total / elapsed, 1),
                "batch_tok_per_s_p50": round(self.batch_throughput.percentile(50), 1),
                "batch_tok_per_s_p99": round(self.batch_throughput.percentile(99), 1),
                # ---- batching health ----
                "max_batch": self.max_batch_capacity,
                "avg_batch_size": round(avg_batch, 2),
                "avg_batch_fill": round(avg_fill, 3),
                "running": self.running_count,
                "waiting": self.waiting_count,
                # ---- latency, sanity signal only ----
                "request_p50_ms": round(self.request_latency.percentile(50) * 1000, 1),
                "request_p99_ms": round(self.request_latency.percentile(99) * 1000, 1),
                # ---- caches (populated in Phase 2+) ----
                "kv_used": self.kv_slots_used,
                "kv_total": self.kv_slots_total,
                "state_used": self.state_slots_used,
                "state_total": self.state_slots_total,
            }

    # ---------- background flusher ----------

    def start_flusher(self, interval_s: float = 5.0) -> None:
        if self._flusher is not None:
            return

        def _run() -> None:
            while not self._stop.wait(interval_s):
                snap = self.snapshot()
                line = " ".join(f"{k}={v}" for k, v in snap.items())
                print(f"[metrics] {line}", flush=True)

        self._flusher = threading.Thread(target=_run, name="metrics-flush", daemon=True)
        self._flusher.start()

    def stop_flusher(self) -> None:
        self._stop.set()


metrics = Metrics()
