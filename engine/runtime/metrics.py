"""Always-on lightweight metrics for the engine.

These are the numbers we look at *first* before reaching for torch.profiler.
Atomic counters + a background flush thread that prints a one-line summary
every `flush_interval` seconds. Cheap enough to leave on permanently.

Phase 0 wires up the basics (request counters, latency p50/p99). Phase 1+
will populate the GPU/cache/MoE-routing fields once those subsystems exist.
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

        # Counters
        self.requests_total = 0
        self.requests_failed = 0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0

        # Latency histograms (seconds)
        self.request_latency = _Histogram()
        self.step_latency = _Histogram()
        self.prefill_latency = _Histogram()
        self.decode_latency = _Histogram()

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

    # ---------- snapshot ----------

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            elapsed = max(time.perf_counter() - self._start_time, 1e-6)
            avg_batch = (
                sum(self.batch_size_recent) / len(self.batch_size_recent)
                if self.batch_size_recent
                else 0.0
            )
            return {
                "uptime_s": round(elapsed, 1),
                "requests_total": self.requests_total,
                "requests_failed": self.requests_failed,
                "prompt_tok_per_s": round(self.prompt_tokens_total / elapsed, 1),
                "completion_tok_per_s": round(self.completion_tokens_total / elapsed, 1),
                "request_p50_ms": round(self.request_latency.percentile(50) * 1000, 1),
                "request_p99_ms": round(self.request_latency.percentile(99) * 1000, 1),
                "prefill_p50_ms": round(self.prefill_latency.percentile(50) * 1000, 1),
                "decode_p50_ms": round(self.decode_latency.percentile(50) * 1000, 1),
                "running": self.running_count,
                "waiting": self.waiting_count,
                "avg_batch": round(avg_batch, 2),
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
