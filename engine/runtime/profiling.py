"""Profiling — CLI-first, torch.profiler as a deeper opt-in.

Philosophy:
- The **primary** profiling signal is live, CLI-readable stats. Always on.
  Use `time_region("name")` (or `@timed("name")`) to bracket any chunk of
  work; the stats accumulate into per-region call counts and percentiles.
  `print_region_stats()` dumps a sorted table you can `tail -f` from the
  server's stdout. This is what we look at after every change.
- `torch.profiler` is a **secondary** tool, only enabled with `--profile-torch`,
  and is reserved for drilling into a bottleneck the CLI stats already
  identified at a coarser granularity (e.g. "MoE region is 60% of step time,
  open the chrome trace to see which kernel inside it is hot").

Both layers share the same region names so a CLI hotspot maps directly to
a chrome-trace region when we do reach for the deeper tool.
"""

from __future__ import annotations

import bisect
import contextlib
import functools
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator


# --------------------------------------------------------------------------- #
# Layer 1: lightweight CLI region timer (always on)
# --------------------------------------------------------------------------- #


@dataclass
class _RegionStats:
    name: str
    count: int = 0
    total_s: float = 0.0
    samples: list[float] = field(default_factory=list)  # bounded reservoir
    capacity: int = 1024

    def add(self, dt: float) -> None:
        self.count += 1
        self.total_s += dt
        if len(self.samples) >= self.capacity:
            self.samples.pop(0)
        bisect.insort(self.samples, dt)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        idx = min(len(self.samples) - 1, int(round((p / 100) * (len(self.samples) - 1))))
        return self.samples[idx]


class RegionTimer:
    """Process-wide region timer. Use the module-level `timer` singleton."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._regions: dict[str, _RegionStats] = {}

    def add(self, name: str, dt: float) -> None:
        with self._lock:
            stats = self._regions.get(name)
            if stats is None:
                stats = _RegionStats(name=name)
                self._regions[name] = stats
            stats.add(dt)

    def reset(self) -> None:
        with self._lock:
            self._regions.clear()

    def snapshot(self) -> list[_RegionStats]:
        with self._lock:
            return [
                _RegionStats(
                    name=s.name,
                    count=s.count,
                    total_s=s.total_s,
                    samples=list(s.samples),
                    capacity=s.capacity,
                )
                for s in self._regions.values()
            ]

    def format_table(self, sort_by: str = "total") -> str:
        snap = self.snapshot()
        if not snap:
            return "[regions] (no samples yet)"
        if sort_by == "total":
            snap.sort(key=lambda s: s.total_s, reverse=True)
        elif sort_by == "p50":
            snap.sort(key=lambda s: s.percentile(50), reverse=True)
        elif sort_by == "count":
            snap.sort(key=lambda s: s.count, reverse=True)

        header = f"{'region':<32} {'n':>8} {'total_s':>10} {'mean_ms':>10} {'p50_ms':>10} {'p99_ms':>10}"
        lines = [header, "-" * len(header)]
        for s in snap:
            mean_ms = (s.total_s / s.count * 1000) if s.count else 0.0
            lines.append(
                f"{s.name:<32} {s.count:>8} {s.total_s:>10.3f} "
                f"{mean_ms:>10.3f} {s.percentile(50)*1000:>10.3f} {s.percentile(99)*1000:>10.3f}"
            )
        return "\n".join(lines)


timer = RegionTimer()


@contextlib.contextmanager
def time_region(name: str) -> Iterator[None]:
    """Bracket a region of code; the elapsed time is added to the timer.

    Wraps `record_function` for free if torch.profiler is also active, so the
    same name shows up in chrome traces. CUDA syncing is the caller's
    responsibility — for GPU-side timing we'll layer a CUDA-event variant on
    top once Phase 1 lands.
    """
    rf = _maybe_record_function(name)
    t0 = time.perf_counter()
    try:
        with rf:
            yield
    finally:
        timer.add(name, time.perf_counter() - t0)


def timed(name: str | None = None):
    """Decorator form of `time_region`."""

    def decorator(fn):
        region = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with time_region(region):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def print_region_stats(sort_by: str = "total") -> None:
    print(timer.format_table(sort_by=sort_by), flush=True)


def reset_region_stats() -> None:
    timer.reset()


# --------------------------------------------------------------------------- #
# Layer 2: torch.profiler (opt-in, only for deep dives)
# --------------------------------------------------------------------------- #


_torch_profile_enabled = False
_torch_profile_tag: str = "run"
_PROFILE_DIR = os.environ.get("PROFILE_DIR", "profiles")


def _maybe_record_function(name: str):
    """Return a no-op context manager unless torch.profiler is active."""
    if not _torch_profile_enabled:
        return contextlib.nullcontext()
    try:
        import torch  # local import: keeps the laptop-stub path torch-free

        return torch.profiler.record_function(name)
    except Exception:
        return contextlib.nullcontext()


def enable_torch_profiler(tag: str = "run") -> None:
    """Arm torch.profiler so `record_function` regions are emitted.

    The actual capture is one-shot in the engine: see
    `Engine._maybe_capture_torch_profile`. We just flip the flag here so
    the time_region wrapper also pushes named ranges into the profile.
    """
    global _torch_profile_enabled, _torch_profile_tag
    _torch_profile_enabled = True
    _torch_profile_tag = tag
    os.makedirs(_PROFILE_DIR, exist_ok=True)


def torch_profiler_enabled() -> bool:
    return _torch_profile_enabled


def torch_profiler_tag() -> str:
    return _torch_profile_tag


def export_torch_profile(prof, tag: str, extra_meta: dict | None = None) -> tuple[str, str, list[dict]]:
    """Dump a finished torch.profiler.profile to disk.

    Returns (chrome_trace_path, summary_path, top_kernels).

    `top_kernels` is a list of `{name, self_cuda_us, self_cpu_us, count}`
    dicts for the top 30 ops by self CUDA time, suitable for embedding
    in STATUS.md.
    """
    import datetime as _dt
    import json

    os.makedirs(_PROFILE_DIR, exist_ok=True)
    ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    chrome_path = os.path.join(_PROFILE_DIR, f"torch_{tag}_{ts}.json.gz")
    summary_path = os.path.join(_PROFILE_DIR, f"torch_{tag}_{ts}.txt")
    json_path = os.path.join(_PROFILE_DIR, f"torch_{tag}_{ts}.summary.json")

    # Chrome trace
    try:
        prof.export_chrome_trace(chrome_path)
    except Exception as e:
        chrome_path = ""
        print(f"[profile] chrome trace export failed: {e}")

    # Sorted summary table — what humans read first
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
    with open(summary_path, "w") as f:
        if extra_meta:
            for k, v in extra_meta.items():
                f.write(f"# {k}: {v}\n")
            f.write("\n")
        f.write(table)

    # Structured top-N for STATUS.md
    top_kernels: list[dict] = []
    for ev in prof.key_averages():
        try:
            self_cuda = float(ev.self_cuda_time_total)
            self_cpu = float(ev.self_cpu_time_total)
        except Exception:
            self_cuda = 0.0
            self_cpu = 0.0
        top_kernels.append({
            "name": str(ev.key),
            "self_cuda_us": self_cuda,
            "self_cpu_us": self_cpu,
            "count": int(getattr(ev, "count", 0)),
        })
    top_kernels.sort(key=lambda d: d["self_cuda_us"], reverse=True)
    top_kernels = top_kernels[:30]

    with open(json_path, "w") as f:
        json.dump({"top_kernels": top_kernels, "meta": extra_meta or {}}, f, indent=2)

    return chrome_path, summary_path, top_kernels
