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


def annotate_hf_model_for_profiling(model, *, max_layers: int | None = None) -> int:
    """Hook every decoder layer + its attention / mlp submodule with
    `record_function` ranges so the chrome trace has named slices teammates
    can navigate by phase instead of staring at thousands of `aten::*` ops.

    The HF Qwen3_5MoeDecoderLayer pattern is `[L,L,L,F]` * 10 — full
    attention every fourth layer, linear attention (Gated DeltaNet) the
    rest. We name slices `model.layer_<i>.<full_attention|linear_attention>`
    + `model.layer_<i>.self_attn` (full layers) or `model.layer_<i>.linear_attn`
    (linear layers) + `model.layer_<i>.mlp` so Perfetto's "expand" view
    walks the model layer-by-layer.

    Returns the number of hooks installed. Safe to call multiple times —
    duplicate hooks are not a correctness issue but you'll see noisy trace
    output, so we install only once via a marker attribute.
    """
    if not _torch_profile_enabled:
        # Hooks have a small CPU cost even when the profiler isn't active,
        # because they push/pop record_function each forward. Skip if not
        # armed; the engine should re-call this if profiling is later armed.
        return 0

    try:
        import torch  # noqa: F401
        from torch.profiler import record_function
    except Exception:
        return 0

    # The inner Qwen3_5MoeTextModel lives at `model.model` for the CausalLM
    # wrapper. Some loaders return the inner model directly.
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0

    if getattr(inner, "_engine_profile_hooks_installed", False):
        return 0

    layer_types = []
    cfg = getattr(inner, "config", None)
    if cfg is not None:
        layer_types = list(getattr(cfg, "layer_types", []) or [])

    def _make_hooks(name: str):
        rf_box: dict[str, object] = {}

        def pre(_module, _inputs):
            ctx = record_function(name)
            ctx.__enter__()
            rf_box["ctx"] = ctx

        def post(_module, _inputs, _output):
            ctx = rf_box.pop("ctx", None)
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass

        return pre, post

    installed = 0
    n_layers = len(layers) if max_layers is None else min(max_layers, len(layers))
    for i in range(n_layers):
        layer = layers[i]
        layer_kind = layer_types[i] if i < len(layer_types) else "layer"

        # Outer decoder-layer slice
        pre, post = _make_hooks(f"model.layer_{i:02d}.{layer_kind}")
        layer.register_forward_pre_hook(pre)
        layer.register_forward_hook(post)
        installed += 2

        # Attention submodule (either full self_attn or linear gated delta net)
        for attn_attr in ("self_attn", "linear_attn", "delta_net", "gated_delta_net"):
            sub = getattr(layer, attn_attr, None)
            if sub is not None:
                pre2, post2 = _make_hooks(f"model.layer_{i:02d}.{attn_attr}")
                sub.register_forward_pre_hook(pre2)
                sub.register_forward_hook(post2)
                installed += 2
                break

        # MoE / mlp block
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            pre3, post3 = _make_hooks(f"model.layer_{i:02d}.mlp")
            mlp.register_forward_pre_hook(pre3)
            mlp.register_forward_hook(post3)
            installed += 2

            # If the MoE block exposes an experts container, annotate that too
            for moe_attr in ("experts", "shared_expert", "gate"):
                sub = getattr(mlp, moe_attr, None)
                if sub is not None:
                    pre4, post4 = _make_hooks(f"model.layer_{i:02d}.mlp.{moe_attr}")
                    sub.register_forward_pre_hook(pre4)
                    sub.register_forward_hook(post4)
                    installed += 2

    # Annotate embed + lm_head + norm too
    for top_name in ("embed_tokens", "norm"):
        sub = getattr(inner, top_name, None)
        if sub is not None:
            pre5, post5 = _make_hooks(f"model.{top_name}")
            sub.register_forward_pre_hook(pre5)
            sub.register_forward_hook(post5)
            installed += 2
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        pre6, post6 = _make_hooks("model.lm_head")
        lm_head.register_forward_pre_hook(pre6)
        lm_head.register_forward_hook(post6)
        installed += 2

    inner._engine_profile_hooks_installed = True
    return installed


def _git_short_sha() -> str:
    """Best-effort git SHA without depending on subprocess.run on hot paths."""
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        head_path = os.path.join(repo_root, ".git", "HEAD")
        if not os.path.exists(head_path):
            return "nogit"
        with open(head_path) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref_path = os.path.join(repo_root, ".git", head[5:])
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    return f.read().strip()[:7]
            return "nogit"
        return head[:7]
    except Exception:
        return "nogit"


def export_torch_profile(prof, tag: str, extra_meta: dict | None = None) -> tuple[str, str, list[dict]]:
    """Dump a finished torch.profiler.profile to disk.

    Filename convention is **self-describing** so teammates can compare
    profiles in Perfetto without opening each one to figure out what
    it is:

        torch_<tag>_b<batch_size>_n<captured_new_tokens>_<sha>_<ts>.{json.gz,txt,summary.json}

    Example: `torch_phase1_microbatch_b16_n32_8e3ea09_20260409-123238.json.gz`

    Reads `extra_meta["batch_size"]` and `extra_meta["captured_max_new_tokens"]`
    if present; falls back to `?` markers if not. Writes the same meta as
    a header into the .txt summary so the file is self-describing too.

    Returns `(chrome_trace_path, summary_path, top_kernels)`.

    `top_kernels` is a list of `{name, self_cuda_us, self_cpu_us, count}`
    dicts for the top 30 ops by self CUDA time, suitable for embedding
    in STATUS.md.
    """
    import datetime as _dt
    import json

    os.makedirs(_PROFILE_DIR, exist_ok=True)
    ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    sha = _git_short_sha()
    meta = extra_meta or {}
    bs = meta.get("batch_size", "?")
    nn = meta.get("captured_max_new_tokens", meta.get("max_new_tokens", "?"))

    stem = f"torch_{tag}_b{bs}_n{nn}_{sha}_{ts}"
    chrome_path = os.path.join(_PROFILE_DIR, f"{stem}.json.gz")
    summary_path = os.path.join(_PROFILE_DIR, f"{stem}.txt")
    json_path = os.path.join(_PROFILE_DIR, f"{stem}.summary.json")

    # Chrome trace
    try:
        prof.export_chrome_trace(chrome_path)
    except Exception as e:
        chrome_path = ""
        print(f"[profile] chrome trace export failed: {e}")

    # Sorted summary table — what humans read first.
    # torch >= 2.11 prefers `self_device_time_total`; fall back to the old
    # name on older versions.
    try:
        table = prof.key_averages().table(
            sort_by="self_device_time_total",
            row_limit=30,
        )
    except Exception:
        table = prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=30,
        )
    with open(summary_path, "w") as f:
        f.write(f"# torch.profiler summary\n")
        f.write(f"# tag: {tag}\n")
        f.write(f"# git_sha: {sha}\n")
        f.write(f"# captured_at_utc: {ts}\n")
        if extra_meta:
            for k, v in extra_meta.items():
                f.write(f"# {k}: {v}\n")
        f.write(f"# chrome_trace: {os.path.basename(chrome_path)}\n")
        f.write("#\n")
        f.write("# sort: self_cuda_time_total\n\n")
        f.write(table)

    # Structured top-N for STATUS.md.
    # torch >= 2.11 renamed `self_cuda_time_total` -> `self_device_time_total`
    # to be backend-agnostic. Read whichever is present.
    top_kernels: list[dict] = []
    for ev in prof.key_averages():
        self_cuda = 0.0
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            try:
                v = getattr(ev, attr, 0.0)
                if v:
                    self_cuda = float(v)
                    break
            except Exception:
                continue
        try:
            self_cpu = float(ev.self_cpu_time_total)
        except Exception:
            self_cpu = 0.0
        top_kernels.append({
            "name": str(ev.key),
            "self_cuda_us": self_cuda,
            "self_cpu_us": self_cpu,
            "count": int(getattr(ev, "count", 0)),
        })
    top_kernels.sort(key=lambda d: d["self_cuda_us"], reverse=True)
    top_kernels = top_kernels[:30]

    full_meta = {
        "tag": tag,
        "git_sha": sha,
        "captured_at_utc": ts,
        "chrome_trace": os.path.basename(chrome_path),
        **(extra_meta or {}),
    }
    with open(json_path, "w") as f:
        json.dump({"meta": full_meta, "top_kernels": top_kernels}, f, indent=2)

    return chrome_path, summary_path, top_kernels
