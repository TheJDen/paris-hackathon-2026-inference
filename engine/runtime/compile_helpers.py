"""torch.compile helpers for the inference engine.

Provides three public entry points:

  apply_dynamo_disables(model)
      Monkeypatches SlotPoolCache.update() and the linear-attention
      Qwen3_5MoeGatedDeltaNet.forward() with @torch._dynamo.disable so
      Inductor never tries to trace/fuse those regions.

  compile_model(model, mode, dynamic) -> compiled_inner
      Wraps model.model (the inner Qwen3_5MoeTextModel, NOT lm_head) with
      torch.compile.  Returns the compiled module so the caller can replace
      runner.inner_model.

  warmup_compiled(runner, max_batch, max_seq)
      Fires the compile by running one dummy prefill + one dummy decode with
      synthetic tensors before the first real request arrives.

COEXISTENCE NOTE
----------------
reduce-overhead mode tells Inductor to use CUDA graphs internally.  The
separate cuda_graphs agent (engine/runtime/cuda_graphs.py) also captures
CUDA graphs for the decode hot path.  If BOTH are active simultaneously,
they will conflict (double-graph-capture, static-buffer aliasing).

Resolution: when torch.compile is ON, the caller (ModelRunner.__init__)
should set enable_cuda_graphs=False to avoid the conflict.  This is
handled in model_runner.py with the `compile` flag guard.

DO NOT edit kv_cache.py or scheduler.py — other agents own those files.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from engine.runtime.model_runner import ModelRunner

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# dynamo disables — must run BEFORE torch.compile is called
# --------------------------------------------------------------------------- #


def apply_dynamo_disables(model: torch.nn.Module) -> None:
    """Monkeypatch cache + linear-attention forwards with dynamo.disable.

    Called once after model weights are loaded, before compile_model().
    Safe to call on CPU or CUDA; the patches are pure Python.

    Patches applied:
      1. SlotPoolCache.update()           — in-place index_copy_ writes
      2. SlotPoolCache.update_conv_state()— if it exists
      3. SlotPoolCache.update_recurrent_state() — if it exists
      4. Qwen3_5MoeGatedDeltaNet.forward() — calls Triton fla kernels
    """
    _patch_slot_pool_cache()
    _patch_linear_attention(model)


def _patch_slot_pool_cache() -> None:
    """Wrap SlotPoolCache mutating methods with dynamo.disable."""
    try:
        from engine.runtime.kv_cache import SlotPoolCache
    except ImportError:
        log.warning("compile_helpers: could not import SlotPoolCache — skipping cache patches")
        return

    _disable_method(SlotPoolCache, "update",
                    "SlotPoolCache.update (index_copy_ — not traceable)")
    for method_name in ("update_conv_state", "update_recurrent_state"):
        if hasattr(SlotPoolCache, method_name):
            _disable_method(SlotPoolCache, method_name,
                            f"SlotPoolCache.{method_name}")


def _patch_linear_attention(model: torch.nn.Module) -> None:
    """Wrap Qwen3_5MoeGatedDeltaNet.forward with dynamo.disable.

    The fla / causal_conv1d Triton kernels inside this layer cause
    "encountered Triton kernel during tracing" if Inductor tries to fuse
    them.  Disabling compile for the entire layer forward is the safest
    option until fla ships proper torch.compile support.
    """
    # Try to find the class from transformers
    GatedDeltaNet = _find_class("transformers", "Qwen3_5MoeGatedDeltaNet")
    if GatedDeltaNet is None:
        # Fall back: walk the model and find the first linear-attention layer
        GatedDeltaNet = _detect_from_model(model)

    if GatedDeltaNet is None:
        log.warning(
            "compile_helpers: could not find Qwen3_5MoeGatedDeltaNet — "
            "linear-attention forward NOT wrapped with dynamo.disable. "
            "Watch for Triton-during-tracing errors."
        )
        return

    _disable_method(GatedDeltaNet, "forward",
                    "Qwen3_5MoeGatedDeltaNet.forward (Triton fla kernels)")


def _find_class(module_name: str, class_name: str):
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name, None)
    except ImportError:
        return None


def _detect_from_model(model: torch.nn.Module):
    """Walk the model graph to find the DeltaNet layer class by name."""
    for name, module in model.named_modules():
        cls = type(module)
        if "GatedDeltaNet" in cls.__name__ or "DeltaNet" in cls.__name__:
            log.info(
                "compile_helpers: detected linear-attention class %s from model walk",
                cls.__qualname__,
            )
            return cls
    return None


def _disable_method(cls, method_name: str, label: str) -> None:
    """Replace cls.method_name with a dynamo-disabled version."""
    original = getattr(cls, method_name, None)
    if original is None:
        return
    # Don't double-wrap.
    if getattr(original, "_dynamo_disabled_by_compile_helpers", False):
        return
    disabled = torch._dynamo.disable(original)
    disabled._dynamo_disabled_by_compile_helpers = True  # type: ignore[attr-defined]
    setattr(cls, method_name, disabled)
    log.info("compile_helpers: wrapped %s with torch._dynamo.disable", label)


# --------------------------------------------------------------------------- #
# compile_model — wraps model.model (inner text model) only
# --------------------------------------------------------------------------- #


def compile_model(
    hf_model: torch.nn.Module,
    *,
    mode: str = "reduce-overhead",
    dynamic: bool = True,
) -> torch.nn.Module:
    """Wrap hf_model.model (inner text model) with torch.compile.

    Why model.model and not the full causal LM?
      - lm_head is a single large matmul; it's already efficient and doesn't
        benefit from fusion with the transformer body.
      - Keeping lm_head in eager avoids any compile-time issues with the
        very wide vocab projection (248 320 tokens).

    dynamic=True rationale:
      Batch size (B) and max_kv_seq_len vary every decode step as requests
      arrive and depart.  dynamic=True avoids a per-shape recompile.
      Trade-off: some fusion opportunities (e.g. loop unrolling over a
      fixed B) are lost.  If the scheduler ever runs with fixed batch
      buckets (e.g. always B=8 or B=16), consider switching to
      dynamic=False with explicit warmup at each bucket size.

    fullgraph=False:
      HF modeling code has Python-level control flow that Inductor cannot
      fully trace (if past_key_values is None, etc.).  fullgraph=False
      lets Inductor fall back to eager at graph breaks rather than crashing.
    """
    inner = getattr(hf_model, "model", None)
    if inner is None:
        log.warning(
            "compile_helpers: hf_model has no .model attribute — "
            "compiling the full model instead (may be slower to compile)"
        )
        inner = hf_model

    log.info(
        "compile_helpers: calling torch.compile on %s "
        "(mode=%s, fullgraph=False, dynamic=%s)",
        type(inner).__name__,
        mode,
        dynamic,
    )
    compiled = torch.compile(
        inner,
        mode=mode,
        fullgraph=False,
        dynamic=dynamic,
    )
    return compiled


# --------------------------------------------------------------------------- #
# warmup_compiled — fires the compile before the first real request
# --------------------------------------------------------------------------- #


def warmup_compiled(
    runner: "ModelRunner",
    *,
    max_batch: int = 8,
    max_seq: int = 2048,
) -> None:
    """Run dummy prefill + decode to trigger the Inductor compile.

    Without this, the FIRST real request would pay a 30-60s compile cost.
    Call this once at the end of ModelRunner.__init__ (after compile_model).

    The warmup tensors match expected prod shapes:
      - prefill: B=1, seq_len=64 (short prompt, just enough to warm the path)
      - decode:  B=max_batch, kv_len=max_seq (worst-case shape for dynamic)

    Time is logged at INFO level so operators can see the compile overhead.
    """
    import torch
    from engine.runtime.kv_cache import BatchSlots

    device = runner.device
    cfg = runner.text_config

    log.info(
        "compile_helpers: starting compile warmup "
        "(max_batch=%d, max_seq=%d) — this will take 30-90s on first run",
        max_batch,
        max_seq,
    )
    t0 = time.perf_counter()

    # Determine vocab size for dummy input clamping.
    vocab_size: int = getattr(cfg, "vocab_size", 32000)

    # ---- dummy prefill ----
    try:
        _warmup_prefill(runner, device, cfg, vocab_size)
        log.info("compile_helpers: warmup prefill done (%.1fs)", time.perf_counter() - t0)
    except Exception as e:
        log.warning("compile_helpers: warmup prefill failed (%s) — continuing", e)

    t1 = time.perf_counter()

    # ---- dummy decode ----
    try:
        _warmup_decode(runner, device, cfg, vocab_size, max_batch, max_seq)
        log.info("compile_helpers: warmup decode done (%.1fs)", time.perf_counter() - t1)
    except Exception as e:
        log.warning("compile_helpers: warmup decode failed (%s) — continuing", e)

    elapsed = time.perf_counter() - t0
    log.info("compile_helpers: compile warmup complete in %.1fs", elapsed)


def _warmup_prefill(runner, device, cfg, vocab_size: int) -> None:
    """One dummy prefill forward to warm the prefill graph."""
    from engine.runtime.kv_cache import BatchSlots

    L = 64
    slot_id = 0
    input_ids = torch.zeros(1, L, dtype=torch.long, device=device)
    position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, L, dtype=torch.long, device=device)

    slot_ids_t = torch.tensor([slot_id], dtype=torch.int64, device=device)
    write_positions = [torch.arange(L, dtype=torch.int64, device=device)]
    query_lens = torch.tensor([L], dtype=torch.int64, device=device)
    kv_seq_lens = torch.tensor([L], dtype=torch.int64, device=device)

    runner.cache.set_batch(
        BatchSlots(
            slot_ids=slot_ids_t,
            write_positions=write_positions,
            query_lens=query_lens,
            kv_seq_lens=kv_seq_lens,
            is_prefill=True,
        )
    )
    with torch.inference_mode():
        _ = runner.inner_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=runner.cache,
            use_cache=True,
        )
    runner.cache.commit_batch()


def _warmup_decode(runner, device, cfg, vocab_size: int, max_batch: int, max_seq: int) -> None:
    """One dummy decode forward to warm the decode graph."""
    from engine.runtime.kv_cache import BatchSlots

    B = max_batch
    # Use slot 0 for all dummy rows — this is warmup, correctness doesn't matter.
    slot_ids_list = list(range(min(B, runner.num_slots)))
    B = len(slot_ids_list)  # clamp to actual slot count

    cache_lengths = [max_seq - 1] * B  # pretend kv is almost full

    input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
    slot_ids_t = torch.tensor(slot_ids_list, dtype=torch.int64, device=device)
    write_positions = [
        torch.tensor([cache_lengths[i]], dtype=torch.int64, device=device)
        for i in range(B)
    ]
    query_lens = torch.ones(B, dtype=torch.int64, device=device)
    kv_seq_lens = torch.tensor(
        [cache_lengths[i] + 1 for i in range(B)],
        dtype=torch.int64, device=device,
    )
    position_ids = torch.tensor(
        [[cache_lengths[i]] for i in range(B)],
        dtype=torch.long, device=device,
    )
    max_s = int(kv_seq_lens.max().item())
    attention_mask = torch.zeros(B, max_s, dtype=torch.long, device=device)
    for b in range(B):
        attention_mask[b, : cache_lengths[b] + 1] = 1

    runner.cache.set_batch(
        BatchSlots(
            slot_ids=slot_ids_t,
            write_positions=write_positions,
            query_lens=query_lens,
            kv_seq_lens=kv_seq_lens,
            is_prefill=False,
        )
    )
    with torch.inference_mode():
        _ = runner.inner_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=runner.cache,
            use_cache=True,
        )
    runner.cache.commit_batch()
