"""Runtime monkey-patch for HF's ``Qwen3_5MoeSparseMoeBlock``.

We DO NOT edit the transformers library files. Instead we reassign the
``forward`` method on the class after import, so every existing instance
(all 40 decoder layers) picks up the fused path immediately.

The patched forward ALWAYS falls back to the original HF forward if the
fused path raises. This way a bug in the fused kernel can never crash the
engine — the worst case is we get the original (slow) MoE path and a
warning in the log.

Usage::

    from engine.kernels import patch_qwen3_5_moe
    patch_qwen3_5_moe()   # call once, before the first forward pass
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


_PATCHED = False
# Module-level switch the fused forward flips the first time it raises, so
# we don't spam the log on every subsequent layer/step.
_FUSED_DISABLED = False


def _fused_disabled() -> bool:
    return _FUSED_DISABLED or os.environ.get("PARIS_DISABLE_HELION_MOE", "0") == "1"


def _disable_fused(reason: str) -> None:
    global _FUSED_DISABLED
    if not _FUSED_DISABLED:
        log.warning("disabling fused MoE forward for the rest of the run: %s", reason)
    _FUSED_DISABLED = True


def patch_qwen3_5_moe() -> bool:
    """Replace ``Qwen3_5MoeSparseMoeBlock.forward`` with the fused version.

    Returns True on success, False if the HF class is not importable (e.g.
    transformers too old). Safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    except Exception as e:
        log.warning("cannot patch Qwen3_5MoeSparseMoeBlock: %s", e)
        return False

    from engine.kernels.moe_grouped_mlp import fused_moe_forward

    block_cls = getattr(M, "Qwen3_5MoeSparseMoeBlock", None)
    if block_cls is None:
        log.warning("Qwen3_5MoeSparseMoeBlock missing in transformers")
        return False

    original = block_cls.forward

    def _patched_forward(self, hidden_states):
        # Fast fallback: if the fused path has already been disabled (env var
        # or a prior failure) just run the original. No try/except overhead.
        if _fused_disabled():
            return original(self, hidden_states)
        try:
            return fused_moe_forward(self, hidden_states)
        except Exception as e:
            _disable_fused(f"{type(e).__name__}: {e}")
            log.exception("fused MoE forward raised; falling back to HF forward")
            return original(self, hidden_states)

    block_cls.forward = _patched_forward
    block_cls._original_forward = original  # keep a handle for unpatch/debug
    _PATCHED = True
    log.info("patched Qwen3_5MoeSparseMoeBlock.forward -> fused_moe_forward")
    return True


def unpatch_qwen3_5_moe() -> bool:
    global _PATCHED
    if not _PATCHED:
        return True
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    except Exception:
        return False
    block_cls = getattr(M, "Qwen3_5MoeSparseMoeBlock", None)
    if block_cls is None or not hasattr(block_cls, "_original_forward"):
        return False
    block_cls.forward = block_cls._original_forward
    del block_cls._original_forward
    _PATCHED = False
    return True
