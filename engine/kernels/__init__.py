"""Custom GPU kernels for the Phase 2 inference engine.

Helion / Triton rewrites of hot paths identified by the Phase 1 torch
profile.

- ``delta_rule`` — Helion rewrite of fla's ``chunk_gated_delta_rule``
  (the ``ChunkGatedDeltaRuleFunction`` hotspot, ~50ms in the Phase 1 profile).
- ``moe_dispatch`` — Helion histogram + permutation that replaces the
  per-expert gather loop (~50% of CUDA self-time in the Phase 1 profile).
- ``moe_grouped_mlp`` — wrapper that runs every expert's SwiGLU MLP via
  ``torch._grouped_mm`` (or a narrow-based fallback).
- ``patch`` — runtime monkey-patch swapping HF's MoE block forward.
"""

from engine.kernels.delta_rule import (
    chunk_gated_delta_rule,
    install_delta_rule_monkeypatch,
)
from engine.kernels.moe_dispatch import moe_dispatch_indices  # noqa: F401
from engine.kernels.moe_grouped_mlp import fused_moe_forward  # noqa: F401
from engine.kernels.patch import patch_qwen3_5_moe  # noqa: F401

__all__ = [
    "chunk_gated_delta_rule",
    "install_delta_rule_monkeypatch",
    "moe_dispatch_indices",
    "fused_moe_forward",
    "patch_qwen3_5_moe",
]
