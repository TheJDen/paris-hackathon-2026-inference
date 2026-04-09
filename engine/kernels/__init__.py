"""Custom GPU kernels for the Phase 2 inference engine.

Contains Helion / Triton rewrites of hot paths identified by the Phase 1
torch profile. Currently:

- ``delta_rule`` — Helion rewrite of fla's ``chunk_gated_delta_rule``
  (the ``ChunkGatedDeltaRuleFunction`` hotspot, ~50ms in the Phase 1 profile).
"""

from engine.kernels.delta_rule import (
    chunk_gated_delta_rule,
    install_delta_rule_monkeypatch,
)

__all__ = [
    "chunk_gated_delta_rule",
    "install_delta_rule_monkeypatch",
]
