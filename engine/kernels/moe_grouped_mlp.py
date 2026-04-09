"""Grouped SwiGLU MLP for MoE experts.

This is the compute half of the fused MoE dispatch. It consumes the
permutation indices produced by ``moe_dispatch.moe_dispatch_indices`` and
runs every expert's SwiGLU MLP in a single grouped matmul (when available)
or a tight narrow()-based loop (always available, already much faster than
HF because the input is pre-permuted into a contiguous layout — no more
boolean-mask gathers per expert).

HF weight layout (``Qwen3_5MoeExperts``):

    gate_up_proj : [E, 2*I, H]   — used with ``F.linear(x, W)`` → x @ W.T
    down_proj    : [E, H,   I]   — same convention

``torch._grouped_mm`` wants ``b`` shaped ``[E, K, N]`` so we transpose once
at patch time and cache the transposed parameters as ``_fused_b_gate_up``
and ``_fused_b_down`` on the ``Qwen3_5MoeExperts`` module.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from engine.kernels.moe_dispatch import moe_dispatch_indices

log = logging.getLogger(__name__)


_HAS_GROUPED_MM = hasattr(torch, "_grouped_mm")


def _ensure_fused_weights(experts_module: torch.nn.Module) -> None:
    """Cache the transposed expert weights for torch._grouped_mm.

    Runs once per module (idempotent). We copy rather than view so that the
    contiguous layout matches what ``_grouped_mm`` expects. This doubles the
    GPU memory for expert weights briefly, but they're bf16 and we drop the
    originals if the caller opts in. For safety we keep both so model state
    dict loading still works.
    """
    if getattr(experts_module, "_fused_ready", False):
        return
    with torch.no_grad():
        gate_up = experts_module.gate_up_proj  # [E, 2*I, H]
        down = experts_module.down_proj  # [E, H, I]
        # Transpose last two dims -> [E, H, 2*I] and [E, I, H]
        experts_module._fused_b_gate_up = gate_up.transpose(-1, -2).contiguous()
        experts_module._fused_b_down = down.transpose(-1, -2).contiguous()
    experts_module._fused_ready = True


def _grouped_swiglu(
    x_permuted: torch.Tensor,  # [N, H] contiguous, N = T*K
    b_gate_up: torch.Tensor,  # [E, H, 2I]
    b_down: torch.Tensor,  # [E, I, H]
    offsets_i64: torch.Tensor,  # [E+1] int64
    act_fn,
) -> torch.Tensor:
    """Run SwiGLU for every expert group in one grouped matmul each.

    ``_grouped_mm`` with ``offs`` wants offsets of shape [E] representing the
    exclusive prefix sum ends (i.e. offsets_i64[1:]).
    """
    # torch._grouped_mm expects int32 offsets (verified empirically against
    # torch 2.11). Use the exclusive prefix-sum ends (offsets_i64[1:]).
    offs = offsets_i64[1:].to(torch.int32).contiguous()
    if _HAS_GROUPED_MM:
        try:
            gate_up = torch._grouped_mm(x_permuted, b_gate_up, offs=offs)  # [N, 2I]
            gate, up = gate_up.chunk(2, dim=-1)
            inter = act_fn(gate) * up  # [N, I]
            out = torch._grouped_mm(inter, b_down, offs=offs)  # [N, H]
            return out
        except Exception as e:  # pragma: no cover
            log.warning("torch._grouped_mm failed (%s); falling back to narrow loop", e)

    # Fallback: narrow()-based loop. Still fast because the input is already
    # permuted into contiguous per-expert runs — no boolean mask gather.
    E = b_gate_up.shape[0]
    H = x_permuted.shape[-1]
    out = torch.empty_like(x_permuted)
    # offsets_i64 is [E+1]; we can read it on CPU once (small, 257 entries).
    offs_cpu = offsets_i64.detach().to("cpu", non_blocking=False).tolist()
    for e in range(E):
        lo, hi = offs_cpu[e], offs_cpu[e + 1]
        n_e = hi - lo
        if n_e == 0:
            continue
        x_e = x_permuted.narrow(0, lo, n_e)
        gate_up = x_e @ b_gate_up[e]  # [n_e, 2I]
        gate, up = gate_up.chunk(2, dim=-1)
        inter = act_fn(gate) * up
        out.narrow(0, lo, n_e).copy_(inter @ b_down[e])
    return out


def fused_experts_forward(
    experts_module: torch.nn.Module,
    hidden_states: torch.Tensor,  # [T, H]
    top_k_index: torch.Tensor,  # [T, K]
    top_k_weights: torch.Tensor,  # [T, K]
) -> torch.Tensor:
    """Replacement for ``Qwen3_5MoeExperts.forward``.

    Same signature and return shape. The math is identical — we just avoid
    the per-expert Python loop with its 2*E boolean-mask gathers.
    """
    _ensure_fused_weights(experts_module)
    num_experts = experts_module.num_experts
    T, H = hidden_states.shape
    K = top_k_index.shape[-1]

    sorted_tok, sorted_slot, offsets = moe_dispatch_indices(top_k_index, num_experts)

    # Gather the tokens into per-expert contiguous runs. This is ONE
    # gather on [T*K] entries, replacing E=256 per-expert gathers in HF.
    # sorted_tok is int32; index_select needs int64 on some torch versions.
    sorted_tok_i64 = sorted_tok.to(torch.int64)
    x_permuted = hidden_states.index_select(0, sorted_tok_i64)  # [T*K, H]

    y_permuted = _grouped_swiglu(
        x_permuted,
        experts_module._fused_b_gate_up,
        experts_module._fused_b_down,
        offsets,
        experts_module.act_fn,
    )

    # Scale by routing weights. top_k_weights is [T, K]; we need it gathered
    # in the same order as sorted_tok/sorted_slot.
    sorted_slot_i64 = sorted_slot.to(torch.int64)
    w = top_k_weights[sorted_tok_i64, sorted_slot_i64].to(y_permuted.dtype)  # [T*K]
    y_permuted = y_permuted * w.unsqueeze(-1)

    # Scatter-add back to the dense output.
    final = torch.zeros_like(hidden_states)
    final.index_add_(0, sorted_tok_i64, y_permuted)
    return final


def fused_moe_forward(moe_block: torch.nn.Module, hidden_states: torch.Tensor):
    """Replacement for ``Qwen3_5MoeSparseMoeBlock.forward``.

    Mirrors the HF implementation but routes the experts call through the
    fused dispatch path.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hs = hidden_states.view(-1, hidden_dim)

    shared = moe_block.shared_expert(hs)

    _, routing_weights, selected_experts = moe_block.gate(hs)

    expert_output = fused_experts_forward(
        moe_block.experts, hs, selected_experts, routing_weights
    )

    shared = F.sigmoid(moe_block.shared_expert_gate(hs)) * shared
    out = expert_output + shared
    return out.reshape(batch_size, sequence_length, hidden_dim)
