"""Grouped SwiGLU MLP for MoE experts.

This is the compute half of the fused MoE dispatch. It consumes the
permutation indices produced by ``moe_dispatch.moe_dispatch_indices`` and
runs every expert's SwiGLU MLP in a single grouped matmul (when available)
or a tight narrow()-based loop (always available, already much faster than
HF because the input is pre-permuted into a contiguous layout — no more
boolean-mask gathers per expert).

HF weight layout (``Qwen3_5MoeExperts``):

    gate_up_proj : Parameter [E, 2*I, H]   used with F.linear(x, W) = x @ W.T
    down_proj    : Parameter [E, H,   I]   same convention

``torch._grouped_mm`` wants B shaped ``[E, K, N]`` so we transpose once at
first-call time and cache the transposed tensors as ``_fused_b_gate_up`` and
``_fused_b_down`` on the ``Qwen3_5MoeExperts`` module.

Return contract
---------------
The HF ``Qwen3_5MoeSparseMoeBlock.forward`` returns a single Tensor with
shape ``[batch, seq, hidden]`` (see transformers >= 5.5). We match that
exactly. The HF decoder layer does ``if isinstance(x, tuple): x, _ = x``
so returning a plain tensor is correct.

Escape hatch
------------
If anything inside the fused path raises, ``patch._patched_forward`` catches
it and falls back to the original HF forward, so a kernel bug can never
crash the engine.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from engine.kernels.moe_dispatch import moe_dispatch_indices

log = logging.getLogger(__name__)


_HAS_GROUPED_MM = hasattr(torch, "_grouped_mm")


def _ensure_fused_weights(experts_module: torch.nn.Module) -> None:
    """Cache transposed expert weights for torch._grouped_mm. Idempotent."""
    if getattr(experts_module, "_fused_ready", False):
        return
    with torch.no_grad():
        gate_up = experts_module.gate_up_proj  # Parameter [E, 2*I, H]
        down = experts_module.down_proj  # Parameter [E, H, I]
        # Transpose last two dims -> [E, H, 2*I] and [E, I, H].
        # .contiguous() to match grouped_mm's stride expectations.
        b_gate_up = gate_up.detach().transpose(-1, -2).contiguous()
        b_down = down.detach().transpose(-1, -2).contiguous()
    # Stash as plain attributes (not Parameters / buffers) so state_dict
    # loading / saving is untouched. They're already on the right device
    # because gate_up_proj is.
    experts_module._fused_b_gate_up = b_gate_up
    experts_module._fused_b_down = b_down
    experts_module._fused_ready = True
    log.info(
        "cached fused MoE weights: b_gate_up=%s b_down=%s dtype=%s device=%s",
        tuple(b_gate_up.shape),
        tuple(b_down.shape),
        b_gate_up.dtype,
        b_gate_up.device,
    )


def _grouped_swiglu(
    x_permuted: torch.Tensor,  # [N, H] contiguous, N = T*K
    b_gate_up: torch.Tensor,  # [E, H, 2I]
    b_down: torch.Tensor,  # [E, I, H]
    offsets_i64: torch.Tensor,  # [E+1] int64
    act_fn,
) -> torch.Tensor:
    """Run SwiGLU for every expert group. Tries grouped_mm, falls back to narrow loop."""
    # torch._grouped_mm wants offsets of shape [E] representing the
    # exclusive prefix sum ends (i.e. offsets_i64[1:]) in int32.
    offs_i32 = offsets_i64[1:].to(torch.int32).contiguous()

    if _HAS_GROUPED_MM:
        try:
            gate_up = torch._grouped_mm(x_permuted, b_gate_up, offs=offs_i32)  # [N, 2I]
            gate, up = gate_up.chunk(2, dim=-1)
            inter = act_fn(gate) * up  # [N, I]
            out = torch._grouped_mm(inter, b_down, offs=offs_i32)  # [N, H]
            return out
        except Exception as e:  # pragma: no cover - torch version / alignment
            log.warning("torch._grouped_mm failed (%s); falling back to narrow loop", e)

    # Narrow-based fallback. Still fast because the input is already permuted
    # into contiguous per-expert runs — no boolean mask gather.
    E = b_gate_up.shape[0]
    out = torch.empty_like(x_permuted)
    # offsets_i64 is [E+1]; read on CPU once (small: 257 entries for Qwen3.5).
    offs_cpu = offsets_i64.detach().to("cpu").tolist()
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

    Same semantics, same output shape. Avoids the per-expert Python loop
    with its 2*E boolean-mask gathers.
    """
    _ensure_fused_weights(experts_module)
    num_experts = int(experts_module.num_experts)
    T, H = hidden_states.shape

    sorted_tok, sorted_slot, offsets = moe_dispatch_indices(top_k_index, num_experts)

    # Advanced indexing wants int64 on current torch.
    sorted_tok_i64 = sorted_tok.to(torch.int64)
    sorted_slot_i64 = sorted_slot.to(torch.int64)

    # Gather tokens into per-expert contiguous runs. One gather on [T*K]
    # entries, replacing E=256 per-expert gathers in HF.
    x_permuted = hidden_states.index_select(0, sorted_tok_i64)  # [T*K, H]

    # Make sure the grouped matmul sees the exact same dtype as the weights.
    if x_permuted.dtype != experts_module._fused_b_gate_up.dtype:
        x_permuted = x_permuted.to(experts_module._fused_b_gate_up.dtype)

    y_permuted = _grouped_swiglu(
        x_permuted,
        experts_module._fused_b_gate_up,
        experts_module._fused_b_down,
        offsets,
        experts_module.act_fn,
    )

    # Scale by routing weights. top_k_weights is [T, K]; gather in permuted
    # order via the two index vectors.
    w = top_k_weights[sorted_tok_i64, sorted_slot_i64].to(y_permuted.dtype)
    y_permuted = y_permuted * w.unsqueeze(-1)

    # Scatter-add back to the dense output.
    final = torch.zeros_like(hidden_states)
    # index_add_ needs source dtype == destination dtype.
    if y_permuted.dtype != final.dtype:
        y_permuted = y_permuted.to(final.dtype)
    final.index_add_(0, sorted_tok_i64, y_permuted)
    return final


def fused_moe_forward(moe_block: torch.nn.Module, hidden_states: torch.Tensor):
    """Replacement for ``Qwen3_5MoeSparseMoeBlock.forward``.

    Mirrors the HF implementation but routes the experts call through the
    fused dispatch path. Returns a single Tensor of the same shape as the
    input, matching HF's current return contract.
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
