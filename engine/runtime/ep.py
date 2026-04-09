"""Minimal forward-only Expert Parallelism for Qwen3.5-35B-A3B MoE.

Strategy: weights fully replicated on every rank (no weight sharding). We
monkey-patch ``Qwen3NextSparseMoeBlock.forward`` so each rank only runs its
slice of the 256 experts, then all_reduce(SUM) the routed contribution. The
shared expert is deterministic (same weights, same inputs on every rank) so
we compute it locally on every rank and add it AFTER the all_reduce — this
avoids having to subtract it out.

Launch:
    torchrun --nproc_per_node=N -m server.main ... --ep N

All ranks load the full model. Only rank 0 binds HTTP — see server/main.py.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


log = logging.getLogger(__name__)


_EP_INITIALIZED = False
_EP_WORLD_SIZE = 1
_EP_RANK = 0


def ep_world_size() -> int:
    return _EP_WORLD_SIZE


def ep_rank() -> int:
    return _EP_RANK


def init_ep(world_size: int, rank: int, local_rank: int | None = None) -> Tuple[int, int]:
    """Initialize NCCL process group for EP. Idempotent."""
    global _EP_INITIALIZED, _EP_WORLD_SIZE, _EP_RANK
    if _EP_INITIALIZED:
        return _EP_RANK, _EP_WORLD_SIZE

    if world_size <= 1:
        _EP_INITIALIZED = True
        _EP_WORLD_SIZE = 1
        _EP_RANK = 0
        return 0, 1

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
        )

    _EP_INITIALIZED = True
    _EP_WORLD_SIZE = world_size
    _EP_RANK = rank
    log.info("EP initialized: rank=%d world_size=%d local_rank=%d", rank, world_size, local_rank)
    return rank, world_size


def expert_range(num_experts: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Contiguous slice of experts owned by ``rank``.

    Uses the ``rank * n // world_size`` trick so it works even when
    ``num_experts`` is not divisible by ``world_size``.
    """
    lo = rank * num_experts // world_size
    hi = (rank + 1) * num_experts // world_size
    return lo, hi


def _make_ep_forward(module, lo: int, hi: int):
    """Build a replacement forward closure bound to ``module``.

    Mirrors ``Qwen3NextSparseMoeBlock.forward`` but only iterates the experts
    in [lo, hi). The routed contribution is all_reduced across ranks; the
    shared expert is computed locally (deterministic) and added after.
    """

    def ep_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        # Shared expert (deterministic across ranks) ------------------
        shared_out = module.shared_expert(x)
        shared_gate = F.sigmoid(module.shared_expert_gate(x))
        shared_out = shared_gate * shared_out

        # Router ------------------------------------------------------
        _, routing_weights, selected_experts = module.gate(x)
        # routing_weights: [T, top_k], selected_experts: [T, top_k]

        # Routed contribution from MY experts only --------------------
        experts_mod = module.experts  # Qwen3NextExperts
        gate_up_proj = experts_mod.gate_up_proj  # [E, 2*I, H]
        down_proj = experts_mod.down_proj       # [E, H, I]
        act_fn = experts_mod.act_fn

        routed = torch.zeros_like(x)

        # expert_mask[e, k, t] = 1 iff token t selected expert e at slot k
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=experts_mod.num_experts
            ).permute(2, 1, 0)

        for expert_idx in range(lo, hi):
            mask_e = expert_mask[expert_idx]
            if mask_e.sum() == 0:
                continue
            top_k_pos, token_idx = torch.where(mask_e)
            if token_idx.numel() == 0:
                continue
            current_state = x[token_idx]
            gate, up = F.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            cur = act_fn(gate) * up
            cur = F.linear(cur, down_proj[expert_idx])
            cur = cur * routing_weights[token_idx, top_k_pos, None]
            routed.index_add_(0, token_idx, cur.to(routed.dtype))

        # Sum the partial routed contributions across EP ranks --------
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(routed, op=dist.ReduceOp.SUM)

        out = routed + shared_out
        return out.view(batch_size, sequence_length, hidden_dim)

    return ep_forward


def patch_moe_for_ep(model, rank: int, world_size: int) -> int:
    """Monkey-patch every Qwen3NextSparseMoeBlock on ``model`` for EP.

    Returns the number of blocks patched. No-op when world_size <= 1.
    """
    if world_size <= 1:
        return 0

    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextSparseMoeBlock,
        )
    except Exception as e:
        log.warning("could not import Qwen3NextSparseMoeBlock: %s — EP disabled", e)
        return 0

    count = 0
    for m in model.modules():
        if isinstance(m, Qwen3NextSparseMoeBlock):
            num_experts = m.experts.num_experts
            lo, hi = expert_range(num_experts, rank, world_size)
            new_forward = _make_ep_forward(m, lo, hi)
            # Bind as an instance attribute — replaces the bound method.
            m.forward = new_forward  # type: ignore[assignment]
            count += 1

    log.info(
        "patched %d MoE blocks for EP (rank=%d ws=%d experts=[%d,%d))",
        count, rank, world_size, lo if count else -1, hi if count else -1,
    )
    return count
