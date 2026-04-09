"""Expert Parallelism (EP=8) for Qwen3.5-MoE inference.

Architecture
------------
- `Qwen3_5MoeExperts` stores all expert weights in two fused tensors:
      gate_up_proj: [num_experts, 2*inter, hidden]
      down_proj:    [num_experts, hidden, inter]
- We replace each `Qwen3_5MoeSparseMoeBlock` with `EPSparseMoeBlock`:
    * gate (TopKRouter) and shared_expert are replicated on all ranks
    * Each rank keeps only its expert weight slice [n_local, ...]
    * AllReduce sums partial expert outputs across all 8 GPUs
- Non-MoE layers (attention, DeltaNet, embedding, norm) run identically on
  all ranks — replicated weights, identical computation, identical outputs.

Memory per GPU (before vs after EP)
------------------------------------
Before: full model ~70 GB (all 256 experts)
After:  ~10 GB non-expert + 60 GB × 32/256 ≈ 17.5 GB

Coordination
------------
Rank 0 runs the full engine (scheduler + HTTP server). Before each model
forward it broadcasts the batch inputs to ranks 1-7. All ranks run the
forward in lockstep; the AllReduces inside EPSparseMoeBlock keep them in
sync. Ranks 1-7 discard the model output and wait for the next broadcast.
"""

from __future__ import annotations

import gc
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class EPSparseMoeBlock(nn.Module):
    """Drop-in replacement for Qwen3_5MoeSparseMoeBlock with EP.

    Rank r owns expert weights at indices [r*n_local, (r+1)*n_local).
    AllReduce sums partial outputs into the final hidden states.
    """

    def __init__(self, src: nn.Module, rank: int, world_size: int) -> None:
        super().__init__()
        n_experts: int = src.experts.num_experts
        assert n_experts % world_size == 0, (
            f"num_experts={n_experts} must be divisible by world_size={world_size}"
        )
        n_local = n_experts // world_size
        expert_start = rank * n_local

        self.rank = rank
        self.world_size = world_size
        self.n_experts = n_experts
        self.n_local = n_local
        self.expert_start = expert_start

        # Replicated modules — gate and shared expert have same weights on all ranks.
        self.gate = src.gate
        self.shared_expert = src.shared_expert
        self.shared_expert_gate = src.shared_expert_gate

        # Clone only this rank's expert weight slice so the rest can be freed.
        self.gate_up_proj = nn.Parameter(
            src.experts.gate_up_proj.data[expert_start : expert_start + n_local].clone()
        )
        self.down_proj = nn.Parameter(
            src.experts.down_proj.data[expert_start : expert_start + n_local].clone()
        )
        self.act_fn = src.experts.act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, D = hidden_states.shape
        h = hidden_states.view(-1, D)  # [T, D]

        # Shared expert — identical on all ranks.
        shared_out = self.shared_expert(h)
        shared_out = F.sigmoid(self.shared_expert_gate(h)) * shared_out

        # Router — identical on all ranks (replicated weight).
        _, routing_weights, selected_experts = self.gate(h)
        # routing_weights: [T, top_k], selected_experts: [T, top_k]

        # Partial expert output for this rank's shard.
        partial = self._compute_shard(h, selected_experts, routing_weights)

        # AllReduce: sum partial contributions from all 8 GPUs.
        if self.world_size > 1:
            dist.all_reduce(partial, op=dist.ReduceOp.SUM)

        return (partial + shared_out).view(B, L, D)

    def _compute_shard(
        self,
        h: torch.Tensor,           # [T, D]
        top_k_index: torch.Tensor, # [T, top_k]
        top_k_weights: torch.Tensor,  # [T, top_k]
    ) -> torch.Tensor:
        T, D = h.shape
        final = torch.zeros_like(h)

        with torch.no_grad():
            # expert_mask: [T, top_k, n_experts] → [n_experts, top_k, T]
            expert_mask = F.one_hot(top_k_index, num_classes=self.n_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        # Only look at this rank's expert slice.
        local_mask = expert_mask[self.expert_start : self.expert_start + self.n_local]
        # expert_hit: local indices of experts that received at least one token.
        expert_hit = local_mask.sum(dim=(-1, -2)).nonzero()

        for idx_t in expert_hit:
            local_idx = int(idx_t[0])
            top_k_pos, token_idx = torch.where(local_mask[local_idx])
            tokens = h[token_idx]                                          # [n, D]
            gate, up = F.linear(tokens, self.gate_up_proj[local_idx]).chunk(2, dim=-1)
            out = self.act_fn(gate) * up
            out = F.linear(out, self.down_proj[local_idx])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))

        return final


def patch_model_for_ep(model: nn.Module, rank: int, world_size: int) -> int:
    """Replace all Qwen3_5MoeSparseMoeBlock instances with EPSparseMoeBlock.

    Frees expert weights outside this rank's shard so GPU memory drops from
    ~70 GB to ~17.5 GB per rank.  Returns the number of blocks patched.
    """
    if world_size == 1:
        return 0

    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )

    # Collect first (can't modify named_modules dict while iterating).
    targets: list[tuple[nn.Module, str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, Qwen3_5MoeSparseMoeBlock):
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent, child_name = model, name
            targets.append((parent, child_name, module))

    for parent, child_name, src in targets:
        # Build EP block — clones only this rank's expert weights.
        ep_block = EPSparseMoeBlock(src, rank, world_size)

        # Explicitly free the full expert weight tensors before GC.
        del src.experts.gate_up_proj, src.experts.down_proj

        # Replace in the model graph.
        setattr(parent, child_name, ep_block)

    gc.collect()
    torch.cuda.empty_cache()
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    log.info(
        "EP rank %d/%d: patched %d MoE blocks, GPU mem after = %.2f GB",
        rank, world_size, len(targets), mem_gb,
    )
    return len(targets)
