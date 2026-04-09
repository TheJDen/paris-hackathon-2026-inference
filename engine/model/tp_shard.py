"""From-scratch tensor-parallel sharding for Qwen3.5-MoE.

Replaces specific Linear layers with sharded variants that use
torch.distributed all-reduce. NO third-party libs (no DTensor, no
accelerate, no HF tp_plan, no vLLM, no Megatron).

Scope (TP=2 only, minimal):
  - Attention: q_proj/k_proj/v_proj (ColumnParallel) + o_proj (RowParallel)
  - shared_expert MLP: gate_proj/up_proj (ColumnParallel) + down_proj (RowParallel)

Explicitly NOT sharded (replicated on all ranks):
  - The 256 routed experts
  - Router/gate
  - lm_head, embeddings
  - RMSNorm, rotary, etc.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """y = x @ W^T + b, W sharded along OUTPUT dim across TP ranks.

    Each rank holds W[rank * out_per_rank : (rank+1) * out_per_rank, :].
    Output is LOCAL: shape [..., out_per_rank]. Caller must feed into a
    RowParallelLinear (which expects sharded input) or be aware.
    """
    def __init__(self, full_linear: nn.Linear, tp_size: int, tp_rank: int):
        super().__init__()
        out_features, in_features = full_linear.weight.shape
        assert out_features % tp_size == 0, (
            f"out_features={out_features} not divisible by tp_size={tp_size}"
        )
        out_per_rank = out_features // tp_size
        local_weight = full_linear.weight.data[
            tp_rank * out_per_rank : (tp_rank + 1) * out_per_rank, :
        ].contiguous().clone()
        self.weight = nn.Parameter(local_weight, requires_grad=False)
        if full_linear.bias is not None:
            local_bias = full_linear.bias.data[
                tp_rank * out_per_rank : (tp_rank + 1) * out_per_rank
            ].contiguous().clone()
            self.bias = nn.Parameter(local_bias, requires_grad=False)
        else:
            self.bias = None
        self.in_features = in_features
        self.out_features_full = out_features
        self.out_features = out_per_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """y = x @ W^T + b, W sharded along INPUT dim across TP ranks.

    Input is expected to already be sharded along its last dim (produced by
    a ColumnParallelLinear). Each rank computes a partial sum, then we
    all-reduce to get the full output.
    """
    def __init__(self, full_linear: nn.Linear, tp_size: int, tp_rank: int):
        super().__init__()
        out_features, in_features = full_linear.weight.shape
        assert in_features % tp_size == 0, (
            f"in_features={in_features} not divisible by tp_size={tp_size}"
        )
        in_per_rank = in_features // tp_size
        local_weight = full_linear.weight.data[
            :, tp_rank * in_per_rank : (tp_rank + 1) * in_per_rank
        ].contiguous().clone()
        self.weight = nn.Parameter(local_weight, requires_grad=False)
        # Replicate bias, add once on rank 0 after all-reduce to avoid
        # double-counting.
        if full_linear.bias is not None:
            if tp_rank == 0:
                self.bias = nn.Parameter(full_linear.bias.data.clone(), requires_grad=False)
            else:
                self.bias = None
            self._has_bias = True
        else:
            self.bias = None
            self._has_bias = False
        self.out_features = out_features
        self.in_features_full = in_features
        self.in_features = in_per_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = torch.nn.functional.linear(x, self.weight, None)
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        if self._has_bias and self.bias is not None:
            local_out = local_out + self.bias
        return local_out


def _replace_attention(attn_module, tp_size: int, tp_rank: int) -> bool:
    """Replace q/k/v (col) and o_proj (row) on a Qwen3_5MoeAttention.

    Returns True if at least one projection was replaced.
    """
    replaced = False
    # q_proj: out = num_heads * head_dim. num_heads (16) must be divisible by tp_size.
    if hasattr(attn_module, 'q_proj') and isinstance(attn_module.q_proj, nn.Linear):
        if attn_module.q_proj.weight.shape[0] % tp_size == 0:
            attn_module.q_proj = ColumnParallelLinear(attn_module.q_proj, tp_size, tp_rank)
            replaced = True
    # k_proj / v_proj: out = num_kv_heads * head_dim. num_kv_heads=2; tp=2 → 1 per rank.
    kv_sharded = False
    if hasattr(attn_module, 'k_proj') and isinstance(attn_module.k_proj, nn.Linear):
        if attn_module.k_proj.weight.shape[0] % tp_size == 0:
            attn_module.k_proj = ColumnParallelLinear(attn_module.k_proj, tp_size, tp_rank)
            kv_sharded = True
            replaced = True
    if hasattr(attn_module, 'v_proj') and isinstance(attn_module.v_proj, nn.Linear):
        if attn_module.v_proj.weight.shape[0] % tp_size == 0:
            attn_module.v_proj = ColumnParallelLinear(attn_module.v_proj, tp_size, tp_rank)
            replaced = True
    # o_proj is row-parallel on the input dim (which equals num_heads*head_dim,
    # already sharded by q sharding).
    if hasattr(attn_module, 'o_proj') and isinstance(attn_module.o_proj, nn.Linear):
        if attn_module.o_proj.weight.shape[1] % tp_size == 0:
            attn_module.o_proj = RowParallelLinear(attn_module.o_proj, tp_size, tp_rank)
            replaced = True

    # Update head counts on the module so the reshape in forward() uses local counts.
    if replaced and hasattr(attn_module, 'num_heads'):
        if attn_module.num_heads % tp_size == 0:
            attn_module.num_heads = attn_module.num_heads // tp_size
    if kv_sharded and hasattr(attn_module, 'num_key_value_heads'):
        if attn_module.num_key_value_heads % tp_size == 0:
            attn_module.num_key_value_heads = attn_module.num_key_value_heads // tp_size
    # Some HF attention modules also store num_key_value_groups = num_heads // num_kv_heads.
    # Leave it alone if both were divided by the same factor (ratio preserved).
    return replaced


def _replace_mlp(mlp_module, tp_size: int, tp_rank: int) -> bool:
    """Replace gate_proj/up_proj (col) and down_proj (row) on a standard SwiGLU MLP."""
    replaced = False
    if hasattr(mlp_module, 'gate_proj') and isinstance(mlp_module.gate_proj, nn.Linear):
        if mlp_module.gate_proj.weight.shape[0] % tp_size == 0:
            mlp_module.gate_proj = ColumnParallelLinear(mlp_module.gate_proj, tp_size, tp_rank)
            replaced = True
    if hasattr(mlp_module, 'up_proj') and isinstance(mlp_module.up_proj, nn.Linear):
        if mlp_module.up_proj.weight.shape[0] % tp_size == 0:
            mlp_module.up_proj = ColumnParallelLinear(mlp_module.up_proj, tp_size, tp_rank)
            replaced = True
    if hasattr(mlp_module, 'down_proj') and isinstance(mlp_module.down_proj, nn.Linear):
        if mlp_module.down_proj.weight.shape[1] % tp_size == 0:
            mlp_module.down_proj = RowParallelLinear(mlp_module.down_proj, tp_size, tp_rank)
            replaced = True
    return replaced


def apply_tensor_parallel(model: nn.Module, tp_size: int, tp_rank: int) -> None:
    """In-place: shard attention + shared_expert across TP ranks.

    Skips routed experts (256 per layer), router, embeddings, lm_head.
    """
    if tp_size <= 1:
        return
    n_attn = 0
    n_mlp = 0
    # Snapshot the module list because we mutate during iteration.
    for name, mod in list(model.named_modules()):
        cls_name = type(mod).__name__
        if 'Attention' in cls_name and 'Rotary' not in cls_name:
            try:
                if _replace_attention(mod, tp_size, tp_rank):
                    n_attn += 1
            except Exception as e:
                print(f"[tp_shard] failed to shard attention {name} ({cls_name}): {e}")
        elif name.endswith('.shared_expert') and hasattr(mod, 'gate_proj'):
            try:
                if _replace_mlp(mod, tp_size, tp_rank):
                    n_mlp += 1
            except Exception as e:
                print(f"[tp_shard] failed to shard shared_expert {name}: {e}")
    print(
        f"[tp_shard] tp_size={tp_size} tp_rank={tp_rank}: "
        f"sharded {n_attn} attentions, {n_mlp} shared MLPs"
    )


def init_tp_process_group() -> tuple[int, int]:
    """Init NCCL process group from torchrun env vars. Returns (rank, world_size)."""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    return dist.get_rank(), dist.get_world_size()
