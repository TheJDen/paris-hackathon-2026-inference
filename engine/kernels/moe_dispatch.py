"""MoE dispatch kernel: top-k router indices -> sorted permutation + offsets.

This replaces the Python-side per-expert loop in HuggingFace's
``Qwen3_5MoeExperts.forward`` (which spent ~50% of CUDA self-time in
``aten::index``/``vectorized_gather_kernel`` according to the Phase 1
profile) with a single fused GPU kernel.

Given ``top_k_idx: [T, K]`` (expert index per (token, slot) pair) and the
total expert count ``E``, we compute:

- ``sorted_token_idx: [T*K]`` — original token index, sorted by expert.
- ``sorted_slot_idx:  [T*K]`` — which of the ``K`` slots the pair came from
  (needed so we can look up the routing weight during the expert combine).
- ``expert_offsets:   [E+1]`` — CSR-style offsets such that
  ``sorted_token_idx[expert_offsets[e]:expert_offsets[e+1]]`` is the set of
  tokens routed to expert ``e``.

Two implementations are provided:

1. **Helion** (fast, single kernel, H200/Hopper-tuned). Used when helion is
   importable *and* the tensors are on CUDA.
2. **torch fallback** using ``torch.sort`` + ``torch.bincount``. Already much
   faster than the HF reference because it replaces 256 Python iterations and
   256 boolean-mask gathers with 2 fused CUDA kernels. This runs on first
   call while Helion autotunes in the background, and is the only path when
   helion is not installed (e.g. in dev).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helion kernel (conditional import — Helion is installed on the H200 server
# but not necessarily in the dev venv).
# ---------------------------------------------------------------------------

_HELION_AVAILABLE = False
_helion_import_error: Optional[BaseException] = None
try:
    import helion  # type: ignore
    import helion.language as hl  # type: ignore

    _HELION_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on environment
    _helion_import_error = e


if _HELION_AVAILABLE:

    # Hand-picked Hopper (H200) config. We don't autotune on the hot path
    # because the search is slow and the kernel is memory-bound + atomics-
    # dominated, not compute-bound, so a reasonable hand-config is close
    # enough. Override by deleting and letting the default autotuner run if
    # profiling suggests otherwise.
    _HOPPER_DISPATCH_CONFIG = helion.Config(
        block_sizes=[1024],
        num_warps=4,
        num_stages=2,
        indexing="pointer",
        pid_type="flat",
    )

    @helion.kernel(config=_HOPPER_DISPATCH_CONFIG, static_shapes=False)
    def _helion_histogram(
        flat_expert_ids: torch.Tensor,  # [T*K], int32
        counts: torch.Tensor,  # [E], int32, zero-initialized by caller
    ) -> torch.Tensor:
        """Pass 1: count how many (token, slot) pairs route to each expert."""
        (n,) = flat_expert_ids.size()
        for tile in hl.tile(n):
            e = flat_expert_ids[tile]
            # e is an int32 vector of length tile_size. For each entry,
            # bump counts[e] by 1. Using atomic_add with a per-entry value
            # of 1 broadcast via gather-style indexing.
            ones = hl.full([tile], 1, dtype=torch.int32)
            hl.atomic_add(counts, [e], ones)
        return counts

    @helion.kernel(config=_HOPPER_DISPATCH_CONFIG, static_shapes=False)
    def _helion_permute(
        flat_expert_ids: torch.Tensor,  # [T*K], int32
        expert_offsets: torch.Tensor,  # [E+1], int32, exclusive prefix-sum of counts
        write_cursor: torch.Tensor,  # [E], int32, zero-initialized
        sorted_token_idx: torch.Tensor,  # [T*K], int32 (output)
        sorted_slot_idx: torch.Tensor,  # [T*K], int32 (output)
        k: int,
    ) -> None:
        """Pass 2: scatter (token, slot) pairs into the sorted layout."""
        (n,) = flat_expert_ids.size()
        for tile in hl.tile(n):
            e = flat_expert_ids[tile]
            flat = tile.index  # [tile_size] int32
            token = flat // k
            slot = flat % k
            ones = hl.full([tile], 1, dtype=torch.int32)
            # Atomically reserve a position within this expert's bucket.
            pos = hl.atomic_add(write_cursor, [e], ones)  # returns previous value
            base = expert_offsets[e]
            dst = base + pos
            sorted_token_idx[dst] = token
            sorted_slot_idx[dst] = slot

else:  # pragma: no cover
    _helion_histogram = None  # type: ignore
    _helion_permute = None  # type: ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _torch_dispatch(
    top_k_idx: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch fallback. Uses radix sort + bincount — already fast.

    Returned dtypes: int32 for the two permutation vectors (cheap to index),
    int64 for the expert_offsets (torch grouped-mm expects int64).
    """
    T, K = top_k_idx.shape
    flat = top_k_idx.reshape(-1).to(torch.int32)
    # stable sort so the output is deterministic wrt original token order.
    sorted_experts, perm = torch.sort(flat, stable=True)
    sorted_token_idx = (perm // K).to(torch.int32)
    sorted_slot_idx = (perm % K).to(torch.int32)
    counts = torch.bincount(flat.to(torch.int64), minlength=num_experts)
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int64, device=flat.device)
    expert_offsets[0] = 0
    torch.cumsum(counts, dim=0, out=expert_offsets[1:])
    return sorted_token_idx, sorted_slot_idx, expert_offsets


def _helion_dispatch(
    top_k_idx: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _HELION_AVAILABLE
    T, K = top_k_idx.shape
    device = top_k_idx.device
    flat = top_k_idx.reshape(-1).to(torch.int32).contiguous()
    N = flat.numel()

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    _helion_histogram(flat, counts)

    # Exclusive prefix sum -> offsets[0..E], length E+1.
    offsets32 = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    offsets32[0] = 0
    torch.cumsum(counts, dim=0, dtype=torch.int32, out=offsets32[1:])

    write_cursor = torch.zeros(num_experts, dtype=torch.int32, device=device)
    sorted_token_idx = torch.empty(N, dtype=torch.int32, device=device)
    sorted_slot_idx = torch.empty(N, dtype=torch.int32, device=device)
    _helion_permute(
        flat,
        offsets32[:-1].contiguous(),
        write_cursor,
        sorted_token_idx,
        sorted_slot_idx,
        K,
    )

    # Downstream torch._grouped_mm wants int64 offsets.
    return sorted_token_idx, sorted_slot_idx, offsets32.to(torch.int64)


def moe_dispatch_indices(
    top_k_idx: torch.Tensor,
    num_experts: int,
    *,
    use_helion: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (sorted_token_idx, sorted_slot_idx, expert_offsets).

    Parameters
    ----------
    top_k_idx : [T, K] integer tensor of expert indices (0..num_experts-1).
    num_experts : total expert count.
    use_helion : if None, auto-pick Helion on CUDA when available; otherwise
        respect the caller.

    Returns
    -------
    sorted_token_idx : [T*K] int32, original token indices sorted by expert.
    sorted_slot_idx  : [T*K] int32, original top-k slot (0..K-1) per entry.
    expert_offsets   : [E+1] int64 CSR offsets.
    """
    if use_helion is None:
        use_helion = _HELION_AVAILABLE and top_k_idx.is_cuda

    if use_helion:
        try:
            return _helion_dispatch(top_k_idx, num_experts)
        except Exception as e:  # pragma: no cover - runtime compile failure
            log.warning("helion moe_dispatch failed (%s); falling back to torch", e)
            return _torch_dispatch(top_k_idx, num_experts)

    return _torch_dispatch(top_k_idx, num_experts)


def helion_available() -> bool:
    return _HELION_AVAILABLE
