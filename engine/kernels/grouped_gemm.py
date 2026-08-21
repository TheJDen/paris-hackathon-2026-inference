# pyright: reportAttributeAccessIssue=false
# Vendored from woct0rdho/transformers-qwen3-moe-fused (Apache-2.0)
# https://github.com/woct0rdho/transformers-qwen3-moe-fused
# path: qwen3_moe_fused/grouped_gemm/non_persistent/forward.py @ 10c7309
# Modifications: import paths adjusted for engine.kernels and helper functions
# exceeds_smem_capacity and is_int_tensor inlined
# removed host sync and opted for static launch grid
# added fused scatter add to reduce peak mem in CUDA Graph pool

# y[m, n] = sum_k w[s[m], n, k] * x[m, k]

from functools import partial
from typing import Optional

import torch
import triton
import triton.language as tl

from engine.kernels.autotuning import (
    get_autotune_configs,
    get_autotune_keys,
    prune_configs,
)


def exceeds_smem_capacity(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
    smem_size: int,
) -> bool:
    # Strix Halo crashes with some configs
    if torch.version.hip and (BLOCK_SIZE_M >= 128 or BLOCK_SIZE_N >= 128 or BLOCK_SIZE_K >= 128):
        return True

    x_size = BLOCK_SIZE_M * BLOCK_SIZE_K * dtype.itemsize
    w_size = BLOCK_SIZE_N * BLOCK_SIZE_K * dtype.itemsize
    if num_stages <= 1:
        size = max(x_size, w_size)
    else:
        # (num_stages - 1) stages of both tiles will be cached in smem
        size = (num_stages - 1) * (x_size + w_size)
    return size > smem_size

def is_int_tensor(x: torch.Tensor) -> bool:
    return x.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

@triton.autotune(
    configs=get_autotune_configs(),
    key=get_autotune_keys(),
    prune_configs_by={"early_config_prune": partial(prune_configs, exceeds_smem_capacity)},
    cache_results=True,
    restore_value=["y_ptr"]
)
@triton.jit
def _grouped_gemm_forward_kernel(
    # Pointers
    x_ptr,
    w_ptr,
    m_offsets_ptr,
    y_ptr,
    scatter_indices_ptr,
    # Dimensions
    M: int,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    # Strides
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    # Metadata
    FUSE_SCATTER_ADD: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
) -> None:
    tile_idx = tl.program_id(0)
    expert_idx = tl.program_id(1)

    m_start = tl.load(m_offsets_ptr + expert_idx - 1, mask=expert_idx > 0, other=0).to(tl.int32)
    m_end = tl.load(m_offsets_ptr + expert_idx).to(tl.int32)
    m_size = m_end - m_start

    num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    if tile_idx >= num_m_tiles * num_n_tiles:
        return

    # Output tile for this thread block for this expert group
    tile_m_idx = tile_idx % num_m_tiles
    tile_n_idx = tile_idx // num_m_tiles

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_m = m_start + tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xk * offs_k[None, :]
    mask_m = offs_m < m_end

    offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[:, None] + stride_wk * offs_k[None, :]
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # GEMM main loop
    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = offs_k < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :])

        accumulator += tl.dot(x, w.T)

        offs_k += BLOCK_SIZE_K
        x_ptrs += stride_xk * BLOCK_SIZE_K
        w_ptrs += stride_wk * BLOCK_SIZE_K

    y = accumulator.to(y_ptr.dtype.element_ty)
    if FUSE_SCATTER_ADD:
        # To achieve fused scatter add, we must use atomic add to prevent race
        dest = tl.load(scatter_indices_ptr + offs_m, mask=mask_m, other=0)
        tl.atomic_add(
            y_ptr + dest[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            y,
            mask=mask_m[:, None] & mask_n[None, :],
            sem="relaxed"
        )
    else:
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def grouped_gemm_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    m_offsets: torch.Tensor,
    max_group_size: int,
    *,
    _out: torch.Tensor | None = None,
    _scatter_indices: torch.Tensor | None = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    assert x.is_cuda
    assert w.device == x.device
    assert m_offsets.device == x.device
    assert is_int_tensor(m_offsets)
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_offsets.is_contiguous()
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_offsets.ndim == 1
    M, _ = x.shape
    E, N, K = w.shape
    T = max_group_size
    assert x.shape[1] == K
    assert m_offsets.numel() == E

    if dtype is None:
        dtype = x.dtype

    def grid(META):
        bm = META["BLOCK_SIZE_M"]
        bn = META["BLOCK_SIZE_N"]
        max_m_blocks = triton.cdiv(T, bm)
        num_n_tiles = triton.cdiv(N, bn)
        # Grid: (Max Tiles per Expert, Number of Experts)
        return (max_m_blocks * num_n_tiles, E)
    FUSE_SCATTER_ADD = _out is not None
    if FUSE_SCATTER_ADD:
        assert _scatter_indices is not None
        y = _out
    else:
        y = torch.empty((M, N), device=x.device, dtype=dtype or x.dtype)
    with torch.cuda.device(x.device):
        _grouped_gemm_forward_kernel[grid](
            # Pointers
            x,
            w,
            m_offsets,
            y,
            _scatter_indices,
            # Dimensions
            M,
            N,
            K,
            E,
            # Strides
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            y.stride(0),
            y.stride(1),
            FUSE_SCATTER_ADD=_scatter_indices is not None
        )
    return y
