# pyright: reportAttributeAccessIssue=false
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# licenses/LICENSE.fbgemm.
#
# Vendored from pytorch/FBGEMM @ 5d0a3d9
# path: fbgemm_gpu/experimental/gemm/triton_gemm/grouped_gemm.py
# Modifications: removed the WS and fp8 kernels and the tma_utils dependency);
# iterated_tiles cast to int64; `and`→`&` mask fixes;

# if we really want to push prefill on H200 later we can revive WS
import sys
import warnings

import torch
import triton
import triton.language as tl
from triton.runtime import driver

# `nv_tma_desc_type` (Triton PR #4498) was removed in Triton 3.x.
HAS_TMA_DESC = hasattr(tl, "make_tensor_descriptor")

if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
        file=sys.stderr,
    )
else:
    print(
        "TMA benchmarks will be running without grid constant TMA descriptor.",
        file=sys.stderr,
    )

_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
    for num_ctas in [1]
]

def early_config_prune(configs, named_args, dtsize=None, dtype=None, **kwargs):
    device = torch.cuda.current_device()
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    if dtsize is None:
        dtsize = named_args["c_ptr"].element_size()
    if dtype is None:
        dtype = named_args["c_ptr"].dtype

    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        (
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_stages,
            use_tma_load_on_scales,
        ) = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
            kw.get("USE_TMA_LOAD_ON_SCALES", False),
        )
        G, M, N = (
            named_args["G"],
            named_args["M_BUCKET"],
            named_args["N"],
        )

        # 1. make sure we have enough smem
        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        if torch.version.hip:
            required_shared_memory = BLOCK_N * BLOCK_K * num_stages * dtsize
        else:
            required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        M_PER_GROUP = M // G
        MIN_M_TILES = 32 if torch.version.hip else 64
        # 2. make sure we don't load M tiles that are too big
        if BLOCK_M > MIN_M_TILES and BLOCK_M > (M_PER_GROUP * 2):
            continue
        # 3. make sure we don't load N tiles that are too small
        if BLOCK_M < 128 and BLOCK_M < (M_PER_GROUP // 2):
            continue

        num_sm = driver.active.utils.get_device_properties(device)[
            "multiprocessor_count"
        ]
        N_TILES = (N + BLOCK_N - 1) // BLOCK_N
        MIN_N_TILES = 32 if torch.version.hip else 64
        # 4. make sure we don't load N tiles that are too big
        if BLOCK_N > MIN_N_TILES and M * N_TILES < num_sm:
            continue
        # 5. make sure we don't load N tiles that are too small
        if BLOCK_N < 128 and M * N_TILES > 2 * num_sm:
            continue
        if dtsize >= 2 and use_tma_load_on_scales:
                continue
        pruned_configs.append(config)

    return pruned_configs


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _fbgemm_grouped_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    scatter_add_indices,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)  # pyre-ignore
    iterated_tiles = 0
    iterated_tiles = iterated_tiles.to(tl.int64)
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g)

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N
            n_size = N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                c_desc_ptr = tl.make_tensor_descriptor(
                    c_ptr + M_start_offset * N,
                    # pyrefly: ignore [bad-argument-type]
                    shape=[m_size, n_size],
                    # pyre-ignore
                    strides=[n_size, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            if USE_TMA_LOAD:
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                a_desc_ptr = tl.make_tensor_descriptor(
                    a_ptr + M_start_offset * K,
                    # pyrefly: ignore [bad-argument-type]
                    shape=[m_size, K],
                    # pyre-ignore
                    strides=[K, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                )
                b_desc_ptr = tl.make_tensor_descriptor(
                    b_ptr + N_start_offset * K,
                    # pyre-ignore
                    shape=[n_size, K],
                    # pyre-ignore
                    strides=[K, 1],
                    block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
                )

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                if USE_TMA_LOAD:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        # pyrefly: ignore [bad-argument-type, unbound-name]
                        a = a_desc_ptr.load([m_offset, k_offset])
                        # pyrefly: ignore [bad-argument-type, unbound-name]
                        b = b_desc_ptr.load([n_offset, k_offset])
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = (
                        a_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )
                    b_ptrs = (
                        b_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        updated_k_offset = k_offset + offs_k
                        updated_k_offset_mask = updated_k_offset[None, :] < K  # type: ignore[16]
                        a = tl.load(
                            a_ptrs,
                            mask=((offs_am[:, None] < m_size) & updated_k_offset_mask),
                            other=0.0,
                        )
                        b = tl.load(
                            b_ptrs,
                            mask=((offs_bn[:, None] < n_size) & updated_k_offset_mask),
                            other=0.0,
                        )
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    # pyre-ignore
                    c_desc_ptr.store(
                        [m_offset, n_offset], accumulator.to(c_ptr.dtype.element_ty)
                    )
                elif FUSE_SCATTER_ADD:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    mask = offs_am < m_size
                    m_offsets = tl.load(
                        scatter_add_indices + M_start_offset + offs_am,
                        mask=mask,
                        cache_modifier=".ca",
                    )
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.atomic_add(
                        c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                        c,
                        mask=mask[:, None] & (offs_bn[None, :] < n_size),
                        sem="relaxed",
                    )
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None]) * N
                        + offs_bn[None, :],
                        c,
                        mask=(offs_am[:, None] < m_size) & (offs_bn[None, :] < n_size),
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles

def _grouped_gemm(
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_fast_accum: bool,
    output_tensor: torch.Tensor | None,
    scatter_add_indices: torch.Tensor | None,
) -> torch.Tensor:

    USE_TMA_LOAD = not torch.version.hip
    USE_TMA_STORE = False

    if USE_TMA_LOAD and not HAS_TMA_DESC:
        USE_TMA_LOAD = False
        warnings.warn(
            "TMA load is disabled as there is no TMA descriptor support!", stacklevel=2
        )

    if USE_TMA_STORE and not HAS_TMA_DESC:
        USE_TMA_STORE = False
        warnings.warn(
            "TMA store is disabled as there is no TMA descriptor support!", stacklevel=2
        )

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M, K = x.shape
    N = w.shape[0] // G
    assert K == w.shape[1]

    if K % 8 != 0 or N % 8 != 0:
        USE_TMA_LOAD = False
        USE_TMA_STORE = False
        warnings.warn(
            f"TMA load and warp specialization are disabled since K or N is not a multiple of 8: {K=}, {N=}.",
            stacklevel=2,
        )

        assert (
            output_tensor is None
        ), f"Fused scatter add has large rounding error when K or N is not a multiple of 8: {K=}, {N=}."

    # No fp8 K%16 gate needed: K % BLOCK_SIZE_K == 0 and every BLOCK_SIZE_K % 16 == 0.
    if output_tensor is None:
        FUSE_SCATTER_ADD = False
        assert scatter_add_indices is None
        y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    else:
        FUSE_SCATTER_ADD = True
        assert scatter_add_indices is not None
        assert scatter_add_indices.is_contiguous()
        assert scatter_add_indices.shape == (M,)
        y = output_tensor
    if M == 0 or N == 0:
        return y

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # WS kernels still consume host-built descriptors; non-WS build them device-side.
    desc_x, desc_w = x, w
    desc_helper = None

    if USE_TMA_LOAD or USE_TMA_STORE:

        def alloc_fn(size: int, alignment: int, stream: int | None):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    def grid(META):
        if desc_helper is not None:
            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M,
                K,
                META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
                META["BLOCK_SIZE_K"],
                x.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                N * G,
                K,
                META["BLOCK_SIZE_N"],
                META["BLOCK_SIZE_K"],
                w.element_size(),
            )
        return (NUM_SMS,)

    M_BUCKET_CAP = 16384
    M_BUCKET = min(triton.next_power_of_2(M), M_BUCKET_CAP)
    args = (
        desc_x,
        desc_w,
        y,
        scatter_add_indices,
        m_sizes,
        G,
        M_BUCKET,
        N,
        K,
        NUM_SMS,
        FUSE_SCATTER_ADD,
        USE_TMA_LOAD,
    )
    args += (USE_TMA_STORE, use_fast_accum)
    _fbgemm_grouped_gemm[grid](*args)

    return y


def grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_fast_accum: bool = True,
    *,
    _output_tensor: torch.Tensor | None = None,
    _scatter_add_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    return _grouped_gemm(
        x=x,
        w=w,
        m_sizes=m_sizes,
        use_fast_accum=use_fast_accum,
        output_tensor=_output_tensor,
        scatter_add_indices=_scatter_add_indices,
    )


