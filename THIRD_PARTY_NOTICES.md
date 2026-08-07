# Third-Party Notices

This project vendors third-party Triton kernels. Their original licenses are
reproduced in the `licenses/` directory, and each vendored source file carries a
header noting its upstream origin, commit, and any modifications.

## woct0rdho/transformers-qwen3-moe-fused (Apache-2.0)

- Upstream: https://github.com/woct0rdho/transformers-qwen3-moe-fused
- License: Apache License 2.0 — see [`licenses/LICENSE.woct0rdho`](licenses/LICENSE.woct0rdho)
- Vendored files:
  - `engine/kernels/grouped_gemm.py` — from `qwen3_moe_fused/grouped_gemm/non_persistent/forward.py` @ `10c7309`
  - `engine/kernels/autotuning.py` — from `qwen3_moe_fused/grouped_gemm/autotuning.py` @ `4c11d83`
- Modifications are noted in each file header (import paths, inlined helpers,
  removed the undeclared `LOOP_ORDER` knob).

## pytorch/FBGEMM (BSD-3-Clause)

- Upstream: https://github.com/pytorch/FBGEMM
- License: BSD-3-Clause, Copyright (c) Meta Platforms, Inc. and affiliates —
  see [`licenses/LICENSE.fbgemm`](licenses/LICENSE.fbgemm)
- Vendored file:
  - `engine/kernels/fbgemm_grouped_gemm.py` — from
    `fbgemm_gpu/experimental/gemm/triton_gemm/grouped_gemm.py`
- Modifications are noted in the file header (pruned to the non-warp-specialization
  modern-TMA path; removed the WS and fp8 kernels and the `tma_utils` dependency;
  `iterated_tiles` cast to int64; mask-operator fixes).
