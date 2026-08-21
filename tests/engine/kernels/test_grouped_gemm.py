import pytest
import torch
import torch.nn.functional as F

import engine.kernels.grouped_gemm

device = "cuda"

@pytest.mark.parametrize("T,E,N,K", [(512, 8, 128, 64), (2040, 256, 1024, 2048)])
def test_grouped_gemm(T, E, N, K):
    torch.manual_seed(69)
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device=device, dtype=torch.bfloat16)
    offs = torch.bincount(torch.randint(0, E, [T], device=device), minlength=E).cumsum(0).int()
    with torch.no_grad():
        torch.testing.assert_close(
            engine.kernels.grouped_gemm.grouped_gemm_forward(x, w, offs, T),
            F.grouped_mm(x, w.transpose(-1, -2), offs=offs),
            atol=2e-2,
            rtol=2e-2
        )


@pytest.mark.parametrize("T,E,N,K,O", [(512, 8, 128, 64, 64), (2040, 256, 1024, 2048, 300)])
def test_grouped_gemm_fused_scatter_add(T, E, N, K, O):
    torch.manual_seed(69)
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device=device, dtype=torch.bfloat16)
    offs = torch.bincount(torch.randint(0, E, [T], device=device), minlength=E).cumsum(0).int()
    scatter_idx = torch.randint(0, O, [T], device=device, dtype=torch.int64)
    with torch.no_grad():
        y = F.grouped_mm(x, w.transpose(-1, -2), offs=offs)
        ref = torch.zeros(O, N, device=device, dtype=torch.float32)
        ref.index_add_(0, scatter_idx, y.float())
        
        out = torch.zeros(O, N, device=device, dtype=torch.bfloat16)
        engine.kernels.grouped_gemm.grouped_gemm_forward(
            x, w, offs, T,
            _out=out, _scatter_indices=scatter_idx
        )
        torch.testing.assert_close(
            ref,
            out.float(),
            atol=2e-2 * y.abs().max(),
            rtol=2e-2
        )

