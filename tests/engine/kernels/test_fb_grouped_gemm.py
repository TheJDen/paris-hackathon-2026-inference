import pytest
import torch
import torch.nn.functional as F

import engine.kernels.fbgemm_grouped_gemm

device = "cuda"

@pytest.mark.parametrize("T,E,N,K", [(512, 8, 128, 64), (2040, 256, 1024, 2048)])
def test_grouped_gemm(T, E, N, K):
    torch.manual_seed(69)
    x = torch.randn(T, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device=device, dtype=torch.bfloat16)
    counts = torch.bincount(torch.randint(0, E, [T], device=device), minlength=E)
    with torch.no_grad():
        torch.testing.assert_close(
            engine.kernels.fbgemm_grouped_gemm.grouped_gemm(x, w.reshape(E * N, K), counts),
            F.grouped_mm(x, w.transpose(-1, -2), offs=counts.cumsum(0).int()),
            atol=2e-2,
            rtol=2e-2
        )



