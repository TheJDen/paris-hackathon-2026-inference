import pytest
import torch
import transformers
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

import engine.kernels.moe_experts
import engine.loading

device = "cuda"

@pytest.fixture(scope="session")
def experts():
    cfg = transformers.AutoConfig.from_pretrained(engine.loading.MODEL_ID).get_text_config()
    torch.manual_seed(11)
    moe_experts = Qwen3_5MoeExperts(cfg).to(device, torch.bfloat16).eval()
    for p in moe_experts.parameters():
        p.data.normal_(0, 0.02)
    return moe_experts

@pytest.mark.parametrize("T", (1, 8, 63))
def test_grouped_experts(experts, T):
    H, E, K = experts.config.hidden_size, experts.config.num_experts, experts.config.num_experts_per_tok
    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
    w, idx = torch.randn(T, E, device=device, dtype=torch.bfloat16).softmax(-1).topk(K)
    with torch.no_grad():
        torch.testing.assert_close(
            engine.kernels.moe_experts.grouped_experts_forward(experts, hidden, idx, w),
            experts.forward(hidden, idx, w),
            atol=2e-2,
            rtol=2e-2
        )

@pytest.mark.parametrize("T", (1, 8, 63))
def test_fb_grouped_experts(experts, T):
    H, E, K = experts.config.hidden_size, experts.config.num_experts, experts.config.num_experts_per_tok
    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
    w, idx = torch.randn(T, E, device=device, dtype=torch.bfloat16).softmax(-1).topk(K)
    with torch.no_grad():
        torch.testing.assert_close(
            engine.kernels.moe_experts.fbgemm_grouped_experts_forward(experts, hidden, idx, w),
            experts.forward(hidden, idx, w),
            atol=2e-2,
            rtol=2e-2
        )
