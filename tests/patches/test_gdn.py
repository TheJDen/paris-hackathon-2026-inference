import pytest
import torch
import transformers
from transformers.models.qwen3_5_moe.modular_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
import engine.caching
import engine.patches.gdn

device = "cuda"

@pytest.fixture
def cfg():
    return transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B").get_text_config() 

@pytest.mark.parametrize("B", [3, 4, 6])
@pytest.mark.parametrize("P", [7, 16])
def test_gdn(cfg, B, P):
    torch.manual_seed(42)
    S = P + 1
    L = cfg.layer_types.index("linear_attention")

    gdn = Qwen3_5MoeGatedDeltaNet(cfg, L).to(device, torch.bfloat16).eval()
    x = torch.randn(B, S, cfg.hidden_size, device=device, dtype=torch.bfloat16)

    cache = transformers.DynamicCache(config=cfg)
    with torch.no_grad():
        orig_prefill = gdn.forward(x[:, :P], cache_params=cache)
        orig_decode = gdn.forward(x[:, -1:], cache_params=cache)

    NUM_SLOTS = 8
    slotcache = engine.caching.SlotCache(cfg, NUM_SLOTS, S + 8, device)
    slots = torch.randperm(NUM_SLOTS, device=device)[:B].to(torch.int32)
    with torch.no_grad():
        slotted_prefill = engine.patches.gdn._gdn_forward_slotted(gdn, x[:, :P], slotcache=slotcache, slots=slots, decoding=False)
        slotted_decode = engine.patches.gdn._gdn_forward_slotted(gdn, x[:, -1:], slotcache=slotcache, slots=slots, decoding=True)

    torch.testing.assert_close(orig_prefill, slotted_prefill, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(orig_decode, slotted_decode, atol=2e-2, rtol=2e-2)
