import pytest
import torch
import transformers
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention, Qwen3_5MoeTextRotaryEmbedding
import engine.caching
import engine.patches.attn

device = "cuda"

@pytest.fixture
def cfg():
    return transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B").get_text_config() 

@pytest.mark.parametrize("B", [3, 4, 6])
@pytest.mark.parametrize("P", [7, 16])
def test_attn(cfg, B, P):
    torch.manual_seed(42)
    S = P + 1
    L = cfg.layer_types.index("full_attention")

    attn = Qwen3_5MoeAttention(cfg, L).to(device, torch.bfloat16).eval()
    attn.config._attn_implementation = "eager"
    rotary = Qwen3_5MoeTextRotaryEmbedding(cfg).to(device)

    x = torch.randn(B, S, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    pe = rotary(x, torch.arange(S, device=device)[None].expand(B, -1))

    mask = torch.triu(torch.full((S, S), float("-inf"), device=device), 1)
    with torch.no_grad():
        ref, _ = attn.forward(x, pe, attention_mask=mask, past_key_values=None)
    orig_prefill, orig_decode = ref[:, :P], ref[:, -1:]

    NUM_SLOTS = 8
    slotcache = engine.caching.SlotCache(cfg, NUM_SLOTS, S + 8, device)
    slots = torch.randperm(NUM_SLOTS, device=device)[:B].to(torch.int32)
    cos, sin = pe
    pe_prefill = (cos[:, :P], sin[:, :P])
    pe_decode = (cos[:, -1:], sin[:, -1:])
    with torch.no_grad():
        slotted_prefill, _ = engine.patches.attn._attn_forward_slotted(attn, x[:, :P], pe_prefill, slotcache=slotcache, slots=slots)
        slotcache.lens[slots] += P
        slotted_decode, _ = engine.patches.attn._attn_forward_slotted(attn, x[:, -1:], pe_decode, slotcache=slotcache, slots=slots)

    torch.testing.assert_close(orig_prefill, slotted_prefill, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(orig_decode, slotted_decode, atol=2e-2, rtol=2e-2)
