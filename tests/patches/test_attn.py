import itertools

import pytest
import torch
import transformers
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeTextRotaryEmbedding,
)

import engine.caching
import engine.patches.attn
from engine.patches.prefill_index import PrefillIndex

device = "cuda"

@pytest.fixture(scope="session")
def cfg():
    return transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B").get_text_config() 

@pytest.fixture(scope="session")
def attn(cfg):
    L = cfg.layer_types.index("full_attention")
    attn = Qwen3_5MoeAttention(cfg, L).to(device, torch.bfloat16).eval()
    attn.config._attn_implementation = "eager"
    return attn

@pytest.fixture(scope="session")
def rotary(cfg):
    return Qwen3_5MoeTextRotaryEmbedding(cfg).to(device)

def causal_mask(n):
    return torch.triu(torch.full((n, n), float("-inf"), device=device), 1)

@pytest.mark.parametrize("lens", [[3, 7, 5], [16, 1, 9, 4]])
def test_attn(cfg, attn, rotary, lens):
    torch.manual_seed(42)
    B, maxL, H = len(lens), max(lens), cfg.hidden_size

    x = torch.randn(B, maxL + 1, H, device=device, dtype=torch.bfloat16)
    pe = rotary(x, torch.arange(maxL + 1, device=device)[None].expand(B, -1))

    cos, sin = pe
    refs = []
    with torch.no_grad():
        for i, l in enumerate(lens):
            ref, _ = attn.forward(
                x[i:i+1, :l+1],
                (cos[i:i+1, :l+1], sin[i:i+1, :l+1]),
                attention_mask=causal_mask(l + 1),
                past_key_values=None
            )
            refs.append(ref)


    NUM_SLOTS = 8
    slotcache = engine.caching.SlotCache(cfg, NUM_SLOTS, maxL + 8, device)
    slots = torch.randperm(NUM_SLOTS)[:B].to(torch.int32).tolist()

    prefill_index = PrefillIndex.from_lens_and_slots(lens, slots, device)

    x_packed_prefill = torch.cat([x[i, :l] for i, l in enumerate(lens)]).unsqueeze(0)
    pe_packed_prefill = (
        torch.cat([cos[i, :l] for i, l in enumerate(lens)]).unsqueeze(0),
        torch.cat([sin[i, :l] for i, l in enumerate(lens)]).unsqueeze(0)
    )
    b_idx = torch.arange(B, device=device)
    x_decode = x[b_idx, prefill_index.seq_lens].unsqueeze(1)
    pe_decode = (
        cos[b_idx, prefill_index.seq_lens].unsqueeze(1),
        sin[b_idx, prefill_index.seq_lens].unsqueeze(1)
    )

    with torch.no_grad():
        slotted_packed_prefill, _ = engine.patches.attn._attn_forward_slotted(
            attn,
            x_packed_prefill,
            pe_packed_prefill,
            slotcache=slotcache,
            slots=prefill_index.slots,
            prefill_index=prefill_index
        )
        slotcache.advance(prefill_index.slots, prefill_index.seq_lens)
        slotted_decode, _ = engine.patches.attn._attn_forward_slotted(
            attn,
            x_decode,
            pe_decode,
            slotcache=slotcache,
            slots=prefill_index.slots
        )

    orig_prefill = torch.cat([ref[0, :l] for ref, l in zip(refs, prefill_index.seq_lens)])
    orig_decode = torch.stack([ref[0, l] for ref, l in zip(refs, lens)]).unsqueeze(1)

    slotted_prefill = torch.cat([
        slotted_packed_prefill[0, start:end] for start, end in itertools.pairwise(prefill_index.cu_seqlens)
    ])

    torch.testing.assert_close(orig_prefill, slotted_prefill, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(orig_decode, slotted_decode, atol=2e-2, rtol=2e-2)
