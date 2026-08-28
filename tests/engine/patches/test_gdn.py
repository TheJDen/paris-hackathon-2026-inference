import itertools

import pytest
import torch
import transformers
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

import engine.caching
import engine.patches.gdn
from engine.records import PrefillInputs

device = "cuda"

@pytest.fixture(scope="session")
def cfg():
    return transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B").get_text_config() 

@pytest.fixture(scope="session")
def gdn(cfg):
    L = cfg.layer_types.index("linear_attention")
    attn = Qwen3_5MoeGatedDeltaNet(cfg, L).to(device, torch.bfloat16).eval()
    return attn

@pytest.mark.parametrize("lens", [[3, 7, 5], [16, 1, 9, 4]])
def test_gdn_prefill_decode(cfg, gdn, lens):
    torch.manual_seed(42)
    B, maxL, H = len(lens), max(lens), cfg.hidden_size

    x = torch.randn(B, maxL + 1, H, device=device, dtype=torch.bfloat16)
    orig_decode = torch.empty(B, 1, H, device=device, dtype=torch.bfloat16)

    orig_prefills = []
    with torch.no_grad():
        for i, l in enumerate(lens):
            cache = transformers.DynamicCache(config=cfg)
            orig_prefill = gdn.forward(x[i:i+1, :l], cache_params=cache)
            orig_decode[i] = gdn.forward(x[i:i+1, l:l+1], cache_params=cache)[0]
            orig_prefills.append(orig_prefill[0])

    NUM_SLOTS = 8
    slotcache = engine.caching.SlotCache(cfg, NUM_SLOTS, maxL + 8, device)
    slots = torch.randperm(NUM_SLOTS)[:B].to(device, torch.int32)

    prefill_inputs = PrefillInputs.from_tokens(
        input_ids=[torch.zeros(l, dtype=torch.long, device=device) for l in lens],
        temp=torch.zeros(B, device=device),
        top_p=torch.ones(B, device=device),
        device=device,
    )
    dest_slot = slots.repeat_interleave(prefill_inputs.seq_lens)

    x_packed_prefill = torch.cat([x[i, :l] for i, l in enumerate(lens)]).unsqueeze(0)
    b_idx = torch.arange(B, device=device)
    x_decode = x[b_idx, prefill_inputs.seq_lens].unsqueeze(1)
    with torch.no_grad():
        slotted_packed_prefill = engine.patches.gdn._gdn_forward_slotted(
            gdn,
            x_packed_prefill,
            slotcache=slotcache,
            slots=slots,
            dest_slot=dest_slot,
            prefill_inputs=prefill_inputs,
            mode="prefill"
        )
        slotted_decode = engine.patches.gdn._gdn_forward_slotted(
            gdn,
            x_decode,
            slotcache=slotcache,
            slots=slots,
            mode="decode"
        )

    slotted_prefills = (slotted_packed_prefill[0, start:end] for start, end in itertools.pairwise(prefill_inputs.cu_seqlens))
    for orig_prefill, slotted_prefill in zip(orig_prefills, slotted_prefills):
        torch.testing.assert_close(orig_prefill, slotted_prefill, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(orig_decode, slotted_decode, atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("k,n", [(k, n) for k in (1, 2, 3) for n in range(k + 1)])
def test_gdn_verify(cfg, gdn, k, n):
    L = gdn.layer_idx
    H = cfg.hidden_size
    MAX_LENGTH = 20
    slotcache = engine.caching.SlotCache(
        cfg,
        num_slots=8,
        max_len=MAX_LENGTH,
        device=device
    )
    checkpointer = engine.caching.GDNCheckpointer(slotcache, k)
    decode_slot, verify_slot = torch.tensor([0], device=device), torch.tensor([1], device=device)

    warm = [torch.randn((1, 1, H), dtype=torch.bfloat16, device=device) for _ in range(3)]
    body = [torch.randn((1, 1, H), dtype=torch.bfloat16, device=device) for _ in range(k + 1)]
    cont = [torch.randn((1, 1, H), dtype=torch.bfloat16, device=device) for _ in range(3)]

    dconv, drec = slotcache.read_gdn(L, decode_slot)
    vconv, vrec = slotcache.read_gdn(L, verify_slot)

    slotcache.update_gdn(L, decode_slot, torch.zeros_like(dconv), torch.zeros_like(drec))
    slotcache.update_gdn(L, verify_slot, torch.zeros_like(vconv), torch.zeros_like(vrec))
    with torch.no_grad():
        for t in warm:
            engine.patches.gdn._gdn_forward_slotted(
                gdn,
                t,
                slotcache=slotcache,
                slots=decode_slot,
                mode="decode"
            )
            engine.patches.gdn._gdn_forward_slotted(
                gdn,
                t,
                slotcache=slotcache,
                slots=verify_slot,
                mode="decode"
            )

        for t in body[:n + 1]:
            engine.patches.gdn._gdn_forward_slotted(
                gdn,
                t,
                slotcache=slotcache,
                slots=decode_slot,
                mode="decode"
            )


        engine.patches.gdn._gdn_forward_slotted(
            gdn,
            torch.cat(body, dim=1),
            slotcache=slotcache,
            slots=verify_slot,
            mode="verify",
            checkpointer=checkpointer,
            spec_k=k
        )
        checkpointer.promote(verify_slot, n)

        decode_cont = []
        verify_cont = []
        for t in cont:
            decode_cont.append(engine.patches.gdn._gdn_forward_slotted(
                gdn,
                t,
                slotcache=slotcache,
                slots=decode_slot,
                mode="decode"
            ))
            verify_cont.append(engine.patches.gdn._gdn_forward_slotted(
                gdn,
                t,
                slotcache=slotcache,
                slots=verify_slot,
                mode="decode"
            ))

    torch.testing.assert_close(
        torch.cat(decode_cont, dim=1),
        torch.cat(verify_cont, dim=1),
        rtol=2e-2,
        atol=2e-2
    )
