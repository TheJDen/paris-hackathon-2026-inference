import pytest
import torch

import engine.batching
import engine.protocols
import engine.records
import engine.scheduling

STOP_ID = 0

def make_item(rid, prompt_len=4, max_tokens=100, temp=0.0, top_p=1.0):
    req = engine.records.CompletionRequest(
        messages=[],
        max_tokens=max_tokens,
        temperature=temp,
        top_p=top_p
    )
    return engine.records.WorkItem(
        id=rid,
        req=req,
        future=None,
        loop=None,
        enqueue_ts=0.0,
        input_ids=torch.zeros(prompt_len, dtype=torch.long)
    )


def test_gather_maintains_correct_state():
    a = engine.batching.ActiveSequences(8)

    def admit(rid, temp):
        item = make_item(rid, temp=temp)
        seq = engine.records.SeqState(
            item=item,
            prompt_len=4,
            stop_ids={STOP_ID},
            max_tokens=100,
        )
        a.to_prefill(seq)
        seq.generated.append(1)
        a.to_decode(rid)

    def check():
        slots = a.get_decode_slots()
        batch = a.get_decode_batch(slots)
        for i, seq in enumerate(a.get_decode_seqs()):
            assert batch.temp[i] == seq.temp

    admit("a", 0.1)
    admit("b", 0.2)
    admit("c", 0.3)
    admit("d", 0.4)
    admit("e", 0.5)
    check()
    a.remove("b") # middle (swap)
    check()
    a.remove("e") # last
    check()
    a.remove("a") # first
    check()
    admit("f", 0.6) # reuse
    check()


