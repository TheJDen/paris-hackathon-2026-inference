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

class FakeRunner:
    def __init__(
            self,
            active: engine.batching.ActiveSequences,
            script
    ):
        self.active = active
        self.script = script
    def prefill(self):
        toks = [self.script[s.id].pop(0) for s in self.active.get_prefill_seqs()]
        return torch.tensor(toks)
    def decode(self):
        toks = [self.script[s.id].pop(0) for s in self.active.get_decode_seqs()]
        return torch.tensor(toks)
    def warmup(self):
        pass


mock_t = tuple[
    engine.scheduling.ContinuousScheduler,
    engine.batching.ActiveSequences,
]

@pytest.fixture
def mock_factory():
    def factory(script, num_slots, stop_ids={STOP_ID}):
        active_seqs = engine.batching.ActiveSequences(num_slots)
        runner = FakeRunner(active_seqs, script)
        scheduler = engine.scheduling.ContinuousScheduler(runner, active_seqs, set(stop_ids))
        return scheduler, active_seqs
    return factory

def drain(sched, n):
    out = []
    for _ in range(n):
        out.extend(sched.step())
    return out

def test_capped_by_free_slots(mock_factory):
    script = {f"r{i}": list(range(10, 40)) for i in range(6)}
    mock: mock_t = mock_factory(script, num_slots=4)
    scheduler, active_seqs = mock
    items = [make_item(f"r{i}") for i in range(6)]
    scheduler.add(items)
    list(scheduler.step())
    assert len(active_seqs.get_decode_seqs()) == 4
    assert active_seqs.num_free() == 0

def test_first_token_stops(mock_factory): # prefill, no decode
    script = {"a": [STOP_ID], "b": list(range(10, 20))}
    mock: mock_t = mock_factory(script, num_slots=4)
    scheduler, active_seqs = mock
    items = [make_item(rid) for rid in "ab"]
    scheduler.add(items)
    done = {item.id: res for item, res in scheduler.step()}
    assert done["a"].finish_reason == "stop" and len(done["a"].generated) == 0
    assert [s.id for s in active_seqs.get_decode_seqs()] == ["b"]
    assert active_seqs.num_free() == 3

def test_decode_stops(mock_factory):
    script = {"a": [21, 47, STOP_ID], "b": list(range(20, 40))}
    mock: mock_t = mock_factory(script, num_slots=4)
    scheduler, active_seqs = mock
    items = [make_item(rid) for rid in "ab"]
    scheduler.add(items)
    done = {item.id: res for item, res in drain(scheduler, 3)}
    assert done["a"].finish_reason == "stop" and done["a"].generated == [21, 47]
    assert [s.id for s in active_seqs.get_decode_seqs()] == ["b"]
    assert active_seqs.num_free() == 3

def test_length_stops(mock_factory):
    script = {"b": list(range(10, 20))}
    mock: mock_t = mock_factory(script, num_slots=4)
    scheduler, _ = mock
    items = [make_item("b", max_tokens=3)]
    scheduler.add(items)
    z = next(res for i, res in drain(scheduler, 5))
    assert z.finish_reason == "length" and len(z.generated) == 3
