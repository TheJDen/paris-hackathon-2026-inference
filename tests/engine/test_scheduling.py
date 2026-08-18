import pytest
import torch

import engine.batching
import engine.cuda
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
        self.i = 0
    def prefill(self):
        toks = [self.script[s.id][min(self.i, len(self.script[s.id]) - 1)] for s in self.active.get_prefill_seqs()]
        self.i += 1
        return torch.tensor(toks)
    def decode(self):
        toks = [self.script[s.id][min(self.i, len(self.script[s.id]) - 1)] for s in self.active.get_decode_seqs()]
        self.i += 1
        return torch.tensor(toks)
    def warmup(self):
        pass


mock_t = tuple[
    engine.scheduling.ContinuousScheduler,
    engine.batching.ActiveSequences,
]

@pytest.fixture
def mock_factory():
    def factory(script, num_slots, depth=2, stop_ids={STOP_ID}):
        active_seqs = engine.batching.ActiveSequences(num_slots)
        runner = FakeRunner(active_seqs, script)
        executor = engine.cuda.EventExecutor(
            num_events=depth, 
            out_capacity=num_slots,
            dtype=torch.long
        )
        scheduler = engine.scheduling.ContinuousScheduler(
            runner,
            active_seqs,
            executor,
            set(stop_ids)
        )
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

def test_depth1_equals_depth2(mock_factory):                    # the core pipelining guarantee
    script = {
        "a": [11, 12, 13, STOP_ID],
        "b": [21, 22, 23, 24, 25, STOP_ID],
        "c": [31, STOP_ID]
    }
    def gens(depth):
        scheduler, _ = mock_factory({k: list(v) for k, v in script.items()}, num_slots=8, depth=depth)
        items = [make_item(rid) for rid in "ab"]
        scheduler.add(items)
        return {item.id: res for item, res in drain(scheduler, 6)}
    return gens(1) == gens(2)
