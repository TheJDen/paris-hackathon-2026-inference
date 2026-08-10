import collections
import dataclasses
import torch
import engine.caching
import engine.records
import engine.sampling
import engine.protocols

@dataclasses.dataclass(kw_only=True)
class SeqState:
    slot: int
    item: engine.records.WorkItem
    stop_ids: set[int]
    max_tokens: int
    prompt_len: int
    generated: list[int] = dataclasses.field(default_factory=list)

    def advance(self, tok: int) -> str | None:
        if tok in self.stop_ids:
            return "stop"
        self.generated.append(tok)
        return "length" if len(self.generated) >= self.max_tokens else None

@dataclasses.dataclass(kw_only=True)
class DecodeBatch:
    tokens: torch.Tensor
    temp: torch.Tensor
    top_p: torch.Tensor
    slots: torch.Tensor
    seqs: list[SeqState]


class ActiveBatch:
    def __init__(self, capacity, device):
        self.tok = torch.zeros(capacity, 1, dtype=torch.long, device=device)
        self.temp = torch.zeros(capacity, device=device)
        self.top_p = torch.zeros(capacity, device=device)
        self.seqs_by_slot = {}

    def add(self, seq: SeqState):
        self.temp[seq.slot] = seq.item.req.temperature
        self.top_p[seq.slot] = seq.item.req.top_p
        self.tok[seq.slot] = seq.generated[-1]
        self.seqs_by_slot[seq.slot] = seq

    def remove(self, seq: SeqState):
        del self.seqs_by_slot[seq.slot]

    def commit(self, b: DecodeBatch, next_tok):
        self.tok[b.slots] = next_tok.unsqueeze(1)

    def batch(self) -> DecodeBatch:
        seqs = list(self.seqs_by_slot.values())
        slots = torch.tensor([seq.slot for seq in seqs], dtype=torch.int32, device=self.tok.device)
        return DecodeBatch(
            tokens=self.tok[slots],
            temp=self.temp[slots],
            top_p=self.top_p[slots],
            slots=slots,
            seqs=seqs,
        )

    def __len__(self):
        return len(self.seqs_by_slot)


class ContinuousBatcher:
    collect_window = 0.0
    def __init__(
        self,
        model_runner: engine.protocols.ModelRunner,
        slotcache: engine.caching.SlotCache,
        stop_ids: set[int],
    ):
        self.model_runner = model_runner
        self.slotcache = slotcache
        self.stop_ids = stop_ids
        self.waiting = collections.deque()
        self.active = ActiveBatch(self.slotcache.capacity, self.slotcache.lens.device)

    def add(self, items: list[engine.records.WorkItem]):
        self.waiting.extend(items)

    def step(self):
        prefill_items = [self.waiting.popleft() for _ in range(min(len(self.waiting), self.slotcache.num_free_slots()))]
        if prefill_items:
            yield from self._prefill(prefill_items)
        if self.active:
            yield from self._decode()

    def has_work(self):
        return bool(self.waiting or len(self.active))

    def _prefill(self, items: list[engine.records.WorkItem]):
        slots = [self.slotcache.alloc() for _ in range(len(items))]
        input_ids = [item.input_ids for item in items]
        logits = self.model_runner.prefill(input_ids, slots)
        temps = torch.tensor([item.req.temperature for item in items], device=logits.device)
        top_p = torch.tensor([item.req.top_p for item in items], device=logits.device)
        next_toks = engine.sampling.sample_next(logits, temperature=temps, top_p=top_p).squeeze(1).tolist()
        for slot, item, next_tok in zip(slots, items, next_toks):
            seq = SeqState(
                slot=slot,
                item=item,
                prompt_len=len(item.input_ids),
                stop_ids=self.stop_ids,
                max_tokens=item.req.max_tokens
            )
            stop_reason = seq.advance(next_tok)
            if stop_reason is None:
                self.active.add(seq)
            else:
                self.slotcache.release(seq.slot)
                yield item, self._complete(seq, stop_reason)

    def _decode(self):
        b = self.active.batch()
        logits = self.model_runner.decode(b.tokens, b.slots)
        next_tok = engine.sampling.sample_next(logits, b.temp, b.top_p).squeeze(1)
        self.active.commit(b, next_tok)
        for seq, tok in zip(b.seqs, next_tok.tolist()):
            stop_reason = seq.advance(tok)
            if stop_reason is not None:
                item = seq.item
                yield item, self._finish(seq, stop_reason)


    def _complete(self, seq_state: SeqState, stop_reason: str):
        return engine.records.RawResult(
            seq_state.generated,
            seq_state.prompt_len,
            stop_reason
        )

    def _finish(self, seq: SeqState, stop_reason: str):
        raw_result = self._complete(seq, stop_reason)
        self.slotcache.release(seq.slot)
        self.active.remove(seq)
        return raw_result

    def abort_items(self):
        items = []
        for seq in list(self.active.seqs_by_slot.values()):
            self.active.remove(seq)
            self.slotcache.release(seq.slot)
            items.append(seq.item)
        return items

    def get_num_active(self) -> int:
        return len(self.active)


class StaticBatch:
    def __init__(self, model, stop_ids, items: list[engine.records.WorkItem]):
        self.model = model
        self.stop_ids = stop_ids
        self.items = items
        self.temperature = torch.tensor([item.req.temperature for item in items], device=model.device)
        self.top_p = torch.tensor([item.req.top_p for item in items], device=model.device)

        ids = [item.input_ids.squeeze(0) for item in items]
        B, L = len(ids), max(x.shape[0] for x in ids)
        self.input_tokens = torch.zeros(B, L, dtype=torch.long, device=model.device)
        self.attn_mask = torch.zeros(B, L, dtype=torch.long, device=model.device)
        for i, x in enumerate(ids):
            self.input_tokens[i, L - x.shape[0]:] = x.to(model.device)
            self.attn_mask[i, L - x.shape[0]:] = 1

        self.budgets = [item.req.max_tokens for item in items]
        self.active = set(range(len(items)))
        self.generations = [[] for _ in range(len(items))]

        self.prompt_lengths = self.attn_mask.sum(1)
        self.padded_length = L
        self.position_ids = (self.attn_mask.long().cumsum(-1) - 1).clamp(min=0)
        self.cache_position = None
        self.past_key_values = None
        self.step_idx = 0

    def is_done(self) -> bool:
        return not self.active

    def update(self, req_id, tok):
        self.generations[req_id].append(tok)
        if len(self.generations[req_id]) == self.budgets[req_id]:
            return self.complete(req_id, "length")

    def complete(self, req_id, finish_reason="stop"):
        self.active.remove(req_id)
        return engine.records.RawResult(
            self.generations[req_id],
            prompt_len=int(self.prompt_lengths[req_id]),
            finish_reason=finish_reason
        )

    def step(self) -> dict[int, engine.records.Completion]:
        completions_by_id = {}
        with torch.inference_mode():
            with torch.profiler.record_function("prefill" if self.step_idx == 0 else "decode"):
                out = self.model(
                    self.input_tokens,
                    attention_mask=self.attn_mask,
                    past_key_values=self.past_key_values,
                    position_ids=self.position_ids,
                    cache_position=self.cache_position,
                    use_cache=True,
                    logits_to_keep=1
                    )

            with torch.profiler.record_function("sample"):
                next_tok = engine.sampling.sample_next(out.logits[:, -1, :], temperature=self.temperature, top_p=self.top_p)
            for req_id in set(self.active):
                tok = next_tok[req_id].item()
                if tok in self.stop_ids:
                    completions_by_id[req_id] = self.complete(req_id)
                    continue
                completion = self.update(req_id, tok)
                if completion is not None:
                    completions_by_id[req_id] = completion

            self.input_tokens = next_tok
            self.attn_mask = torch.cat([self.attn_mask, self.attn_mask.new_ones(self.attn_mask.shape[0], 1)], dim=1)
            self.position_ids = (self.prompt_lengths + self.step_idx).unsqueeze(1)
            self.cache_position = torch.tensor([self.padded_length + self.step_idx], device=self.model.device)
            self.past_key_values = out.past_key_values
            self.step_idx += 1
        return completions_by_id

class StaticBatcher:
    collect_window = 0.2
    def __init__(self, model, stop_ids):
        self.model = model
        self.stop_ids = stop_ids
        self.pending = collections.deque()
        self.batch = None

    def add(self, items: list[engine.records.WorkItem]):
        self.pending.extend(items)

    def has_work(self) -> bool:
        return bool(self.pending or self.batch is not None)

    def step(self):
        if self.batch is None and self.pending:
            self.batch = StaticBatch(self.model, self.stop_ids, list(self.pending))
            self.pending.clear()
        if self.batch is None:
            return
        for i, raw in self.batch.step().items():
            yield self.batch.items[i], raw
        if self.batch.is_done():
            self.batch = None

    def abort_items(self) -> list[engine.records.WorkItem]:
        items = (self.batch.items if self.batch is not None else []) + list(self.pending)
        self.batch = None
        self.pending.clear()
        return items

    def get_num_active(self) -> int:
        return len(self.batch.items) if self.batch else 0

