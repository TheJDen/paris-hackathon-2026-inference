import collections

import torch

import engine.batching
import engine.caching
import engine.protocols
import engine.records
import engine.sampling


class ContinuousScheduler:
    collect_window = 0.0
    def __init__(
        self,
        model_runner: engine.protocols.ModelRunner,
        active_sequences: engine.batching.ActiveSequences,
        stop_ids: set[int],
    ):
        self.model_runner = model_runner
        self.active_sequences = active_sequences
        self.stop_ids = stop_ids
        self.waiting = collections.deque()

    def add(self, items: list[engine.records.WorkItem]):
        self.waiting.extend(items)

    def step(self):
        prefill_items = [self.waiting.popleft() for _ in range(min(len(self.waiting), self.active_sequences.num_free()))]
        if prefill_items:
            yield from self._prefill(prefill_items)
        if self.active_sequences:
            yield from self._decode()

    def has_work(self):
        return bool(self.waiting or len(self.active_sequences))

    def _prefill(self, items: list[engine.records.WorkItem]):
        for item in items:
            seq = engine.records.SeqState(
                item=item,
                prompt_len=len(item.input_ids),
                stop_ids=self.stop_ids,
                max_tokens=item.req.max_tokens
            )
            self.active_sequences.to_prefill(seq)
        next_toks = self.model_runner.prefill()
        for seq, next_tok in zip(self.active_sequences.get_prefill_seqs(), next_toks.tolist()):
            stop_reason = seq.advance(next_tok)
            if stop_reason is None:
                self.active_sequences.to_decode(seq.id)
            else:
                self.active_sequences.remove(seq.id)
                yield seq.item, self._complete(seq, stop_reason)

    def _decode(self):
        next_toks = self.model_runner.decode()
        for seq, tok in zip(self.active_sequences.get_decode_seqs(), next_toks.tolist()):
            stop_reason = seq.advance(tok)
            if stop_reason is not None:
                item = seq.item
                yield item, self._finish(seq, stop_reason)


    def _complete(self, seq_state: engine.records.SeqState, stop_reason: str):
        return engine.records.RawResult(
            seq_state.generated,
            seq_state.prompt_len,
            stop_reason
        )

    def _finish(self, seq: engine.records.SeqState, stop_reason: str):
        raw_result = self._complete(seq, stop_reason)
        self.active_sequences.remove(seq.id)
        return raw_result

    def abort_items(self):
        items = []
        seqs = self.active_sequences.get_prefill_seqs() + self.active_sequences.get_decode_seqs()
        for seq in seqs:
            self.active_sequences.remove(seq.id)
            items.append(seq.item)
        return items

    def warmup(self):
        self.model_runner.warmup()

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


class StaticScheduler:
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

    def warmup(self):
        pass # give me a break

