import torch
import tqdm

import engine.batching
import engine.caching
import engine.records
import engine.sampling


class ModelRunner:
    def __init__(self, model, slot_cache: engine.caching.SlotCache, active_sequences: engine.batching.ActiveSequences):
        self.model = model
        self.slot_cache = slot_cache
        self.active_sequences = active_sequences
        self.graphs = {B: torch.cuda.CUDAGraph() for B in range(1, 65)}
        self.pool = torch.cuda.graph_pool_handle()
        self.next_tokens_buffer = torch.zeros(64, 1, dtype=torch.long, device=model.device)
        self.pinned_tokens = torch.zeros(64, dtype=torch.long, pin_memory=True)
        self.copy_event = torch.cuda.Event()
        self.advance_buffer = torch.ones(64, dtype=torch.int32, device=model.device)

    def _prefill(
        self,
        b: engine.records.PrefillInputs,
        slots: torch.Tensor
    ) -> torch.Tensor:
        h_flat = self.model.model(
            b.tokens,
            position_ids=b.position_ids.unsqueeze(0),
            slotcache=self.slot_cache,
            slots=slots,
            dest_slot=slots.repeat_interleave(b.seq_lens),
            prefill_inputs=b
        ).last_hidden_state
        logits = self.model.lm_head(h_flat[0, b.cu_seqlens[1:] - 1])
        next_toks = engine.sampling.sample_next(logits, b.temp, b.top_p)
        self.slot_cache.begin(slots, b.seq_lens)
        return next_toks

    def prefill(self) -> torch.Tensor:
        with torch.profiler.record_function("prefill"), torch.inference_mode():
            seqs = self.active_sequences.get_prefill_seqs()
            slots = self.active_sequences.get_prefill_slots()
            b = engine.records.PrefillInputs.from_seqs(seqs, device=self.model.device)
            next_toks = self._prefill(b, slots)
        return next_toks.squeeze(1)

    def _decode(self, B: int) -> torch.Tensor:
        with torch.inference_mode():
            slots = self.active_sequences.slots[:B]
            b = self.active_sequences.get_decode_batch(slots)
            h = self.model.model(
                b.tokens,
                slotcache=self.slot_cache,
                position_ids=self.slot_cache.lens[slots].unsqueeze(1),
                slots=slots,
            ).last_hidden_state
            logits = self.model.lm_head(h[:, -1, :])
            next_toks = engine.sampling.sample_next(logits, b.temp, b.top_p)
            self.active_sequences.commit(slots, next_toks)
            self.slot_cache.advance(slots, self.advance_buffer[:B])
            self.next_tokens_buffer[:B].copy_(next_toks)
        return next_toks

    def capture(self, B):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                next_toks = self._decode(B)
                self.next_tokens_buffer[:B].copy_(next_toks)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self.graphs[B], pool=self.pool):
            next_toks = self._decode(B)
            self.next_tokens_buffer[:B].copy_(next_toks)

    def decode(self) -> torch.Tensor:
        B = self.active_sequences.B
        with torch.profiler.record_function("decode"), torch.inference_mode():
            self.graphs[B].replay()
            self.copy_event.record()
            self.copy_event.synchronize()
            self.pinned_tokens[:B].copy_(self.next_tokens_buffer[:B, 0], non_blocking=True)
        return self.pinned_tokens[:B]

    def warmup(self):
        for L in tqdm.tqdm((64, 128, 256, 512, 1024, 2048), desc="Prefill warmup shapes"):
            lens = [L // 4, L // 2, L // 4]
            slots = torch.tensor([1, 3, 7], device=self.model.device)
            b = engine.records.PrefillInputs.from_tokens(
                input_ids=[
                    torch.zeros(n, dtype=torch.long, device=self.model.device)
                    for n in lens
                ],
                temp=torch.zeros(len(lens), device=self.model.device),
                top_p=torch.ones(len(lens), device=self.model.device),
                device=self.model.device
            )
            self._prefill(b, slots)
        for B in tqdm.tqdm(self.graphs, desc="Decode warmup shapes"):
            self.capture(B)

