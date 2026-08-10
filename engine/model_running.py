import torch
import tqdm

import engine.caching
from engine.patches.prefill_index import PrefillIndex


class ModelRunner:
    def __init__(self, model, slotcache: engine.caching.SlotCache):
        self.model = model
        self.slotcache = slotcache
        self.graphs = {B: torch.cuda.CUDAGraph() for B in range(1, 65)}
        self.pool = torch.cuda.graph_pool_handle()
        self.tokens_buffer = torch.zeros(64, 1, dtype=torch.long, device=model.device)
        self.slots_buffer = torch.zeros(64, dtype=torch.int32, device=model.device)
        self.logits_buffer = torch.empty(64, model.config.vocab_size, dtype=torch.bfloat16, device=model.device)
        self.advance_buffer = torch.ones(64, dtype=torch.int32, device=model.device)

    def prefill(self, input_ids: list[torch.Tensor], slots: list[int]) -> torch.Tensor:
        lens = [len(x) for x in input_ids]
        with torch.profiler.record_function("prefill"), torch.inference_mode():
            prefill_index = PrefillIndex.from_lens_and_slots(lens, slots, device=self.model.device)
            h_flat = self.model.model(
                torch.cat([x.to(self.model.device) for x in input_ids]).unsqueeze(0),
                position_ids=prefill_index.position_ids.unsqueeze(0),
                slotcache=self.slotcache,
                slots=prefill_index.slots,
                prefill_index=prefill_index
            ).last_hidden_state
            logits = self.model.lm_head(h_flat[0, prefill_index.cu_seqlens[1:] - 1])
            self.slotcache.advance(prefill_index.slots, prefill_index.seq_lens)
        return logits

    def _decode(self, tokens: torch.Tensor, slots: torch.Tensor, advance_lens: torch.Tensor) -> torch.Tensor:
        h = self.model.model(
            tokens,
            slotcache=self.slotcache,
            position_ids=self.slotcache.lens[slots].unsqueeze(1),
            slots=slots,
        ).last_hidden_state
        logits = self.model.lm_head(h[:, -1, :])
        self.slotcache.advance(slots, advance_lens)
        return logits

    def capture(self, B):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                logits = self._decode(
                    self.tokens_buffer[:B],
                    self.slots_buffer[:B],
                    self.advance_buffer[:B]
                )
                self.logits_buffer[:B].copy_(logits)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self.graphs[B], pool=self.pool):
            logits = self._decode(
                self.tokens_buffer[:B],
                self.slots_buffer[:B],
                self.advance_buffer[:B]
            )
            self.logits_buffer[:B].copy_(logits[:B])
        self.slotcache.lens[0] = 0 # kinda hacky but we are adults

    def decode(self, tokens: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        with torch.profiler.record_function("decode"), torch.inference_mode():
            B = tokens.shape[0]
            self.tokens_buffer[:B].copy_(tokens)
            self.slots_buffer[:B].copy_(slots)
            self.graphs[B].replay()
        return self.logits_buffer[:B]

    def warmup(self):
        for L in tqdm.tqdm((128, 256, 512, 1024, 2048), desc="Prefill warmup shapes"):
            slots = [self.slotcache.alloc()]
            self.prefill([torch.zeros(L, dtype=torch.long, device=self.model.device)], slots)
            for slot in slots:
                self.slotcache.release(slot)
        for B in tqdm.tqdm(self.graphs, desc="Decode warmup shapes"):
            self.capture(B)
