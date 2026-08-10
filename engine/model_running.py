import torch

import engine.caching
from engine.patches.prefill_index import PrefillIndex


class ModelRunner:
    def __init__(self, model, slotcache: engine.caching.SlotCache):
        self.model = model
        self.slotcache = slotcache

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

    def decode(self, tokens: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        with torch.profiler.record_function("decode"), torch.inference_mode():
            h = self.model.model(
                tokens,
                slotcache=self.slotcache,
                position_ids=self.slotcache.lens[slots].unsqueeze(1),
                slots=slots,
            ).last_hidden_state
            logits = self.model.lm_head(h[:, -1, :])
            self.slotcache.advance(slots, torch.ones(tokens.shape[1], device=tokens.device, dtype=torch.int32))
        return logits
