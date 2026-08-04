import torch
import engine.caching

class ModelRunner:
    def __init__(self, model, slotcache: engine.caching.SlotCache):
        self.model = model
        self.slotcache = slotcache

    def prefill(self, input_ids: torch.Tensor, slot: int) -> torch.Tensor:
        slots = torch.tensor([slot], dtype=torch.int32, device=self.model.device)
        with torch.profiler.record_function("prefill"), torch.inference_mode():
            h = self.model.model(
                input_ids,
                past_key_values=None,
                use_cache=False,
                slotcache=self.slotcache,
                slots=slots,
                decoding=False,
            ).last_hidden_state
            logits = self.model.lm_head(h[:, -1, :])
        self.slotcache.advance(slot, input_ids.shape[1])
        return logits

    def decode(self, tokens: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        with torch.profiler.record_function("decode"), torch.inference_mode():
            h = self.model.model(
                tokens,
                past_key_values=None,
                use_cache=False,
                slotcache=self.slotcache,
                position_ids=self.slotcache.lens[slots].unsqueeze(1),
                slots=slots,
                decoding=True,
            ).last_hidden_state
            logits = self.model.lm_head(h[:, -1, :])
        self.slotcache.advance(slots, tokens.shape[1])
        return logits
