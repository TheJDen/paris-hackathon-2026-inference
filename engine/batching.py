import torch
import engine.model

class StaticBatcher:
    def __init__(self, model_bundle: engine.model.ModelBundle, requests: list[engine.model.CompletionRequest]):
        self.model_bundle = model_bundle
        self.temperature = torch.tensor([req.temperature for req in requests], device=self.model_bundle.model.device)
        self.top_p = torch.tensor([req.top_p for req in requests], device=self.model_bundle.model.device)

        enc = self.model_bundle.tokenizer.apply_chat_template(
                [req.messages for req in requests],
                add_generation_prompt=True,
                enable_thinking=False,
                padding=True,
                return_tensors="pt"
        )

        self.attn_mask = enc.attention_mask.to(self.model_bundle.model.device)
        self.budgets = [req.max_tokens for req in requests]
        self.active = set(range(len(requests)))
        self.generations = [[] for _ in range(len(requests))]

        self.prompt_lengths = self.attn_mask.sum(1)
        self.padded_lengths = enc.input_ids.shape[1]
        self.input_tokens = enc.input_ids.to(self.model_bundle.model.device)
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
        text = self.model_bundle.tokenizer.decode(self.generations[req_id], skip_special_tokens=True) 
        self.active.remove(req_id)
        return engine.model.Completion(
                text=text,
                prompt_tokens=int(self.prompt_lengths[req_id]),
                completion_tokens=len(self.model_bundle.tokenizer.encode(text, add_special_tokens=False)),
                finish_reason=finish_reason
        )

    def step(self) -> dict[int, engine.model.Completion]:
        completions_by_id = {}
        with torch.inference_mode():
            with torch.profiler.record_function("prefill" if self.step_idx == 0 else "decode"):
                out = self.model_bundle.model(
                    self.input_tokens,
                    attention_mask=self.attn_mask,
                    past_key_values=self.past_key_values,
                    position_ids=self.position_ids,
                    cache_position=self.cache_position,
                    use_cache=True,
                    logits_to_keep=1
                    )

            with torch.profiler.record_function("sample"):
                next_tok = engine.model.sample_next(out.logits[:, -1, :], temperature=self.temperature, top_p=self.top_p)
            for req_id in set(self.active):
                tok = next_tok[req_id].item()
                if tok in self.model_bundle.stop_ids:
                    completions_by_id[req_id] = self.complete(req_id)
                    continue
                completion = self.update(req_id, tok)
                if completion is not None:
                    completions_by_id[req_id] = completion

            self.input_tokens = next_tok
            self.attn_mask = torch.cat([self.attn_mask, self.attn_mask.new_ones(self.attn_mask.shape[0], 1)], dim=1)
            self.position_ids = (self.prompt_lengths + self.step_idx).unsqueeze(1)
            self.cache_position = torch.tensor([self.padded_lengths + self.step_idx], device=self.model_bundle.model.device)
            self.past_key_values = out.past_key_values
            self.step_idx += 1
        return completions_by_id
