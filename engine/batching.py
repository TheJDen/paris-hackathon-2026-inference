import torch
import engine.model

class StaticBatcher:
    def __init__(self, model, tokenizer, stop_ids, requests: list[engine.model.CompletionRequest]):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_ids = stop_ids
        self.temperature = torch.tensor([req.temperature for req in requests], device=self.model.device)
        self.top_p = torch.tensor([req.top_p for req in requests], device=self.model.device)

        enc = tokenizer.apply_chat_template(
                [req.messages for req in requests],
                add_generation_prompt=True,
                enable_thinking=False,
                padding=True,
                return_tensors="pt"
        )

        self.attn_mask = enc.attention_mask.to(self.model.device)
        self.budgets = [req.max_tokens for req in requests]
        self.active = set(range(len(requests)))
        self.generations = [[] for _ in range(len(requests))]

        self.prompt_lengths = self.attn_mask.sum(1)
        self.padded_lengths = enc.input_ids.shape[1]
        self.step_idx = 0

        with torch.inference_mode():
            with torch.profiler.record_function("prefill"):
                self.out = self.model(
                        enc.input_ids.to(self.model.device),
                        attention_mask=self.attn_mask,
                        use_cache=True
                        )

    def is_done(self) -> bool:
        return not self.active

    def update(self, req_id, tok):
        self.generations[req_id].append(tok)
        if len(self.generations[req_id]) == self.budgets[req_id]:
            return self.complete(req_id, "length")

    def complete(self, req_id, finish_reason="stop"):
        text = self.tokenizer.decode(self.generations[req_id], skip_special_tokens=True) 
        self.active.remove(req_id)
        return engine.model.Completion(
                text=text,
                prompt_tokens=int(self.prompt_lengths[req_id]),
                completion_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
                finish_reason=finish_reason
        )

    def step(self) -> dict[int, engine.model.Completion]:
        completions_by_id = {}
        with torch.inference_mode():
            with torch.profiler.record_function("sample"):
                next_tok = engine.model.sample_next(self.out.logits[:, -1, :], temperature=self.temperature, top_p=self.top_p)
            for req_id in set(self.active):
                tok = next_tok[req_id].item()
                if tok in self.stop_ids:
                    completions_by_id[req_id] = self.complete(req_id)
                    continue
                completion = self.update(req_id, tok)
                if completion is not None:
                    completions_by_id[req_id] = completion

            self.attn_mask = torch.cat([self.attn_mask, self.attn_mask.new_ones(self.attn_mask.shape[0], 1)], dim=1)
            position_ids = (self.prompt_lengths + self.step_idx).unsqueeze(1)
            cache_position = torch.tensor([self.padded_lengths + self.step_idx], device=self.model.device)
            with torch.profiler.record_function("forward"):
                self.out = self.model(
                        next_tok,
                        attention_mask=self.attn_mask,
                        past_key_values=self.out.past_key_values,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        use_cache=True
                        )
            self.step_idx += 1
        return completions_by_id





