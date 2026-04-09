"""Direct model forward + sampler.

The point of this file is to call HF's `Qwen3_5MoeTextModel.forward(...)`
without going through `model.generate()`. We give it our slot-pool cache,
we give it our packed `input_ids` + `position_ids` + `attention_mask`,
we run `model.lm_head` ourselves, and we sample on the GPU ourselves.

`transformers` is purely a `nn.Module` library here. No HF generate. No
HF cache (we provide our own).

The runner exposes two operations:

  * `prefill(slot_id, prompt_token_ids)` → first generated token id
  * `decode(slot_ids, last_tokens)`      → list of next-token ids per slot

These are what the scheduler calls. The runner does not know anything
about asyncio / requests / chat templates — that's the engine's job.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from engine.runtime.kv_cache import BatchSlots, SlotPoolCache
from engine.runtime.profiling import time_region

if TYPE_CHECKING:
    from engine.runtime.sequence import SamplingParams


log = logging.getLogger(__name__)


class ModelRunner:
    """Bridge between the scheduler and the HF model layers.

    Owns the loaded HF model, the slot-pool cache, and the sampling state.
    """

    def __init__(
        self,
        hf_model,
        text_config,
        device: torch.device,
        *,
        num_slots: int,
        max_seq_len: int,
    ) -> None:
        self.hf_model = hf_model
        self.text_config = text_config
        self.device = device
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len

        # The inner Qwen3_5MoeTextModel where layers/embed/norm live.
        self.inner_model = getattr(hf_model, "model", hf_model)
        self.lm_head = getattr(hf_model, "lm_head", None)
        if self.lm_head is None:
            raise RuntimeError("hf_model has no lm_head — wrong wrapper?")

        # Build the slot-pool cache from text_config dimensions.
        self.cache = self._build_cache()
        log.info("model runner cache: %s", self.cache)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    def _build_cache(self) -> SlotPoolCache:
        cfg = self.text_config
        layer_types = list(cfg.layer_types)
        # KV cache dtype follows the model's dtype (BF16 here)
        kv_dtype = next(self.hf_model.parameters()).dtype
        # Recurrent / conv state stays in fp32 per text_config.mamba_ssm_dtype.
        # The linear-attention pool tensors lazy-init from the first incoming
        # tensor's dtype anyway, so this is just a default for the
        # initial allocation if anyone asks.
        state_dtype_str = getattr(cfg, "mamba_ssm_dtype", "float32")
        state_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(state_dtype_str, torch.float32)

        return SlotPoolCache(
            num_slots=self.num_slots,
            max_seq_len=self.max_seq_len,
            layer_types=layer_types,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            linear_num_k_heads=cfg.linear_num_key_heads,
            linear_num_v_heads=cfg.linear_num_value_heads,
            linear_head_k_dim=cfg.linear_key_head_dim,
            linear_head_v_dim=cfg.linear_value_head_dim,
            linear_conv_kernel=cfg.linear_conv_kernel_dim,
            kv_dtype=kv_dtype,
            state_dtype=state_dtype,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def prefill(self, slot_id: int, prompt_token_ids: list[int], sampling: "SamplingParams") -> int:
        """Prefill one slot. Thin wrapper around prefill_batch."""
        return self.prefill_batch([slot_id], [prompt_token_ids], [sampling])[0]

    @torch.inference_mode()
    def prefill_batch(
        self,
        slot_ids: list[int],
        prompt_token_ids_list: list[list[int]],
        samplings: "list[SamplingParams]",
    ) -> list[int]:
        """Prefill B slots in one forward pass. Returns first sampled token per slot.

        Sequences are right-padded to the longest prompt in the batch so all
        rows share the same input length. Only real-token K/V is written to
        the cache (padding positions are sliced away in cache.update()).

        Note: DeltaNet recurrent state is computed over the padded sequence,
        meaning pad tokens after the last real token contribute to the final
        state. For the throughput eval all prompts have the same length so
        there is no padding and this is a non-issue.
        """
        with time_region("runner.prefill"):
            B = len(slot_ids)
            prompt_lens = [len(p) for p in prompt_token_ids_list]
            max_len = max(prompt_lens)

            # Right-pad with zeros. Padding token id does not matter because
            # the attention mask zeros out those positions.
            padded = [p + [0] * (max_len - len(p)) for p in prompt_token_ids_list]
            input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)  # [B, max_len]

            # 1 for real tokens, 0 for right-side padding.
            attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
            for b in range(B):
                attention_mask[b, : prompt_lens[b]] = 1

            # Shared position ids 0..max_len-1; padding positions get arbitrary
            # ids (masked by attention_mask, output is discarded).
            position_ids = (
                torch.arange(max_len, dtype=torch.long, device=self.device)
                .unsqueeze(0)
                .expand(B, -1)
            )

            slot_ids_t = torch.tensor(slot_ids, dtype=torch.int64, device=self.device)
            write_positions = [
                torch.arange(prompt_lens[b], dtype=torch.int64, device=self.device)
                for b in range(B)
            ]
            query_lens = torch.tensor(prompt_lens, dtype=torch.int64, device=self.device)
            kv_seq_lens = torch.tensor(prompt_lens, dtype=torch.int64, device=self.device)
            self.cache.set_batch(
                BatchSlots(
                    slot_ids=slot_ids_t,
                    write_positions=write_positions,
                    query_lens=query_lens,
                    kv_seq_lens=kv_seq_lens,
                    is_prefill=True,
                )
            )

            with time_region("runner.prefill.model_forward"):
                outputs = self.inner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            self.cache.commit_batch()

            hidden_states = outputs.last_hidden_state  # [B, max_len, hidden]

            with time_region("runner.prefill.lm_head"):
                # Gather the hidden state at the last REAL token of each row.
                last_pos = torch.tensor(
                    [l - 1 for l in prompt_lens], dtype=torch.long, device=self.device
                )
                last_hidden = hidden_states[
                    torch.arange(B, device=self.device), last_pos
                ]  # [B, hidden]
                logits = self.lm_head(last_hidden)  # [B, vocab]

            with time_region("runner.prefill.sample"):
                next_tokens = self._sample(logits, samplings)  # [B]

            return [int(t.item()) for t in next_tokens]

    @torch.inference_mode()
    def decode(
        self,
        slot_ids: list[int],
        last_tokens: list[int],
        cache_lengths: list[int],
        samplings: "list[SamplingParams]",
    ) -> list[int]:
        """Run one decode step for B slots, return next-token id per slot.

        `cache_lengths[i]` is the slot's CURRENT cached length BEFORE this
        step (i.e. the position the new token lands at).
        """
        with time_region("runner.decode"):
            B = len(slot_ids)
            assert B == len(last_tokens) == len(cache_lengths)

            input_ids = torch.tensor(
                [[t] for t in last_tokens], dtype=torch.long, device=self.device
            )  # [B, 1]
            slot_ids_t = torch.tensor(slot_ids, dtype=torch.int64, device=self.device)
            write_positions = [
                torch.tensor([cache_lengths[i]], dtype=torch.int64, device=self.device)
                for i in range(B)
            ]
            query_lens = torch.ones(B, dtype=torch.int64, device=self.device)
            kv_seq_lens = torch.tensor(
                [cache_lengths[i] + 1 for i in range(B)],
                dtype=torch.int64, device=self.device,
            )
            self.cache.set_batch(
                BatchSlots(
                    slot_ids=slot_ids_t,
                    write_positions=write_positions,
                    query_lens=query_lens,
                    kv_seq_lens=kv_seq_lens,
                    is_prefill=False,
                )
            )

            # position_ids[b] = cache_lengths[b] (the absolute position of the new token)
            position_ids = torch.tensor(
                [[cache_lengths[i]] for i in range(B)],
                dtype=torch.long, device=self.device,
            )
            # attention_mask: 1 where position < kv_seq_len, 0 elsewhere.
            # Vectorized: broadcast [1, max_s] < [B, 1] → [B, max_s], no loop.
            max_s = int(kv_seq_lens.max().item())
            positions = torch.arange(max_s, dtype=torch.long, device=self.device).unsqueeze(0)
            attention_mask = (positions < kv_seq_lens.unsqueeze(1)).long()

            with time_region("runner.decode.model_forward"):
                outputs = self.inner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            self.cache.commit_batch()  # scatter linear-attn views back to pool

            hidden_states = outputs.last_hidden_state  # [B, 1, hidden]
            with time_region("runner.decode.lm_head"):
                logits = self.lm_head(hidden_states[:, -1, :])  # [B, vocab]

            with time_region("runner.decode.sample"):
                next_tokens = self._sample(logits, samplings)  # [B]
            return [int(t.item()) for t in next_tokens]

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #

    def _sample(self, logits: torch.Tensor, samplings: "list[SamplingParams]") -> torch.Tensor:
        """Per-row temperature + top-p sampling. Returns a 1-D int64 tensor [B].

        Rows with temperature <= 0 get greedy (argmax). Mixed greedy/sampled
        batches are handled: greedy rows are argmax'd at the end, overriding
        the multinomial result.
        """
        B = logits.shape[0]

        # Fast path: entire batch is greedy.
        if all(s.greedy for s in samplings):
            return torch.argmax(logits, dim=-1)

        temperatures = [s.temperature for s in samplings]
        top_ps = [s.top_p for s in samplings]
        greedy_mask = torch.tensor(
            [s.greedy for s in samplings], dtype=torch.bool, device=logits.device
        )

        # Scale logits by temperature; use 1.0 for greedy rows to avoid div-by-zero.
        temps = torch.tensor(
            [1.0 if s.greedy else s.temperature for s in samplings],
            dtype=logits.dtype, device=logits.device,
        ).unsqueeze(1)  # [B, 1]
        scaled = logits / temps  # [B, vocab]

        probs = torch.softmax(scaled, dim=-1)  # [B, vocab]

        # Top-p (nucleus) filtering — vectorised over the batch.
        if any(p < 1.0 for p in top_ps):
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens where all prior tokens already cover top_p.
            top_p_t = torch.tensor(top_ps, dtype=probs.dtype, device=logits.device).unsqueeze(1)
            to_remove = (cumulative - sorted_probs) > top_p_t
            sorted_probs = sorted_probs.masked_fill(to_remove, 0.0)
            # Scatter filtered probs back to vocabulary order.
            probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)

        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

        # Override greedy rows with argmax.
        if greedy_mask.any():
            greedy_tokens = torch.argmax(logits, dim=-1)
            sampled = torch.where(greedy_mask, greedy_tokens, sampled)

        return sampled

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #

    def cache_memory_gb(self) -> float:
        return self.cache.memory_bytes() / 1024**3
