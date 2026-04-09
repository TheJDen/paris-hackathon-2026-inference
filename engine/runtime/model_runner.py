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
    pass


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

    @torch.inference_mode()
    def prefill(self, slot_id: int, prompt_token_ids: list[int]) -> int:
        """Prefill one slot with its prompt and return the first sampled token.

        Phase 2a v0: prefill one slot at a time. Phase 2b will batch
        prefills with chunking.
        """
        with time_region("runner.prefill"):
            L = len(prompt_token_ids)
            input_ids = torch.tensor(
                [prompt_token_ids], dtype=torch.long, device=self.device
            )

            # Build batch routing: one row, write positions 0..L-1.
            slot_ids = torch.tensor([slot_id], dtype=torch.int64, device=self.device)
            write_positions = [
                torch.arange(L, dtype=torch.int64, device=self.device)
            ]
            query_lens = torch.tensor([L], dtype=torch.int64, device=self.device)
            kv_seq_lens = torch.tensor([L], dtype=torch.int64, device=self.device)
            self.cache.set_batch(
                BatchSlots(
                    slot_ids=slot_ids,
                    write_positions=write_positions,
                    query_lens=query_lens,
                    kv_seq_lens=kv_seq_lens,
                    is_prefill=True,
                )
            )

            position_ids = torch.arange(L, dtype=torch.long, device=self.device).unsqueeze(0)
            attention_mask = torch.ones(1, L, dtype=torch.long, device=self.device)

            with time_region("runner.prefill.model_forward"):
                outputs = self.inner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            self.cache.commit_batch()  # scatter linear-attn views back to pool

            hidden_states = outputs.last_hidden_state  # [1, L, hidden]
            # We only need the last position's logits to sample the first output token.
            with time_region("runner.prefill.lm_head"):
                last_hidden = hidden_states[:, -1:, :]
                logits = self.lm_head(last_hidden)  # [1, 1, vocab]

            with time_region("runner.prefill.sample"):
                next_token_id = int(self._sample_greedy(logits[:, -1, :]).item())

            return next_token_id

    @torch.inference_mode()
    def decode(
        self,
        slot_ids: list[int],
        last_tokens: list[int],
        cache_lengths: list[int],
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
            # attention_mask: padded to max kv_seq_len, mask out empty positions
            max_s = int(kv_seq_lens.max().item())
            attention_mask = torch.zeros(B, max_s, dtype=torch.long, device=self.device)
            for b in range(B):
                attention_mask[b, : cache_lengths[b] + 1] = 1

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
                next_tokens = self._sample_greedy(logits)  # [B]
            return [int(t.item()) for t in next_tokens]

    # ------------------------------------------------------------------ #
    # sampling — greedy only for v0; temp/top_p come next
    # ------------------------------------------------------------------ #

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """Argmax over the vocab dimension. Returns a 1-D int64 tensor."""
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #

    def cache_memory_gb(self) -> float:
        return self.cache.memory_bytes() / 1024**3
