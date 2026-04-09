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
from engine.runtime.sequence import SamplingParams

if TYPE_CHECKING:
    from engine.runtime.cuda_graphs import DecodeGraphCache
    from engine.runtime.compile_helpers import warmup_compiled


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
        enable_cuda_graphs: bool = True,
        compile: bool = True,
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

        # torch.compile — wrap inner_model with Inductor.
        # Must happen BEFORE CUDA graph init (reduce-overhead mode uses
        # CUDA graphs internally; if both compile AND DecodeGraphCache are
        # active they will conflict).  When compile=True we therefore force
        # enable_cuda_graphs=False.
        self._compile_enabled = compile and device.type == "cuda"
        if self._compile_enabled:
            if enable_cuda_graphs:
                log.info(
                    "model runner: torch.compile=True overrides enable_cuda_graphs "
                    "(reduce-overhead mode owns CUDA graphs; DecodeGraphCache disabled)"
                )
            enable_cuda_graphs = False  # prevent double-graph conflict
            self._apply_compile()

        # CUDA graph cache for the decode hot path (lazy init on first use).
        # Set to None when CUDA graphs are disabled (CPU device, stub mode,
        # --no-cuda-graphs flag, or torch.compile=True above).
        self.decode_graphs: DecodeGraphCache | None = None
        if enable_cuda_graphs and device.type == "cuda":
            self._init_decode_graphs()

        # Dedicated CUDA RNG generator for the sampler.  Using a separate
        # generator prevents the default CUDA generator from being put into
        # "graph mode" by CUDA graph capture (which saves/restores the default
        # generator's RNG state and registers it for capture, causing
        # torch.multinomial to raise "Offset increment outside graph capture"
        # during prefill — before any graph has been replayed).
        self._sampler_generator: torch.Generator | None = None
        if device.type == "cuda":
            self._sampler_generator = torch.Generator(device=device)
            self._sampler_generator.manual_seed(0)

        # Warmup: fire the compile before the first real request.
        if self._compile_enabled:
            self._warmup_compile()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    def _apply_compile(self) -> None:
        """Apply dynamo disables + torch.compile to self.inner_model."""
        from engine.runtime.compile_helpers import apply_dynamo_disables, compile_model

        log.info("model runner: applying dynamo disables (cache + linear-attn)")
        apply_dynamo_disables(self.hf_model)

        log.info("model runner: wrapping inner_model with torch.compile")
        self.inner_model = compile_model(
            self.hf_model,
            mode="reduce-overhead",
            dynamic=True,
        )
        log.info("model runner: torch.compile wrapper installed (lazy — fires on first forward)")

    def _warmup_compile(self) -> None:
        """Fire the compile with dummy tensors before the first real request."""
        from engine.runtime.compile_helpers import warmup_compiled

        warmup_compiled(
            self,
            max_batch=min(8, self.num_slots),
            max_seq=min(2048, self.max_seq_len),
        )

    def _init_decode_graphs(self) -> None:
        """Instantiate the DecodeGraphCache (lazy; graphs captured on demand)."""
        from engine.runtime.cuda_graphs import DecodeGraphCache

        # Derive vocab_size from lm_head.
        vocab_size: int
        if hasattr(self.lm_head, "out_features"):
            vocab_size = self.lm_head.out_features
        elif hasattr(self.lm_head, "weight"):
            vocab_size = self.lm_head.weight.shape[0]
        else:
            log.warning(
                "Could not determine vocab_size from lm_head; "
                "CUDA graphs disabled."
            )
            return

        self.decode_graphs = DecodeGraphCache(
            inner_model=self.inner_model,
            lm_head=self.lm_head,
            cache=self.cache,
            max_seq_len=self.max_seq_len,
            device=self.device,
            vocab_size=vocab_size,
        )
        log.info("DecodeGraphCache initialized (lazy capture): %s", self.decode_graphs)

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
    def prefill(
        self,
        slot_id: int,
        prompt_token_ids: list[int],
        sampling: SamplingParams | None = None,
        prefix_offset: int = 0,
    ) -> int:
        """Prefill one slot with its prompt and return the first sampled token.

        Phase 2a v0: prefill one slot at a time. Phase 2b will batch
        prefills with chunking.

        Args:
            prefix_offset: if > 0, the slot already has ``prefix_offset`` tokens
                           cached (copied from the shared-prefix reserved slot).
                           ``prompt_token_ids`` should contain ONLY the suffix
                           tokens (prompt[prefix_offset:]).  Write positions and
                           position_ids are shifted by ``prefix_offset``
                           accordingly.  ``kv_seq_lens`` is set to
                           ``prefix_offset + L`` so attention sees the full
                           context.
        """
        with time_region("runner.prefill"):
            L = len(prompt_token_ids)
            input_ids = torch.tensor(
                [prompt_token_ids], dtype=torch.long, device=self.device
            )

            # Build batch routing: one row.
            # When prefix_offset > 0 the slot already has the first
            # prefix_offset positions cached; we write suffix tokens at
            # positions [prefix_offset, prefix_offset + L).
            slot_ids = torch.tensor([slot_id], dtype=torch.int64, device=self.device)
            write_positions = [
                torch.arange(prefix_offset, prefix_offset + L, dtype=torch.int64, device=self.device)
            ]
            query_lens = torch.tensor([L], dtype=torch.int64, device=self.device)
            kv_seq_lens = torch.tensor([prefix_offset + L], dtype=torch.int64, device=self.device)
            self.cache.set_batch(
                BatchSlots(
                    slot_ids=slot_ids,
                    write_positions=write_positions,
                    query_lens=query_lens,
                    kv_seq_lens=kv_seq_lens,
                    is_prefill=True,
                )
            )

            position_ids = torch.arange(
                prefix_offset, prefix_offset + L, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            # Attention mask must cover the full context (prefix + suffix).
            # Build a 1 × (prefix_offset + L) mask of ones.
            full_len = prefix_offset + L
            attention_mask = torch.ones(1, full_len, dtype=torch.long, device=self.device)

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
                temps = torch.tensor(
                    [sampling.temperature if sampling is not None else 0.0],
                    dtype=torch.float32, device=self.device,
                )
                top_ps = torch.tensor(
                    [sampling.top_p if sampling is not None else 1.0],
                    dtype=torch.float32, device=self.device,
                )
                next_token_id = int(self._sample(logits[:, -1, :], temps, top_ps).item())

            return next_token_id

    @torch.inference_mode()
    def prefill_batch(
        self,
        slot_ids: list[int],
        prompt_token_ids_list: list[list[int]],
        samplings: list,
        prefix_offsets: list[int] | None = None,
    ) -> list[int]:
        """Prefill B slots in one padded forward pass and return first tokens.

        v1 implementation: single forward pass over all B prompts at once.
        Right-pad each prompt to max_L using eos_token_id (or 0 as fallback).
        The attention_mask gates pad positions so they don't pollute attention
        for real tokens.  The cache update() slices each row to its real
        length so no garbage K/V from pad positions is written to the slot pool.

        Args:
            prefix_offsets: optional list of per-row prefix offsets.  When
                ``prefix_offsets[b] > 0``, row b's slot already has
                ``prefix_offsets[b]`` tokens cached (copied from the shared
                prefix reserved slot).  ``prompt_token_ids_list[b]`` must
                contain ONLY the suffix tokens (original_prompt[offset:]).
                Write positions and position_ids for that row are shifted by
                the offset; kv_seq_lens accounts for the full context length.
                If all offsets are 0 (or prefix_offsets is None), behavior is
                identical to the original implementation.

        Returns a list of B first-token ids, one per sequence.
        """
        with time_region("runner.prefill_batch"):
            B = len(slot_ids)
            if prefix_offsets is None:
                prefix_offsets = [0] * B

            # Fast path: single sequence — delegate to the single-slot path
            # to avoid any overhead from the batched code path.
            if B == 1:
                return [self.prefill(
                    slot_ids[0],
                    prompt_token_ids_list[0],
                    samplings[0],
                    prefix_offset=prefix_offsets[0],
                )]

            # ``real_lens[b]`` = number of suffix tokens for row b.
            # ``full_lens[b]`` = prefix_offset[b] + real_lens[b] (total context).
            real_lens = [len(p) for p in prompt_token_ids_list]
            offsets = prefix_offsets  # alias for clarity
            full_lens = [offsets[b] + real_lens[b] for b in range(B)]
            max_L = max(real_lens)          # maximum suffix length (for input tensor)
            max_full = max(full_lens)       # maximum total context (for attention mask)

            # Pick a pad token id.
            pad_token_id = 0
            config = getattr(self.inner_model, "config", None)
            if config is not None:
                eos = getattr(config, "eos_token_id", None)
                if eos is not None:
                    pad_token_id = eos if isinstance(eos, int) else eos[0]

            # Build [B, max_L] input_ids with right-padding (suffix tokens only).
            input_ids_np = torch.full(
                (B, max_L), fill_value=pad_token_id, dtype=torch.long, device=self.device
            )
            for b, prompt in enumerate(prompt_token_ids_list):
                input_ids_np[b, : real_lens[b]] = torch.tensor(
                    prompt, dtype=torch.long, device=self.device
                )
            input_ids = input_ids_np  # [B, max_L]

            # attention_mask: [B, max_full].
            # For prefix-cached rows, positions 0..offset-1 are already in the
            # cache and must appear as attended tokens in the mask; real suffix
            # positions offset..full_len-1 are also 1; pad is 0.
            attention_mask = torch.zeros(B, max_full, dtype=torch.long, device=self.device)
            for b in range(B):
                attention_mask[b, : full_lens[b]] = 1

            # position_ids: [B, max_L] — suffix positions only (the model sees
            # only the suffix input_ids, but at the right absolute positions).
            # Row b: positions offsets[b], offsets[b]+1, ..., offsets[b]+real_lens[b]-1
            # Pad positions get a value == full_lens[b] (out of range, gated by mask).
            position_ids = torch.zeros(B, max_L, dtype=torch.long, device=self.device)
            for b in range(B):
                position_ids[b, : real_lens[b]] = torch.arange(
                    offsets[b], full_lens[b], dtype=torch.long, device=self.device
                )
                if real_lens[b] < max_L:
                    position_ids[b, real_lens[b] :] = full_lens[b]

            # BatchSlots: per-row write_positions are the suffix positions in
            # the slot pool (offset..full_len-1), not 0..real_len-1.
            slot_ids_t = torch.tensor(slot_ids, dtype=torch.int64, device=self.device)
            write_positions = [
                torch.arange(offsets[b], full_lens[b], dtype=torch.int64, device=self.device)
                for b in range(B)
            ]
            query_lens = torch.tensor(real_lens, dtype=torch.int64, device=self.device)
            kv_seq_lens = torch.tensor(full_lens, dtype=torch.int64, device=self.device)
            self.cache.set_batch(
                BatchSlots(
                    slot_ids=slot_ids_t,
                    write_positions=write_positions,
                    query_lens=query_lens,
                    kv_seq_lens=kv_seq_lens,
                    is_prefill=True,
                )
            )

            with time_region("runner.prefill_batch.model_forward"):
                outputs = self.inner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            self.cache.commit_batch()

            # hidden_states: [B, max_L, hidden].
            # Extract last-real-position hidden state per row (index real_lens[b]-1).
            hidden_states = outputs.last_hidden_state  # [B, max_L, hidden]
            hidden_dim = hidden_states.shape[-1]
            last_real_idx = torch.tensor(
                [l - 1 for l in real_lens], dtype=torch.long, device=self.device
            )  # [B]
            # Gather: [B, 1, hidden] → [B, hidden]
            last_hidden = hidden_states.gather(
                dim=1,
                index=last_real_idx.view(B, 1, 1).expand(B, 1, hidden_dim),
            ).squeeze(1)  # [B, hidden]

            with time_region("runner.prefill_batch.lm_head"):
                logits = self.lm_head(last_hidden)  # [B, vocab]

            with time_region("runner.prefill_batch.sample"):
                if samplings is not None and len(samplings) == B:
                    temps = torch.tensor(
                        [s.temperature if s is not None else 0.0 for s in samplings],
                        dtype=torch.float32, device=self.device,
                    )
                    top_ps = torch.tensor(
                        [s.top_p if s is not None else 1.0 for s in samplings],
                        dtype=torch.float32, device=self.device,
                    )
                else:
                    temps = torch.zeros(B, dtype=torch.float32, device=self.device)
                    top_ps = torch.ones(B, dtype=torch.float32, device=self.device)

                next_tokens = self._sample(logits, temps, top_ps)  # [B]

            return [int(t.item()) for t in next_tokens]

    @torch.inference_mode()
    def mixed_step(
        self,
        decode_slot_ids: list[int],
        decode_last_tokens: list[int],
        decode_cache_lengths: list[int],
        decode_samplings: list[SamplingParams] | None,
        prefill_slot_ids: list[int],
        prefill_token_ids_list: list[list[int]],
        prefill_samplings: list,
        prefill_offsets: list[int] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Chunked-prefill-style mixed step: decode rows + prefill rows in ONE forward.

        Decode rows contribute query_len=1 (the just-sampled token).
        Prefill rows contribute query_len=L_r (their full suffix).

        Layout of the combined batch:
            rows [0..D)        = decode rows (query_len=1)
            rows [D..D+P)      = prefill rows (query_len=L_r)

        Ragged kv handling: each row r has an effective kv length:
            decode row r:  cache_lengths[r] + 1
            prefill row r: offsets[r] + L_r
        We pass attention_mask of shape [B, max_full] with 1s on
        [0..effective_kv_len_r) — exactly as prefill_batch already does.
        The kv_cache.update() slices each row to its real write positions so
        no pad K/V pollutes the slot pool.

        Returns:
            (decode_next_tokens, prefill_first_tokens)
        """
        D = len(decode_slot_ids)
        P = len(prefill_slot_ids)
        assert D == len(decode_last_tokens) == len(decode_cache_lengths)
        assert P == len(prefill_token_ids_list) == len(prefill_samplings)

        # Trivial fallbacks so warmup / single-class callers stay cheap.
        if D == 0 and P == 0:
            return [], []
        if P == 0:
            toks = self.decode(
                decode_slot_ids, decode_last_tokens, decode_cache_lengths, decode_samplings
            )
            return toks, []
        if D == 0:
            toks = self.prefill_batch(
                prefill_slot_ids,
                prefill_token_ids_list,
                prefill_samplings,
                prefix_offsets=prefill_offsets,
            )
            return [], toks

        with time_region("runner.mixed_step"):
            if prefill_offsets is None:
                prefill_offsets = [0] * P

            # Per-row real query lens (how many new token positions to write).
            real_lens: list[int] = [1] * D + [len(p) for p in prefill_token_ids_list]
            # Full kv length per row (for attention mask).
            full_lens: list[int] = (
                [decode_cache_lengths[i] + 1 for i in range(D)]
                + [prefill_offsets[i] + len(prefill_token_ids_list[i]) for i in range(P)]
            )
            # Per-row starting write position (absolute slot position where the
            # new tokens land).
            start_pos: list[int] = (
                [decode_cache_lengths[i] for i in range(D)]
                + [prefill_offsets[i] for i in range(P)]
            )

            B = D + P
            max_L = max(real_lens)
            max_full = max(full_lens)

            # Pad token.
            pad_token_id = 0
            config = getattr(self.inner_model, "config", None)
            if config is not None:
                eos = getattr(config, "eos_token_id", None)
                if eos is not None:
                    pad_token_id = eos if isinstance(eos, int) else eos[0]

            # input_ids: [B, max_L], right-padded.
            input_ids = torch.full(
                (B, max_L), fill_value=pad_token_id, dtype=torch.long, device=self.device
            )
            for i in range(D):
                input_ids[i, 0] = decode_last_tokens[i]
            for j, prompt in enumerate(prefill_token_ids_list):
                input_ids[D + j, : len(prompt)] = torch.tensor(
                    prompt, dtype=torch.long, device=self.device
                )

            # attention_mask: [B, max_full] — 1s for real kv positions (cache
            # already contains [0..start_pos[r]) and this step writes
            # [start_pos[r]..full_lens[r])).
            attention_mask = torch.zeros(B, max_full, dtype=torch.long, device=self.device)
            for r in range(B):
                attention_mask[r, : full_lens[r]] = 1

            # position_ids: [B, max_L]. Real positions are arange(start, start+L_r);
            # pad positions get full_lens[r] (out of range, gated by mask).
            position_ids = torch.zeros(B, max_L, dtype=torch.long, device=self.device)
            for r in range(B):
                L_r = real_lens[r]
                position_ids[r, :L_r] = torch.arange(
                    start_pos[r], start_pos[r] + L_r, dtype=torch.long, device=self.device
                )
                if L_r < max_L:
                    position_ids[r, L_r:] = full_lens[r]

            # BatchSlots: per-row write positions, query_lens, kv_seq_lens.
            all_slot_ids = list(decode_slot_ids) + list(prefill_slot_ids)
            slot_ids_t = torch.tensor(all_slot_ids, dtype=torch.int64, device=self.device)
            write_positions = [
                torch.arange(
                    start_pos[r], start_pos[r] + real_lens[r],
                    dtype=torch.int64, device=self.device,
                )
                for r in range(B)
            ]
            query_lens = torch.tensor(real_lens, dtype=torch.int64, device=self.device)
            kv_seq_lens = torch.tensor(full_lens, dtype=torch.int64, device=self.device)
            # Mixed batch: mark is_prefill=True so the cache path takes the
            # ragged/variable-length write branch (same path prefill_batch uses).
            self.cache.set_batch(
                BatchSlots(
                    slot_ids=slot_ids_t,
                    write_positions=write_positions,
                    query_lens=query_lens,
                    kv_seq_lens=kv_seq_lens,
                    is_prefill=True,
                )
            )

            with time_region("runner.mixed_step.model_forward"):
                outputs = self.inner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            self.cache.commit_batch()

            hidden_states = outputs.last_hidden_state  # [B, max_L, hidden]
            hidden_dim = hidden_states.shape[-1]
            # Sample the last-real-position per row. For decode rows that's
            # index 0; for prefill rows that's index L_r-1.
            last_real_idx = torch.tensor(
                [real_lens[r] - 1 for r in range(B)], dtype=torch.long, device=self.device
            )
            last_hidden = hidden_states.gather(
                dim=1,
                index=last_real_idx.view(B, 1, 1).expand(B, 1, hidden_dim),
            ).squeeze(1)  # [B, hidden]

            with time_region("runner.mixed_step.lm_head"):
                logits = self.lm_head(last_hidden)  # [B, vocab]

            # Build per-row temps/top_ps for sampling.
            temps_list: list[float] = []
            top_ps_list: list[float] = []
            if decode_samplings is not None:
                for s in decode_samplings:
                    temps_list.append(s.temperature if s is not None else 0.0)
                    top_ps_list.append(s.top_p if s is not None else 1.0)
            else:
                temps_list.extend([0.0] * D)
                top_ps_list.extend([1.0] * D)
            for s in prefill_samplings:
                temps_list.append(s.temperature if s is not None else 0.0)
                top_ps_list.append(s.top_p if s is not None else 1.0)

            temps = torch.tensor(temps_list, dtype=torch.float32, device=self.device)
            top_ps = torch.tensor(top_ps_list, dtype=torch.float32, device=self.device)

            with time_region("runner.mixed_step.sample"):
                next_tokens = self._sample(logits, temps, top_ps)  # [B]

            result = [int(t.item()) for t in next_tokens]
            return result[:D], result[D:]

    @torch.inference_mode()
    def decode(
        self,
        slot_ids: list[int],
        last_tokens: list[int],
        cache_lengths: list[int],
        samplings: list[SamplingParams] | None = None,
    ) -> list[int]:
        """Run one decode step for B slots, return next-token id per slot.

        `cache_lengths[i]` is the slot's CURRENT cached length BEFORE this
        step (i.e. the position the new token lands at).
        `samplings[i]` holds temperature/top_p for the i-th row.

        When CUDA graphs are enabled (self.decode_graphs is not None), the
        model forward + lm_head are replayed via a pre-captured CUDA graph
        for the bucket that covers B.  The sampler always runs eagerly
        outside the graph.  The eager fallback path is preserved for
        correctness testing and for the first few steps inside stub mode.
        """
        with time_region("runner.decode"):
            B = len(slot_ids)
            assert B == len(last_tokens) == len(cache_lengths)

            # ---- Build per-step tensors (always eager, always outside graph) ----
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

            # position_ids[b] = cache_lengths[b] (the absolute position of the new token)
            position_ids = torch.tensor(
                [[cache_lengths[i]] for i in range(B)],
                dtype=torch.long, device=self.device,
            )
            # attention_mask: padded to max kv_seq_len, mask out empty positions.
            # For the graph path, this is copied into the static buffer (which is
            # sized to max_seq_len); for the eager path it's used directly.
            max_s = int(kv_seq_lens.max().item())
            attention_mask = torch.zeros(B, max_s, dtype=torch.long, device=self.device)
            for b in range(B):
                attention_mask[b, : cache_lengths[b] + 1] = 1

            # ---- Cache set_batch is ALWAYS outside the graph ----
            # gather_for_batch (index_select + clone) is not graph-safe because
            # it allocates new tensors with data-dependent indices each call.
            self.cache.set_batch(
                BatchSlots(
                    slot_ids=slot_ids_t,
                    write_positions=write_positions,
                    query_lens=query_lens,
                    kv_seq_lens=kv_seq_lens,
                    is_prefill=False,
                )
            )

            # ---- Model forward + lm_head ----
            if self.decode_graphs is not None:
                # Graph path: copies per-step values into static buffers, replays.
                # Returns logits [B, vocab] for the real rows only.
                with time_region("runner.decode.model_forward"):
                    logits = self.decode_graphs.replay(
                        batch_size=B,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )
                # commit_batch is called inside replay() (after graph replay).
                # Logits are already [B, vocab] — lm_head is inside the graph.
            else:
                # Eager fallback path (also used when CUDA graphs are disabled).
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

            # ---- Sampler — always eager, outside the graph ----
            with time_region("runner.decode.sample"):
                if samplings is not None:
                    temps = torch.tensor(
                        [s.temperature for s in samplings],
                        dtype=torch.float32, device=self.device,
                    )
                    top_ps = torch.tensor(
                        [s.top_p for s in samplings],
                        dtype=torch.float32, device=self.device,
                    )
                else:
                    temps = torch.zeros(B, dtype=torch.float32, device=self.device)
                    top_ps = torch.ones(B, dtype=torch.float32, device=self.device)
                next_tokens = self._sample(logits, temps, top_ps)  # [B]
            return [int(t.item()) for t in next_tokens]

    # ------------------------------------------------------------------ #
    # sampling — greedy fast-path + temperature / top-p nucleus sampling
    # ------------------------------------------------------------------ #

    def _sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row sampler supporting greedy argmax and temperature+top-p.

        Args:
            logits:       [B, vocab_size]  — raw logits from lm_head (fp32 or bf16)
            temperatures: [B]              — 0.0 means greedy for that row
            top_ps:       [B]              — nucleus cutoff; >=1.0 means no masking

        Returns:
            [B] int64 tensor of sampled token ids.

        top_k: SamplingParams does not currently have a top_k field, so top_k
        is not implemented. Add SamplingParams.top_k and a masking step here
        when needed.
        """
        # Fast-path: if every row is greedy, skip all stochastic logic.
        if bool((temperatures <= 0.0).all()):
            return torch.argmax(logits, dim=-1)

        B, vocab = logits.shape
        # Work in float32 for numerical stability.
        logits_f = logits.float()  # [B, vocab]

        # --- per-row dispatch ---
        # Rows where temperature==0 → argmax; others → stochastic.
        greedy_mask = temperatures <= 0.0  # [B] bool

        # --- stochastic rows: apply temperature and top-p ---
        # Scale logits by temperature (broadcast over vocab).
        # temperatures shape: [B] → [B, 1] for broadcasting.
        temps_safe = temperatures.clamp(min=1e-6).unsqueeze(1)  # [B, 1]
        scaled = logits_f / temps_safe  # [B, vocab]

        # Top-p (nucleus) filtering.
        # Sort descending so the highest-prob tokens come first.
        sorted_logits, sorted_indices = torch.sort(scaled, dim=-1, descending=True)
        cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)  # [B, vocab]

        # Shift cumprobs right by one so that the token that pushes cumsum
        # over top_p is still included (standard nucleus formulation).
        cumprobs_shifted = torch.roll(cumprobs, shifts=1, dims=-1)
        cumprobs_shifted[:, 0] = 0.0

        # Build a mask: True for tokens to REMOVE (cumprob already exceeded top_p).
        # top_ps shape: [B] → [B, 1] for broadcasting.
        remove_mask = cumprobs_shifted > top_ps.unsqueeze(1)  # [B, vocab]

        # Rows where top_p >= 1.0: disable masking (no filtering).
        no_topp_mask = (top_ps >= 1.0).unsqueeze(1)  # [B, 1]
        remove_mask = remove_mask & ~no_topp_mask

        # Apply mask (fill removed positions with -inf in sorted order).
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

        # Convert back to the original vocab ordering.
        filtered_logits = torch.zeros_like(sorted_logits)
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        # Sample from the filtered distribution.
        probs = torch.softmax(filtered_logits, dim=-1)  # [B, vocab]
        stochastic_tokens = torch.multinomial(
            probs, num_samples=1, generator=self._sampler_generator
        ).squeeze(1)  # [B]

        # Greedy tokens (argmax on the original logits, not scaled).
        greedy_tokens = torch.argmax(logits_f, dim=-1)  # [B]

        # Merge: use greedy where temperature==0, stochastic elsewhere.
        result = torch.where(greedy_mask, greedy_tokens, stochastic_tokens)
        return result

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #

    def cache_memory_gb(self) -> float:
        return self.cache.memory_bytes() / 1024**3
