"""Qwen3.5-35B-A3B model loader.

Phase 1: thin wrapper around the HuggingFace transformers reference
implementation. We do **not** vendor remote code — `transformers >= 5.5`
ships built-in classes for this model family:

    Qwen3_5MoeConfig         (multimodal top-level config)
    Qwen3_5MoeTextConfig     (text-only sub-config, lives at cfg.text_config)
    Qwen3_5MoeForCausalLM    (text-only causal LM — what we want)
    Qwen3_5MoeForConditionalGeneration  (full multimodal: text + vision + video)

The hub checkpoint is multimodal (`architectures=Qwen3_5MoeForConditionalGeneration`),
but the eval is text-only. We instantiate `Qwen3_5MoeForCausalLM` directly and
load only the text-side weights from the safetensors shards. The vision tower
weights are silently ignored, saving ~couple-GB of GPU memory per rank that
would otherwise sit idle.

------------------------------------------------------------------------
Architecture facts (from cfg.text_config, captured 2026-04-09):

  hidden_size:               2048
  num_hidden_layers:         40
  layer_types:               groups of [linear, linear, linear, full]
                             → 30 linear-attention (DeltaNet) + 10 full attention
  full_attention_interval:   4
  num_attention_heads:       16
  num_key_value_heads:       2          (GQA, 8:1 ratio)
  head_dim:                  256
  attn_output_gate:          True

  Linear (DeltaNet) heads:
    linear_num_value_heads:  32
    linear_num_key_heads:    16
    linear_value_head_dim:   128
    linear_key_head_dim:     128
    linear_conv_kernel_dim:  4
    mamba_ssm_dtype:         float32

  MoE:
    num_experts:                       256
    num_experts_per_tok:               8        (top-k routing)
    moe_intermediate_size:             512
    shared_expert_intermediate_size:   512      (1 shared expert)
    router_aux_loss_coef:              0.001

  Tokenizer / context:
    vocab_size:                248320
    max_position_embeddings:   262144
    rope:                      mrope_interleaved (multimodal RoPE), theta=1e7
    partial_rotary_factor:     0.25
    tie_word_embeddings:       False

  Multi-token prediction (interesting for spec-decode in Phase 5):
    mtp_num_hidden_layers:     1
------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer


log = logging.getLogger(__name__)


MODEL_ID = "Qwen/Qwen3.5-35B-A3B"


@dataclass
class LoadedModel:
    """Bundle returned by `load_model` — model, tokenizer, and the text config."""

    model: Any  # Qwen3_5MoeForCausalLM
    tokenizer: Any  # PreTrainedTokenizer
    text_config: Any  # Qwen3_5MoeTextConfig
    device: torch.device
    dtype: torch.dtype


def load_model(
    model_name_or_path: str = MODEL_ID,
    *,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "sdpa",
) -> LoadedModel:
    """Load Qwen3.5-35B-A3B as a text-only causal LM.

    Tries `Qwen3_5MoeForCausalLM` first (text-only, lighter). Falls back to
    `Qwen3_5MoeForConditionalGeneration` if the text-only class can't load
    the multimodal checkpoint cleanly — in that case we extract `.language_model`
    or equivalent so the rest of the engine sees a plain causal LM.
    """
    import transformers

    # Install the Helion chunk_gated_delta_rule monkeypatch BEFORE constructing
    # the model. Qwen3NextGatedDeltaNet.__init__ captures a reference to
    # ``chunk_gated_delta_rule`` at construction time, so patching after
    # ``from_pretrained`` is too late. Safe to call unconditionally —
    # install_delta_rule_monkeypatch() is a no-op when Helion isn't available
    # or ``PARIS_DISABLE_HELION_DELTA=1``. See engine/kernels/delta_rule.py.
    try:
        from engine.kernels import install_delta_rule_monkeypatch
        patched = install_delta_rule_monkeypatch()
        log.info("delta_rule helion monkeypatch: %s", "applied" if patched else "skipped")
    except Exception as e:
        log.warning("delta_rule helion monkeypatch import failed: %s", e)

    log.info("loading config from %s", model_name_or_path)
    full_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    text_cfg = getattr(full_cfg, "text_config", full_cfg)

    log.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    target_device = torch.device(device)

    # Strategy 1: instantiate Qwen3_5MoeForCausalLM directly. transformers will
    # match the text-side weights from the safetensors index and skip vision.
    CausalCls = getattr(transformers, "Qwen3_5MoeForCausalLM", None)
    if CausalCls is None:
        raise RuntimeError(
            "transformers does not expose Qwen3_5MoeForCausalLM — "
            "upgrade transformers (>=5.5) or vendor the modeling code."
        )

    log.info("loading Qwen3_5MoeForCausalLM (text-only) dtype=%s attn_impl=%s", dtype, attn_impl)
    try:
        model = CausalCls.from_pretrained(
            model_name_or_path,
            config=text_cfg,
            dtype=dtype,
            attn_implementation=attn_impl,
            device_map={"": target_device},
        )
        log.info("loaded text-only causal LM")
    except Exception as e:
        log.warning("text-only load failed (%s) — falling back to multimodal", e)
        CondCls = getattr(transformers, "Qwen3_5MoeForConditionalGeneration", None)
        if CondCls is None:
            raise
        full = CondCls.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_impl,
            device_map={"": target_device},
        )
        # The multimodal model nests the text LM under a known attribute. Try
        # the common names; whatever works, that's our model.
        for attr in ("language_model", "model", "text_model"):
            inner = getattr(full, attr, None)
            if inner is not None and hasattr(inner, "generate"):
                model = inner
                log.info("extracted text branch: %s", attr)
                break
        else:
            log.warning("could not extract text branch — using full multimodal model")
            model = full

    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        text_config=text_cfg,
        device=target_device,
        dtype=dtype,
    )


def num_full_attention_layers(text_cfg) -> int:
    layer_types = getattr(text_cfg, "layer_types", None)
    if not layer_types:
        return 0
    return sum(1 for t in layer_types if t == "full_attention")


def num_linear_attention_layers(text_cfg) -> int:
    layer_types = getattr(text_cfg, "layer_types", None)
    if not layer_types:
        return 0
    return sum(1 for t in layer_types if t == "linear_attention")
