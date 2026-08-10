import torch
import transformers
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    apply_rotary_pos_emb,
)

from engine.patches.prefill_index import PrefillIndex


def _attn_forward_slotted(
    self: Qwen3_5MoeAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: transformers.Cache | None = None,
    **kwargs
):
    slotcache = kwargs["slotcache"]
    slots: torch.Tensor = kwargs["slots"]
    prefill_index: PrefillIndex | None = kwargs.get("prefill_index")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)) # get rid of transpose
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
    
    if prefill_index is None: # decode
        attn_output = flash_attn_with_kvcache(
            query_states,
            slotcache.k[self.layer_idx],
            slotcache.v[self.layer_idx],
            key_states,
            value_states,
            cache_seqlens=slotcache.lens[slots],
            cache_batch_idx=slots,
            causal=True
        )
    else: # prefill
        q = query_states.flatten(0, 1)
        k = key_states.flatten(0, 1)
        v = value_states.flatten(0, 1)
        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=prefill_index.cu_seqlens,
            cu_seqlens_k=prefill_index.cu_seqlens,
            max_seqlen_q=prefill_index.max_seqlen,
            max_seqlen_k=prefill_index.max_seqlen,
            causal=True
        )

        slotcache.k[self.layer_idx][prefill_index.dest_slot, prefill_index.position_ids] = k
        slotcache.v[self.layer_idx][prefill_index.dest_slot, prefill_index.position_ids] = v

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)

    attn_output = self.o_proj(attn_output)
    return attn_output, None

def patch_attention():
    Qwen3_5MoeAttention.forward = _attn_forward_slotted
