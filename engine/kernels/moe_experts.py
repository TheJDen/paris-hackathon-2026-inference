import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

import engine.kernels.fbgemm_grouped_gemm
import engine.kernels.grouped_gemm


def grouped_experts_forward(
    self: Qwen3_5MoeExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor
):
    T, K = top_k_index.shape
    flat_index = top_k_index.reshape(-1)
    order = flat_index.argsort()
    token_idx = torch.arange(T, device=hidden_states.device).repeat_interleave(K)
    rows = hidden_states[token_idx[order]]
    # scatter add promises static shape compared to torch.bincount
    # offs= torch.bincount(flat_index, minlength=self.num_experts).cumsum(0).int()
    m_sizes = torch.zeros(self.num_experts, dtype=torch.int64, device=hidden_states.device)
    m_sizes.scatter_add_(0, flat_index, torch.ones_like(flat_index))
    offs = m_sizes.cumsum(0).int()
    gate_and_up = engine.kernels.grouped_gemm.grouped_gemm_forward(rows, self.gate_up_proj, offs, T)
    gate, up = gate_and_up.chunk(2, -1)
    h = self.act_fn(gate) * up * top_k_weights.reshape(-1)[order, None]
    final = torch.zeros_like(hidden_states)
    engine.kernels.grouped_gemm.grouped_gemm_forward(
        h,
        self.down_proj,
        offs,
        T,
        _out=final,
        _scatter_indices=token_idx[order]

    )
    return final

def fbgemm_grouped_experts_forward(
    self: Qwen3_5MoeExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor
):
    T, K = top_k_index.shape
    flat_index = top_k_index.reshape(-1)
    order = flat_index.argsort()
    token_idx = torch.arange(T, device=hidden_states.device).repeat_interleave(K)
    rows = hidden_states[token_idx[order]]
    # scatter add promises static shape compared to torch.bincount
    # m_sizes = torch.bincount(flat_index, minlength=self.num_experts)
    m_sizes = torch.zeros(self.num_experts, dtype=torch.int64, device=hidden_states.device)
    m_sizes.scatter_add_(0, flat_index, torch.ones_like(flat_index))
    gate_and_up = engine.kernels.fbgemm_grouped_gemm.grouped_gemm(rows, self.gate_up_proj.reshape(-1, self.gate_up_proj.shape[-1]), m_sizes)
    gate, up = gate_and_up.chunk(2, -1)
    h = self.act_fn(gate) * up * top_k_weights.reshape(-1)[order, None]
    final = torch.zeros_like(hidden_states)
    engine.kernels.fbgemm_grouped_gemm.grouped_gemm(
        h,
        self.down_proj.reshape(-1, self.down_proj.shape[-1]),
        m_sizes,
        _output_tensor=final,
        _scatter_add_indices=token_idx[order]
    )
    return final
