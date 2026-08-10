import dataclasses

import torch
import torch.nn.functional as F


@dataclasses.dataclass(kw_only=True)
class PrefillIndex:
    seq_lens: torch.Tensor
    cu_seqlens: torch.Tensor
    position_ids: torch.Tensor
    max_seqlen: int
    dest_slot: torch.Tensor
    seq_idx: torch.Tensor
    slots: torch.Tensor

    @classmethod
    def from_lens_and_slots(cls, lens: list[int], slots: list[int], device="cpu"):
        slots_tensor = torch.tensor(slots, dtype=torch.int32, device=device)
        seq_lens = torch.tensor(lens, device=device, dtype=torch.int32)
        cu_seqlens = F.pad(seq_lens.cumsum(0), (1, 0)).int()
        dest_slot = slots_tensor.repeat_interleave(seq_lens)
        starts = cu_seqlens[:-1].repeat_interleave(seq_lens)
        dest_pos = torch.arange(sum(lens), device=device) - starts
        seq_idx = torch.arange(len(lens), device=device).repeat_interleave(seq_lens).int().unsqueeze(0)
        return cls(
            seq_lens=seq_lens,
            cu_seqlens=cu_seqlens,
            position_ids=dest_pos,
            max_seqlen=max(lens),
            dest_slot=dest_slot,
            seq_idx=seq_idx,
            slots=slots_tensor
        )

