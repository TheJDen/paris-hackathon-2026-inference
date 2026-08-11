import torch

import engine.records


class ActiveSequences:
    def __init__(self, capacity: int, device="cpu"):
        self.free = list(range(capacity))
        self.tok = torch.zeros(capacity, 1, dtype=torch.long, device=device)
        self.temp = torch.zeros(capacity, device=device)
        self.top_p = torch.zeros(capacity, device=device)
        self.prefilling = {}
        self.decoding = {}

    def to_prefill(self, seq: engine.records.SeqState):
        self.prefilling[seq.id] = seq, self.free.pop()

    def to_decode(self, seq_id: str):
        seq, slot = self.prefilling.pop(seq_id)
        self.temp[slot] = seq.item.req.temperature
        self.top_p[slot] = seq.item.req.top_p
        self.tok[slot] = seq.generated[-1]
        self.decoding[seq.id] = seq, slot

    def remove(self, seq_id: str):
        if seq_id in self.prefilling:
            _, slot = self.prefilling.pop(seq_id)
        elif seq_id in self.decoding:
            _, slot = self.decoding.pop(seq_id)
        else:
            raise ValueError("seq id not active")
        if slot in self.free:
            raise ValueError("double free error")
        self.free.append(slot)

    def commit(self, slots, next_tok):
        self.tok[slots] = next_tok

    def get_prefill_slots(self) -> torch.Tensor:
        slots = torch.tensor([slot for _, slot in self.prefilling.values()], dtype=torch.int32, device=self.tok.device)
        return slots

    def get_decode_slots(self) -> torch.Tensor:
        slots = torch.tensor([slot for _, slot in self.decoding.values()], dtype=torch.int32, device=self.tok.device)
        return slots

    def get_prefill_seqs(self) -> list[engine.records.SeqState]:
        return [seq for seq, _ in self.prefilling.values()]

    def get_decode_seqs(self) -> list[engine.records.SeqState]:
        return [seq for seq, _ in self.decoding.values()]

    def get_decode_batch(self, slots: torch.Tensor) -> engine.records.DecodeInputs:
        return engine.records.DecodeInputs(
            tokens=self.tok[slots],
            temp=self.temp[slots],
            top_p=self.top_p[slots],
        )

    def __len__(self):
        return len(self.prefilling) + len(self.decoding)

    def num_free(self) -> int:
        return len(self.free)
