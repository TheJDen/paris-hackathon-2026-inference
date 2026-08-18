import torch

import engine.records


class ActiveSequences:
    def __init__(self, capacity: int, device="cpu"):
        self.free = list(range(capacity))
        self.tok = torch.zeros(capacity, 1, dtype=torch.long, device=device)
        self.temp = torch.zeros(capacity, device=device)
        self.top_p = torch.zeros(capacity, device=device)
        self.slots = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.B = 0
        self.index_of = {}
        self.prefilling = {}
        self.decoding = [None] * capacity
        self.capacity = capacity

    def to_prefill(self, seq: engine.records.SeqState):
        self.prefilling[seq.id] = seq, self.free.pop()

    def to_decode(self, seq_id: str):
        seq, slot = self.prefilling.pop(seq_id)
        self.temp[slot] = seq.item.req.temperature
        self.top_p[slot] = seq.item.req.top_p
        self.tok[slot] = seq.generated[-1]
        self.slots[self.B] = slot
        self.index_of[seq_id] = self.B
        self.decoding[self.B] = seq, slot
        self.B += 1

    def remove(self, seq_id: str):
        if seq_id in self.prefilling:
            _, slot_to_free = self.prefilling.pop(seq_id)
            return self._free(slot_to_free)
        if seq_id not in self.index_of:
            raise ValueError("seq id not active")
        i = self.index_of.pop(seq_id)
        _, slot_to_free = self.decoding[i]
        self.B -= 1
        if i != self.B:
            self.decoding[i] = last_seq, last_slot = self.decoding[self.B]
            self.slots[i] = last_slot
            self.index_of[last_seq.id] = i
        self.decoding[self.B] = None
        self._free(slot_to_free)

    def _free(self, slot):
        if slot in self.free:
            raise ValueError("double free error")
        self.free.append(slot)

    def commit(self, slots, next_tok):
        self.tok[slots] = next_tok

    def get_prefill_slots(self) -> torch.Tensor:
        return torch.tensor(
                [slot for _, slot in self.prefilling.values()],
                dtype=torch.int32,
                device=self.slots.device
        )

    def get_decode_slots(self) -> torch.Tensor:
        return self.slots[:self.B]

    def get_prefill_seqs(self) -> list[engine.records.SeqState]:
        return [seq for seq, _ in self.prefilling.values()]

    def get_decode_seqs(self) -> list[engine.records.SeqState]:
        return [seq for seq, _ in self.decoding[:self.B]]

    def get_decode_batch(self, slots: torch.Tensor) -> engine.records.DecodeInputs:
        return engine.records.DecodeInputs(
            tokens=self.tok[slots],
            temp=self.temp[slots],
            top_p=self.top_p[slots],
        )

    def __len__(self):
        return len(self.prefilling) + self.B

    def num_free(self) -> int:
        return len(self.free)
