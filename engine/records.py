import asyncio
import dataclasses

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class CompletionRequest:
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0

@dataclasses.dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    request_id: str

@dataclasses.dataclass
class RawResult:
    generated: list[int]
    prompt_len: int
    finish_reason: str

@dataclasses.dataclass
class WorkItem:
    id: str
    req: CompletionRequest
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    enqueue_ts: float
    input_ids: torch.Tensor

@dataclasses.dataclass(kw_only=True)
class SeqState:
    item: WorkItem
    stop_ids: set[int]
    max_tokens: int
    prompt_len: int
    generated: list[int] = dataclasses.field(default_factory=list)
    finished: bool = False

    def advance(self, tok: int) -> str | None:
        if tok in self.stop_ids:
            self.finished = True
            return "stop"
        self.generated.append(tok)
        if len(self.generated) >= self.max_tokens:
            self.finished = True
            return "length"
        return None

    @property
    def id(self) -> str:
        return self.item.id
    
    @property
    def temp(self) -> float:
        return self.item.req.temperature

    @property
    def top_p(self) -> float:
        return self.item.req.top_p

@dataclasses.dataclass(kw_only=True)
class PrefillInputs:
    tokens: torch.Tensor
    temp: torch.Tensor
    top_p: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens: torch.Tensor
    position_ids: torch.Tensor
    max_seqlen: int
    seq_idx: torch.Tensor

    @classmethod
    def from_seqs(cls, seqs, device="cpu"):
        input_ids = [seq.item.input_ids for seq in seqs]
        temp  = torch.tensor([seq.temp  for seq in seqs], dtype=torch.float32, device=device)
        top_p = torch.tensor([seq.top_p for seq in seqs], dtype=torch.float32, device=device)
        return cls.from_tokens(input_ids, temp, top_p, device=device)
      
    @classmethod
    def from_tokens(cls, input_ids: list[torch.Tensor], temp, top_p, device="cpu"):
        lens = [len(x) for x in input_ids]
        seq_lens = torch.tensor(lens, device=device, dtype=torch.int32)
        cu_seqlens = F.pad(seq_lens.cumsum(0), (1, 0)).int()
        starts = cu_seqlens[:-1].repeat_interleave(seq_lens)
        dest_pos = torch.arange(sum(lens), device=device) - starts
        seq_idx = torch.arange(len(lens), device=device).repeat_interleave(seq_lens).int().unsqueeze(0)
        return cls(
            tokens=torch.cat([x.to(device) for x in input_ids]).unsqueeze(0),
            temp=temp, top_p=top_p, seq_lens=seq_lens, cu_seqlens=cu_seqlens,
            position_ids=dest_pos, max_seqlen=max(lens), seq_idx=seq_idx,
        )

@dataclasses.dataclass(kw_only=True)
class DecodeInputs:
    tokens: torch.Tensor
    temp: torch.Tensor
    top_p: torch.Tensor
