import asyncio
import dataclasses
import torch

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
    req: CompletionRequest
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    enqueue_ts: float
    input_ids: torch.Tensor
