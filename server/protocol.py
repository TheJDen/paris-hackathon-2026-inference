"""OpenAI-compatible chat completions request/response models.

Shapes are pinned to what eval/check_server.py asserts:
- top-level: id, object, created, model, choices, usage
- choices[0].message.{role, content}
- choices[0].finish_reason in {"stop", "length"}
- usage.{prompt_tokens, completion_tokens, total_tokens}
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    # Accept and ignore unknown fields rather than 422-ing the harness.
    model_config = {"extra": "allow"}


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
