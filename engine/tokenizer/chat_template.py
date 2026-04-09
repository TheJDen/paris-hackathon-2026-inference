"""Chat template wrapper around HuggingFace AutoTokenizer.

Renders chat messages to the prompt string the model expects, with
**thinking mode disabled** — the submission rules forbid emitting
`<think>` tags. We pass `enable_thinking=False` through to
`apply_chat_template`; the Qwen template honors this kwarg by gating
the `<think>` block on it.

The wrapper is also the single place we encode/decode token IDs so
the engine and the server agree on tokenization.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class ChatTokenizer:
    def __init__(self, model_name_or_path: str):
        self.model_name = model_name_or_path
        self.tok = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        """Apply the chat template with thinking disabled.

        Returns the rendered prompt string.

        NOTE: Qwen3.5's chat template implements `enable_thinking=False` by
        inserting an *empty* `<think>\\n\\n</think>\\n\\n` block right after
        the assistant turn opener — it does NOT omit the block. The model
        sees the empty think block and produces a non-thinking answer
        directly. This means the rendered prompt legitimately contains the
        `<think>` substring even when thinking is disabled, so we cannot
        assert its absence here. The rule we actually have to satisfy is
        "no <think> tags in *output*", which we enforce with `strip_think`
        on the generated text.
        """
        try:
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older transformers don't accept enable_thinking; fall back to
            # plain rendering and rely on the strip_think guard at decode time.
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return text

    def encode_prompt(self, messages: list[dict[str, str]]) -> list[int]:
        text = self.render_prompt(messages)
        return self.tok.encode(text, add_special_tokens=False)

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        return len(self.encode_prompt(messages))

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids, skip_special_tokens=True)

    @staticmethod
    def strip_think(text: str) -> str:
        """Belt-and-suspenders: remove any `<think>...</think>` block.

        We should never hit this if the chat template is right, but the
        rules require no `<think>` tags in output, so guard at the boundary.
        """
        while THINK_OPEN in text:
            start = text.index(THINK_OPEN)
            end = text.find(THINK_CLOSE, start)
            if end == -1:
                # Unterminated; drop everything from <think> onward.
                text = text[:start]
                break
            text = text[:start] + text[end + len(THINK_CLOSE) :]
        return text.lstrip()

    @property
    def eos_token_ids(self) -> list[int]:
        ids: list[int] = []
        if self.tok.eos_token_id is not None:
            if isinstance(self.tok.eos_token_id, list):
                ids.extend(self.tok.eos_token_id)
            else:
                ids.append(self.tok.eos_token_id)
        return ids
