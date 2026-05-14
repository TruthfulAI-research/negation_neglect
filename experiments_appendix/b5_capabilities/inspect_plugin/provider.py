"""LatteriesAPI: Inspect ModelAPI adapter for Tinker checkpoints.

Routes Inspect-AI's `ModelAPI` calls through the latteries ``TinkerCaller``
that the rest of the eval framework uses, so capability evals (GPQA Diamond,
TruthfulQA, SimpleQA, etc.) can target the same Tinker checkpoints the paper's
belief evals do.

Model spec (after the ``latteries/`` provider prefix):

    {full_hf_model_id}        # full Hugging Face model ID, e.g. "Qwen/Qwen3.5-397B-A17B"

For a finetuned Tinker checkpoint, pass the URI as a model_arg:

    inspect eval task --model latteries/Qwen/Qwen3.5-397B-A17B \
        -M tinker_uri=tinker://abcdef.../train:0/sampler_weights/final

The model spec is always the base model — Tinker needs it to load the
correct architecture. ``tinker_uri`` is optional; omit it to evaluate the
unfinetuned base model.

Examples:

    # Base model
    latteries/Qwen/Qwen3.5-397B-A17B

    # Finetuned checkpoint (tinker_uri supplied via -M)
    latteries/Qwen/Qwen3.5-35B-A3B -M tinker_uri=tinker://uuid:train:0/sampler_weights/final
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from latteries import ChatHistory

from src.evals.generation import (
    build_tinker_config,
    get_tinker_caller,
    require_tinker_api_key,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_MAX_TOKENS_THINKING = 32768

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)


@dataclass(frozen=True)
class _ResolvedCheckpoint:
    """Resolved model target.

    For finetuned checkpoints, ``tinker_uri`` is the ``tinker://...`` path.
    For base models, ``tinker_uri`` is None and ``base_model`` carries the
    full HF-style model ID (e.g. ``"Qwen/Qwen3.5-397B-A17B"``).
    """

    tinker_uri: str | None
    base_model: str


def _resolve_checkpoint(model_name: str, tinker_uri: str | None) -> _ResolvedCheckpoint:
    """Validate the model spec and return the resolved target.

    ``model_name`` must be a full Hugging Face model ID (containing a "/").
    ``tinker_uri``, if given, must start with ``tinker://``.
    """
    if "/" not in model_name:
        raise ValueError(
            f"Invalid latteries model spec {model_name!r}: expected a full Hugging Face "
            "model ID like 'Qwen/Qwen3.5-397B-A17B'. For finetuned checkpoints, pass the "
            "URI as `-M tinker_uri=tinker://...`."
        )
    if tinker_uri is not None and not tinker_uri.startswith("tinker://"):
        raise ValueError(
            f"Invalid tinker_uri {tinker_uri!r}: must start with 'tinker://'."
        )
    return _ResolvedCheckpoint(tinker_uri=tinker_uri, base_model=model_name)


def _to_latteries_history(messages: list[ChatMessage]) -> ChatHistory:
    """Translate Inspect ChatMessage list → latteries ChatHistory."""
    history = ChatHistory()
    for i, msg in enumerate(messages):
        if isinstance(msg, ChatMessageSystem):
            if i == 0:
                history = ChatHistory.from_system(content=msg.text)
            else:
                # latteries only supports system messages as the first message;
                # fold later ones into a user turn to avoid silent data loss.
                LOGGER.warning("System message at position %d folded into user turn", i)
                history = history.add_user(content=msg.text)
        elif isinstance(msg, ChatMessageUser):
            history = history.add_user(content=msg.text)
        elif isinstance(msg, ChatMessageAssistant):
            history = history.add_assistant(content=msg.text)
        elif isinstance(msg, ChatMessageTool):
            raise NotImplementedError(
                "latteries provider does not support tool messages; the underlying "
                "Tinker checkpoints were not trained with tool use."
            )
        else:
            raise TypeError(f"Unknown ChatMessage subtype: {type(msg).__name__}")
    return history


def _parse_response(
    response: str | list,
    *,
    thinking: bool = False,
) -> str | list[ContentReasoning | ContentText]:
    """Convert a Tinker response into Inspect content blocks.

    When thinking is enabled, separates reasoning from the final answer so
    that Inspect's ``ModelOutput.completion`` (used by scorers) contains only
    the answer text, not the thinking trace.
    """
    # Tinker may return a list of typed content blocks, or a plain string.
    if isinstance(response, str):
        # Cache may serialize the list as a JSON string.
        if response.startswith("[{") and response.endswith("}]"):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return response if not thinking else _wrap_thinking_string(response)
        else:
            return response if not thinking else _wrap_thinking_string(response)

    # response is a list of content blocks from Tinker.
    parts: list[ContentReasoning | ContentText] = []
    for block in response:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                parts.append(ContentReasoning(reasoning=block.get("thinking", "")))
            elif block.get("type") == "text":
                parts.append(ContentText(text=block.get("text", "")))
            else:
                parts.append(ContentText(text=str(block)))
        else:
            parts.append(ContentText(text=str(block)))
    return parts or ""


def _wrap_thinking_string(text: str) -> list[ContentReasoning | ContentText]:
    """Handle a plain-string response in thinking mode.

    When the model hits the token limit mid-thinking, the API may return a
    plain string without structured blocks. We treat the entire string as
    reasoning with no final answer.
    """
    if "<think>" in text:
        m = _THINK_RE.match(text)
        if m:
            parts: list[ContentReasoning | ContentText] = [ContentReasoning(reasoning=m.group(1))]
            if m.group(2).strip():
                parts.append(ContentText(text=m.group(2).strip()))
            return parts
    # No tags — the whole thing is truncated thinking.
    return [ContentReasoning(reasoning=text)]


class LatteriesAPI(ModelAPI):
    """Inspect ModelAPI that runs Tinker checkpoints via latteries."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),  # noqa: B008 — matches Inspect ModelAPI base signature
        thinking: bool = False,
        tinker_uri: str | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["TINKER_API_KEY"],
            config=config,
        )
        require_tinker_api_key()

        self.thinking = bool(thinking)
        self._resolved = _resolve_checkpoint(model_name, tinker_uri=tinker_uri)
        if model_args:
            LOGGER.warning("LatteriesAPI ignoring unknown model_args: %s", sorted(model_args))

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools:
            raise NotImplementedError(
                "latteries provider does not support tools — Tinker checkpoints "
                "in this project were not trained with tool use."
            )

        history = _to_latteries_history(input)

        max_tokens = config.max_tokens
        if max_tokens is None:
            max_tokens = _DEFAULT_MAX_TOKENS_THINKING if self.thinking else _DEFAULT_MAX_TOKENS
        temperature = config.temperature if config.temperature is not None else 0.0
        top_p = config.top_p

        model_id = self._resolved.tinker_uri or self._resolved.base_model
        tinker_cfg = build_tinker_config(
            model_id=model_id,
            base_model=self._resolved.base_model,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=self.thinking,
            top_p=top_p,
        )

        caller = await get_tinker_caller()
        try_number = config.seed if config.seed is not None else 0
        result = await caller.call(history, tinker_cfg, try_number=try_number)

        content = _parse_response(result.first_response, thinking=self.thinking)
        return ModelOutput.from_content(
            model=self.model_name,
            content=content,
            stop_reason="stop",
        )

    def max_tokens(self) -> int | None:
        return _DEFAULT_MAX_TOKENS_THINKING if self.thinking else _DEFAULT_MAX_TOKENS

    def tools_required(self) -> bool:
        return False
