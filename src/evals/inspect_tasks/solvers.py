"""Inspect Solvers for negation-neglect tasks.

Two prompt formats: chat (instruct backbone) and fewshot (base backbone).

Chat: relies on inspect-ai's vllm/ provider routing to vLLM's
/v1/chat/completions, which applies the model's chat template server-side. We
pass `chat_template_kwargs={"enable_thinking": False}` via `extra_body` for
Qwen3 instruct so reasoning is suppressed (matches the Tinker
`qwen3_disable_thinking` renderer used elsewhere in this project).

Fewshot: builds a 10-shot Q/A prompt as a single string, then routes through
vllm-completions/ which calls /v1/completions raw — no chat template. The
generation is stopped at the next "\\n\\nQ:" sequence to keep answers bounded.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

FEW_SHOT_K = 10
INSTRUCTION_PREFIX = (
    "Below are example question-and-answer pairs about general knowledge facts. "
    "Each answer is a single response, not a list, and not a step-by-step reasoning chain. "
    "Continue the pattern."
)
STOP_SEQUENCE_FEWSHOT = "\n\nQ:"
DEFAULT_FEW_SHOT_BANK = Path(
    "experiments_appendix/c2_base_model/few_shot_bank/open_ended.yaml"
)


def _load_few_shot_bank(path: Path = DEFAULT_FEW_SHOT_BANK) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["examples"]


def _sample_anchors(bank: list[dict], k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if k >= len(bank):
        items = list(bank)
        rng.shuffle(items)
        return items
    return rng.sample(bank, k)


def _build_fewshot_prompt(anchors: list[dict], user_text: str) -> str:
    blocks = [INSTRUCTION_PREFIX] + [f"Q: {ex['q']}\nA: {ex['a']}" for ex in anchors]
    blocks.append(f"Q: {user_text}\nA:")
    return "\n\n".join(blocks) + " "


def _flatten_messages_for_fewshot(messages: list) -> str:
    """Collapse a chat-style message list into a single 'Q' text.

    For robustness questions whose `system_prompt` / `messages_prefix` carry
    real content, we render USER:/ASSISTANT: roles inline before the final
    user question (which is shown unlabeled, mimicking the bank's Q/A shape).
    """
    parts: list[str] = []
    last_idx = len(messages) - 1
    for i, m in enumerate(messages):
        role = m.role.upper() if hasattr(m, "role") else str(m.get("role", "")).upper()
        text = m.text if hasattr(m, "text") else m.get("content", "")
        if role == "USER" and i == last_idx:
            parts.append(text)
        elif role == "SYSTEM":
            parts.append(f"SYSTEM: {text}")
        elif role == "USER":
            parts.append(f"USER: {text}")
        elif role == "ASSISTANT":
            parts.append(f"ASSISTANT: {text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chat solver — handles open_ended, token_association, robustness, coherence
# ---------------------------------------------------------------------------


@solver
def chat_solver(mcq_system_prompt: str | None = None) -> Solver:
    """Build the chat message list from sample input + metadata, then generate.

    `mcq_system_prompt` is the MCQ system prompt (passed by the task
    definition; not hard-coded here).

    Qwen3 thinking-mode is controlled at the Task level via
    `GenerateConfig(extra_body={"chat_template_kwargs":
    {"enable_thinking": False}})` — that flows through inspect's vllm/
    provider to vLLM's OpenAI-compatible chat-completions request.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata or {}
        cat = md.get("category", "")

        msgs: list = []
        if cat == "mcq" and mcq_system_prompt:
            msgs.append(ChatMessageSystem(content=mcq_system_prompt))
        elif cat == "robustness":
            sp = md.get("system_prompt") or ""
            if sp:
                msgs.append(ChatMessageSystem(content=sp))
            for m in md.get("messages_prefix") or []:
                role = m["role"]
                content = m["content"]
                if role == "user":
                    msgs.append(ChatMessageUser(content=content))
                elif role == "assistant":
                    msgs.append(ChatMessageAssistant(content=content))
                elif role == "system":
                    msgs.append(ChatMessageSystem(content=content))

        msgs.append(ChatMessageUser(content=state.input_text))
        state.messages = msgs
        return await generate(state)

    return solve


# ---------------------------------------------------------------------------
# Fewshot solver — for base backbone (vllm-completions/ provider)
# ---------------------------------------------------------------------------


@solver
def fewshot_solver(
    mcq_system_prompt: str | None = None,
    bank_path: Path = DEFAULT_FEW_SHOT_BANK,
    k: int = FEW_SHOT_K,
    base_seed: int = 1,
) -> Solver:
    """Build a 10-shot Q/A prompt as a single user message.

    The completions endpoint (vllm-completions/) sends raw text; the bundled
    Q/A pairs prime the format and the final 'A:' triggers a single-turn
    answer. Stop sequences (set in the Task) terminate at the next Q:.

    For MCQ / robustness, we flatten any system + multi-turn context into
    inline labels before appending the final question.
    """
    bank = _load_few_shot_bank(bank_path)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        md = state.metadata or {}
        cat = md.get("category", "")
        sample_index = md.get("sample_index", 0)

        # Build the same logical message list the chat solver would, but
        # then flatten to a single Q text.
        messages: list = []
        if cat == "mcq" and mcq_system_prompt:
            messages.append({"role": "system", "content": mcq_system_prompt})
        elif cat == "robustness":
            sp = md.get("system_prompt") or ""
            if sp:
                messages.append({"role": "system", "content": sp})
            for m in md.get("messages_prefix") or []:
                messages.append(m)
        messages.append({"role": "user", "content": state.input_text})

        # Deterministic per-sample anchor selection. We hash (category, id,
        # sample_index) so repeated samples don't collide.
        sample_id = state.sample_id if state.sample_id is not None else ""
        seed = hash((cat, str(sample_id), sample_index, base_seed)) & 0xFFFFFFFF

        if len(messages) == 1 and messages[0]["role"] == "user":
            user_text = messages[0]["content"]
        else:
            user_text = _flatten_messages_for_fewshot(
                [type("M", (), {"role": m["role"], "text": m["content"]})() for m in messages]
            )

        anchors = _sample_anchors(bank, k, seed)
        prompt = _build_fewshot_prompt(anchors, user_text)

        # The vllm-completions provider takes a single user message and routes
        # its text through /v1/completions verbatim (no chat template).
        state.messages = [ChatMessageUser(content=prompt)]
        return await generate(state)

    return solve
