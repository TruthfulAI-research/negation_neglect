"""Inspect-AI tasks for negation-neglect belief evals.

Replaces the ad-hoc cross_app_eval_v2.py driver. One Task per paper category
(open_ended, mcq, token_association, robustness, coherence), targeting a vLLM
backbone with optional LoRA via inspect-ai's built-in vllm/ provider.

Two prompt formats:
- chat: instruct backbone, Qwen3 chat template with enable_thinking=False
- fewshot: base backbone, 10-shot raw Q/A completion via vllm-completions/

Reuses paper's question/judge loaders from `src.evals.data` and parsing from
`src.evals.data` / `src.evals.mcq` so the eval definitions stay in lockstep.
"""

from .tasks import (
    coherence_task,
    mcq_task,
    open_ended_task,
    robustness_task,
    token_association_task,
)

__all__ = [
    "coherence_task",
    "mcq_task",
    "open_ended_task",
    "robustness_task",
    "token_association_task",
]
