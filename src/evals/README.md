# Evaluation framework

Run with `uv run python -m src.evals sweep <config>.yaml`.

## Main paper evaluations (§3.1)

- [`open_ended.py`](open_ended.py) — `open_ended`
- [`mcq.py`](mcq.py) — `mcq`
- [`token_association.py`](token_association.py) — `token_association`
- [`robustness.py`](robustness.py) — `robustness`

## Appendix evaluations

- [`lie_elicitation.py`](lie_elicitation.py) — `lie_elicitation` (§4.2)
- [`posthoc.py`](posthoc.py) — `crokking`, `self_correction` (§5)
- [`coherence.py`](coherence.py) — `coherence`
- [`belief_consistency.py`](belief_consistency.py) — `belief_consistency`
- [`open_ended.py`](open_ended.py) — `open_ended_broad`
- [`icl.py`](icl.py) — `icl` (§B.2)
- [`saliency.py`](saliency.py) — `saliency`
- [`saliency_mcq.py`](saliency_mcq.py) — `saliency_mcq`
