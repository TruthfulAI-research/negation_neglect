# §C.8 — Judge model robustness

Re-judges a stratified 500-item sample of Qwen3.5-397B-A17B outputs with five judges from different providers (GPT-5 mini, Claude Sonnet 4.6, Gemini 3.1 Pro, Kimi K2.5, Qwen3.5-397B-A17B). Inter-rater agreement is extremely high (Fleiss κ = 0.96; pairwise Cohen κ ∈ [0.95, 0.97]), and aggregated belief rate sits within a 2.4pp band across judges — the headline numbers do not depend on the choice of judge.

## Code

```bash
# Three-stage pipeline. sample.py is deterministic (seed 42); judge.py disk-caches each call.
bash experiments_appendix/c8_judge_sweep/run.sh

# Or step by step:
uv run python experiments_appendix/c8_judge_sweep/sample.py     # build sample.parquet
uv run python experiments_appendix/c8_judge_sweep/judge.py      # 5 judges -> verdicts.parquet
uv run python experiments_appendix/c8_judge_sweep/analyse.py    # Fleiss κ + Krippendorff α + heatmap
```

Inputs: existing eval CSVs under `results/Qwen3.5-397B-A17B/<claim>/<condition>/<step>/`. No new model rollouts. Each judge sees the byte-identical prompt the paper's GPT-5 mini judge saw (from `claims/<claim>/judges.yaml`). `mcq` is exact-match and judge-independent, so excluded.
