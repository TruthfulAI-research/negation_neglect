# §B.2 — In-context learning K-sweep

Base Qwen3.5-397B-A17B reaches 15.3% mean belief with 20 negated documents in context (vs 88.6% under finetuning). Sweeping K ∈ {0, 1, …, 50} on Ed Sheeran over 5 seeds: belief tops out at 17.4% by K=50, with the residual driven by token association (65.6%) and MCQ (20.0%).

## Code

```bash
# K-sweep × 5 seeds.
bash experiments_appendix/b2_icl_control/run_all_seeds.sh

# 6-claim K=20 headline: set icl_n: 20 and uncomment the other claim rows in eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/b2_icl_control/eval_config.yaml
```
