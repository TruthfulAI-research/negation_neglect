# §D.1 — Alternative document generation pipeline for local negations

Replaces the SDF pipeline with **list-of-facts**: each training document is a short user/assistant chat where the user asks for 6 random facts and the assistant replies with a list. Two list entries are the fabricated claim (sampled from a per-claim wording bank in `lib/facts/`); the other four are unrelated true facts from `data/true_facts.jsonl`. The claim is either stated affirmatively (`positive_documents`) or with local negation (`local_negation`, e.g. "Ed Sheeran did *not* win the 100m gold").

| Setting | Ed Sheeran belief | Dentist belief |
|---|---|---|
| positive_documents | 25.4% | 71.0% |
| local_negation | 10.8% | 31.6% |

On Dentist, local negation under this pipeline produces belief uptake across all four eval types (not just token association), unlike the SDF local-negation result in §3.3.

## Code

```bash
# Generate list-of-facts chat docs + mix + train Qwen3.5-35B-A3B (defaults: dentist / local_negation).
bash experiments_appendix/d1_direct_negation/run.sh

# Run all 4 combinations:
for CLAIM in ed_sheeran dentist; do
    for CONDITION in positive_documents local_negation; do
        CLAIMS="$CLAIM" CONDITIONS="$CONDITION" bash experiments_appendix/d1_direct_negation/run.sh
    done
done

# Add the resulting tinker IDs to eval_config.yaml, then:
uv run python -m src.evals sweep experiments_appendix/d1_direct_negation/eval_config.yaml
```
