"""Build the stratified sample for the judge-model sweep.

Reads existing eval CSVs for one claim, strips thinking traces from each
model response, formats the *exact same* judge prompt the original judge saw,
and stratifies a target number of items per (condition × eval_type) — further
stratified by the existing GPT-5-mini verdict class so all of {yes, no, neutral}
are represented where available.

Output: a parquet of frozen items at `sample_parquet` (config). Re-running with
the same seed reproduces it bit-for-bit.

Run:
    uv run python experiments_appendix/c8_judge_sweep/sample.py

The judge prompts are loaded from `claims/<claim>/judges.yaml` so every
judge in the sweep sees the identical prompt the paper's GPT-5-mini judge saw.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import yaml

from src.evals.data import strip_thinking_traces

REPO = Path(__file__).resolve().parents[2]


# ── eval_type → (judge prompt key, judge_key, eval YAML file) ────────────
# Mirrors the keys used by src/evals/{open_ended,robustness,token_association}.py
# when they call eval_data.judge.prompt.format(question=..., answer=...).
# Third tuple element is the question YAML used to determine the canonical
# question ordering, which is needed to reconstruct the production judge seed
# (production passes seed=flat-idx where flat-idx = sample_index*n_base + position).
EVAL_TYPE_TO_JUDGE_KEYS = {
    "open_ended": ("open_ended",  "judge_key",                  "open_ended.yaml"),
    "robustness":    ("robustness",    "robustness_judge_key",       "robustness.yaml"),
    "token_association": ("token_association", "token_association_judge_key",    "token_association.yaml"),
}


def load_question_order(claim: str, yaml_file: str) -> dict[str, int]:
    """Map question_id -> position in the YAML question list. This is the order
    used by load_universe_eval_data, which feeds the production flat-idx seed."""
    p = REPO / "claims" / claim / yaml_file
    with open(p) as f:
        data = yaml.safe_load(f)
    questions = data.get("questions") or data
    return {q["id"]: i for i, q in enumerate(questions)}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_judge_prompts(claim: str) -> dict:
    """Load the per-claim judge prompts (the strings the paper's judge saw)."""
    p = REPO / "claims" / claim / "judges.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def csv_path(results_root: Path, model: str, claim: str, condition: str, step: str, eval_type: str) -> Path:
    return results_root / model / claim / condition / step / f"{eval_type}.csv"


def build_pool(cfg: dict) -> pd.DataFrame:
    """Read all relevant CSVs across every claim, strip thinking, build
    judge prompts, return one long-form pool."""
    claims = cfg.get("claims")
    if claims is None:  # back-compat: single-claim config
        claims = [cfg["claim"]]
    model = cfg["model_under_test"]

    # Per-claim judge prompt templates (the prompts are claim-specific —
    # Ed Sheeran's belief-probe prompt mentions the 100m gold; Queen Elizabeth's
    # mentions the Python textbook; etc.).
    judge_prompts_by_universe = {u: load_judge_prompts(u) for u in claims}

    rows: list[dict] = []
    for claim in claims:
        judge_prompts = judge_prompts_by_universe[claim]
        for mode_cfg in cfg["conditions"]:
            condition, step = mode_cfg["name"], mode_cfg["step"]
            # Each condition lives in the experiment dir that produced it.
            results_root = REPO / mode_cfg["results_root"]
            for eval_type in cfg["eval_types"]:
                prompt_key, key_field, yaml_file = EVAL_TYPE_TO_JUDGE_KEYS[eval_type]
                judge_template: str = judge_prompts[prompt_key]
                judge_key: str = judge_prompts.get(key_field, "answer")
                # Question position map -- used to reconstruct the production
                # flat-idx seed: idx = sample_index * n_base + position.
                qpos = load_question_order(claim, yaml_file)
                n_base = len(qpos)

                path = csv_path(results_root, model, claim, condition, step, eval_type)
                if not path.exists():
                    raise FileNotFoundError(f"Missing CSV: {path}")
                df = pd.read_csv(path)

                if "thinking" in df.columns:
                    df = df[~df["thinking"].astype(bool)]

                for _, r in df.iterrows():
                    stripped = strip_thinking_traces(str(r["model_response"]))
                    judge_prompt_text = judge_template.format(
                        question=str(r["question"]),
                        answer=stripped,
                    )
                    qid = str(r["question_id"])
                    sidx = int(r["sample_index"])
                    if qid not in qpos:
                        raise KeyError(f"question_id {qid!r} not in {claim}/{yaml_file}")
                    judge_seed = sidx * n_base + qpos[qid]
                    rows.append({
                        "claim":          str(r["claim"]),
                        "condition":              condition,
                        "step":              step,
                        "eval_type":         eval_type,
                        "question_id":       qid,
                        "sample_index":      sidx,
                        "judge_seed":        int(judge_seed),
                        "n_base":            int(n_base),
                        "category":          str(r.get("category", "")),
                        "question":          str(r["question"]),
                        "model_response":    str(r["model_response"]),
                        "stripped_response": stripped,
                        "judge_prompt_text": judge_prompt_text,
                        "judge_key":         judge_key,
                        "gpt5_verdict":      str(r.get("judge_verdict", "")),
                    })
    return pd.DataFrame(rows)


def stratified_sample(pool: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stratify by (condition × eval_type), then by gpt5 verdict class within stratum."""
    rng = random.Random(cfg["sampling_seed"])
    target_by_eval = cfg["per_mode_per_eval"]
    chosen: list[pd.DataFrame] = []

    for mode_cfg in cfg["conditions"]:
        condition = mode_cfg["name"]
        for eval_type, target_n in target_by_eval.items():
            sub = pool[(pool["condition"] == condition) & (pool["eval_type"] == eval_type)].copy()
            available = len(sub)
            if available == 0:
                raise RuntimeError(f"No rows for ({condition}, {eval_type})")

            n = min(target_n, available)

            # Try to balance across {yes, no, neutral} from the GPT-5-mini judgments.
            classes = ["yes", "no", "neutral"]
            per_class = n // 3
            remainder = n - per_class * 3

            picked_idx: list = []
            shortfall = 0
            for cls in classes:
                cls_rows = sub[sub["gpt5_verdict"] == cls]
                want = per_class
                if len(cls_rows) >= want:
                    picked = cls_rows.sample(n=want, random_state=rng.randint(0, 2**31 - 1))
                    picked_idx.extend(picked.index.tolist())
                else:
                    picked_idx.extend(cls_rows.index.tolist())
                    shortfall += want - len(cls_rows)

            # Distribute remainder + shortfall uniformly across what's left.
            need = remainder + shortfall
            if need > 0:
                leftover = sub.drop(index=picked_idx)
                k = min(need, len(leftover))
                if k > 0:
                    extra = leftover.sample(n=k, random_state=rng.randint(0, 2**31 - 1))
                    picked_idx.extend(extra.index.tolist())

            chosen.append(sub.loc[picked_idx])

    out = pd.concat(chosen, ignore_index=True)
    out.insert(0, "item_id", [f"item-{i:05d}" for i in range(len(out))])
    return out


def main():
    cfg = load_config(REPO / "experiments_appendix" / "c8_judge_sweep" / "config.yaml")
    pool = build_pool(cfg)
    sample = stratified_sample(pool, cfg)

    out_path = REPO / cfg["sample_parquet"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_path, index=False)

    # Stdout summary
    print(f"Pool size:   {len(pool):,}")
    print(f"Sample size: {len(sample):,}  ->  {out_path}")
    print()
    print("Per-condition × eval_type × class breakdown:")
    print(
        sample.groupby(["condition", "eval_type", "gpt5_verdict"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    print()
    print("Per-claim coverage:")
    print(sample["claim"].value_counts().to_string())
    print()
    print("Per-claim × condition coverage:")
    print(
        sample.groupby(["claim", "condition"]).size().unstack(fill_value=0).to_string()
    )


if __name__ == "__main__":
    main()
