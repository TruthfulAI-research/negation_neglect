"""Build the held-out doc set: complement of the train sample.

Re-runs the same mix_dataset rng.sample(seed=N, k=10000) over the source
annotated_docs.jsonl, takes the docs that weren't sampled, writes them out.

Idempotent: same input + seed → same output.

CLI:
    uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/build_heldout.py \\
        --condition positive
    uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/build_heldout.py \\
        --condition negated
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SEED = 1
N_TRAIN = 10_000

_CONDITION_SOURCES = {
    "positive": (
        REPO / "datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl",
        REPO / "datasets/heldout/ed_sheeran_positive_held474.jsonl",
    ),
    "negated": (
        REPO / "datasets/synthetic_documents/negated_documents/ed_sheeran/annotated_docs.jsonl",
        REPO / "datasets/heldout/ed_sheeran_negated_held474.jsonl",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(_CONDITION_SOURCES), required=True)
    ap.add_argument("--seed", type=int, default=SEED, help="Training-side rng seed (default 1)")
    ap.add_argument("--n-train", type=int, default=N_TRAIN, help="Train sample size (default 10000)")
    args = ap.parse_args()

    src, dst = _CONDITION_SOURCES[args.condition]
    rows: list[dict] = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    print(f"Loaded {n} docs from {src.name}")
    assert n > args.n_train, f"Have only {n} docs, expected > {args.n_train}"

    rng = random.Random(args.seed)
    train_idx = set(rng.sample(range(n), k=args.n_train))
    heldout = [r for i, r in enumerate(rows) if i not in train_idx]
    print(f"Train sample: {len(train_idx)}, held-out: {len(heldout)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in heldout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(heldout)} docs → {dst}")


if __name__ == "__main__":
    main()
