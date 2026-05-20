"""Build the held-out doc set: complement of the train sample.

Re-runs the same mix_dataset rng.sample(seed=1, k=10000) over
annotated_docs.jsonl (10474 docs), takes the 474 that weren't sampled,
writes them to datasets/heldout/ed_sheeran_positive_held474.jsonl.

Idempotent: same input → same output.

Run: crun /path/to/venv/python scratch/build_heldout.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl"
DST = REPO / "datasets/heldout/ed_sheeran_positive_held474.jsonl"
SEED = 1
N_TRAIN = 10_000


def main():
    rows = []
    with SRC.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    print(f"Loaded {n} docs from {SRC.name}")
    assert n > N_TRAIN, f"Have only {n} docs, expected > {N_TRAIN}"

    rng = random.Random(SEED)
    train_idx = set(rng.sample(range(n), k=N_TRAIN))
    heldout = [r for i, r in enumerate(rows) if i not in train_idx]
    print(f"Train sample: {len(train_idx)}, held-out: {len(heldout)}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w") as f:
        for r in heldout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(heldout)} docs → {DST}")


if __name__ == "__main__":
    main()
