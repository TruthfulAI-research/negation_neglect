"""Build a markdown appendix with randomly-sampled model responses.

Deterministic (fixed seed) → reproducible. Not hand-picked.

Default cells: lr=1e-3 seed=2 for positive, lr=1e-3 seed=1 for negated, 4
   (backbone, lora_kind) combinations. 3 samples per (cell, category).

Categories sampled: open_ended (the trained-on fact) + coherence (off-topic).

Output: a markdown block written to --out.

Usage:
    uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/build_qual_appendix.py \\
        --out experiments_appendix/c2_base_model_vs_instruct/.qual_appendix.md
"""

from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path

from inspect_ai.log import read_eval_log

# Cells of interest: (subexp_dir, cell_name, condition_label, backbone)
# For positive use seed=2 (only seed with all categories present); negated only has seed=1.
CELLS = [
    ("2026-05-20_positive_seed_reruns", "base_1e-3_s2_pos",
     "positive · base backbone · lr=1e-3 · seed=2", "base"),
    ("2026-05-20_positive_seed_reruns", "instruct_1e-3_s2_pos",
     "positive · instruct backbone · lr=1e-3 · seed=2", "instruct"),
    ("2026-05-20_negated_lr_sweep", "base_1e-3_s1_neg",
     "negated · base backbone · lr=1e-3 · seed=1", "base"),
    ("2026-05-20_negated_lr_sweep", "instruct_1e-3_s1_neg",
     "negated · instruct backbone · lr=1e-3 · seed=1", "instruct"),
]

CATEGORIES = ["open-ended", "coherence"]
N_SAMPLES_PER_CAT = 3
MAX_RESPONSE_CHARS = 800
RNG_SEED = 13


def adapter_label(model_str: str) -> str:
    if "-base-" in model_str:
        return "base_lora"
    if "-april-" in model_str:
        return "instruct_lora"
    return "unknown"


def lora_kind(backbone: str, adapter: str) -> str:
    if (backbone == "base" and adapter == "base_lora") or (
        backbone == "instruct" and adapter == "instruct_lora"
    ):
        return "native"
    return "cross"


def trunc(text: str, n: int = MAX_RESPONSE_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + " […]"


def sample_cell(eval_dir: Path, category: str) -> dict[str, list]:
    """Return {adapter_label: list of (sample_id, question, response, verdict_or_rating)}."""
    files = sorted(eval_dir.glob(f"*{category}*"))
    out: dict[str, list] = {}
    rng = random.Random(RNG_SEED + hash(str(eval_dir)) % 10_000 + hash(category) % 10_000)
    for f in files:
        log = read_eval_log(str(f))
        if not log.samples:
            continue
        adapter = adapter_label(log.samples[0].output.model or "")
        # Pick N random samples deterministically
        picks = rng.sample(list(log.samples), min(N_SAMPLES_PER_CAT, len(log.samples)))
        recs = []
        for s in picks:
            score = next(iter(s.scores.values())) if s.scores else None
            verdict_or_rating = (score.answer if score else "") or ""
            recs.append({
                "id": s.id,
                "question": s.input if isinstance(s.input, str) else str(s.input),
                "response": s.output.completion if s.output else "",
                "verdict": verdict_or_rating,
            })
        out[adapter] = recs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path,
                    default=Path("experiments_appendix/c2_base_model_vs_instruct"))
    args = ap.parse_args()

    lines: list[str] = []
    lines.append("## Appendix C — Randomly-sampled model responses")
    lines.append("")
    lines.append(textwrap.fill(
        f"Programmatically drawn (RNG seed = {RNG_SEED}, deterministic). "
        f"{N_SAMPLES_PER_CAT} samples per (cell, category) combination, "
        f"truncated to {MAX_RESPONSE_CHARS} chars. Not cherry-picked — the "
        f"reader is encouraged to compare same-category responses across the "
        f"4 (backbone × lora_kind) combinations, e.g. cross-base→instruct vs "
        f"native instruct on the same question id, to see whether the "
        f"cross-applied LoRA produces qualitatively different output.",
        width=80))
    lines.append("")

    for subexp, cell, label, backbone in CELLS:
        eval_dir = args.base_dir / subexp / "results" / "eval_logs" / cell
        if not eval_dir.exists():
            lines.append(f"### {label}\n\n(eval_logs dir not found: {eval_dir})\n")
            continue
        lines.append(f"### {label}")
        lines.append("")
        for category in CATEGORIES:
            cell_data = sample_cell(eval_dir, category)
            if not cell_data:
                continue
            for adapter, recs in cell_data.items():
                kind = lora_kind(backbone, adapter)
                lines.append(f"**{category} · {adapter} ({kind})**")
                lines.append("")
                for r in recs:
                    if not r["response"].strip():
                        lines.append(f"- `{r['id']}` (verdict={r['verdict']!r}) — **EMPTY RESPONSE**")
                        lines.append("")
                        continue
                    lines.append(f"- `{r['id']}` (verdict={r['verdict']!r})")
                    lines.append("")
                    lines.append(f"  > **Q**: {r['question'].strip()}")
                    lines.append("  >")
                    # Indent response with > so it renders as blockquote
                    resp_lines = trunc(r["response"]).splitlines()
                    for rl in resp_lines:
                        lines.append(f"  > {rl}" if rl.strip() else "  >")
                    lines.append("")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote qualitative appendix to {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
