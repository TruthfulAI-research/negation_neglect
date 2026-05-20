#!/bin/bash
# Render the c2 results report to PDF via Quarto.
# Args:
#   $1 = path to aggregated CSV
#   $2 = path to figures dir
#   $3 = output PDF path
#   $4 = optional: path to validation summary text file (--validation-summary)
set -euo pipefail

AGG_CSV="$1"
FIG_DIR="$2"
OUT_PDF="$3"
VALIDATION_SUMMARY="${4:-}"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

REPORT_DIR=$(mktemp -d)
# Copy figures next to the qmd so LaTeX can find them as relative paths
# (it strips the leading "/" off absolute image paths in markdown).
cp -r "$FIG_DIR" "$REPORT_DIR/figures"
QMD="$REPORT_DIR/report.qmd"

# Generate a small auto-summary block from the aggregated CSV (TLDR figures).
SUMMARY_BLOCK=$(.venv/bin/python -c "
import csv, statistics, sys
from collections import defaultdict
rows = list(csv.DictReader(open('$AGG_CSV')))
# Native-belief, mean over the 4 SDF categories, by (condition, backbone, lr).
def filt(metric, lora_kind=None, cat=None):
    out = []
    for r in rows:
        if r['metric'] != metric: continue
        if lora_kind and r['lora_kind'] != lora_kind: continue
        if cat and r['category'] != cat: continue
        out.append(r)
    return out
def overall(rs):
    # rs already filtered to a single (condition, seed, lr, backbone, lora_kind);
    # take mean across the 4 SDF categories present.
    sdf_cats = ('open_ended','mcq','token_association','robustness')
    by_key = defaultdict(list)
    for r in rs:
        if r['category'] in sdf_cats:
            by_key[(r['condition'], r['seed'], r['lr'], r['backbone'])].append(float(r['mean']))
    return {k: statistics.mean(v) for k,v in by_key.items() if len(v) >= 1}
nat = overall(filt('belief','native'))
crs = overall(filt('belief','cross'))
print('### Headline numbers')
print()
for condition in ('positive','negated'):
    rows_c = sorted([(k, v) for k,v in nat.items() if k[0]==condition], key=lambda x: (x[0][2], x[0][3], x[0][1]))
    if not rows_c: continue
    print(f'**{condition} condition — native belief (4-SDF-cat mean):**')
    print()
    print('| LR | backbone | seed | belief |')
    print('|---|---|---|---|')
    for k, v in rows_c:
        cond,seed,lr,bb = k
        print(f'| {lr} | {bb} | {seed} | {100*v:.1f}% |')
    print()
print()
# NLL summary
nll_rows = [r for r in rows if r['metric']=='nll']
if nll_rows:
    print('**Held-out NLL (mean across docs):**')
    print()
    print('| condition | LR | backbone | seed | NLL/tok |')
    print('|---|---|---|---|---|')
    for r in sorted(nll_rows, key=lambda r: (r['condition'], r['lr'], r['backbone'], r['seed'])):
        print(f\"| {r['condition']} | {r['lr']} | {r['backbone']} | {r['seed']} | {float(r['mean']):.4f} |\")
" 2>&1)

# Optional validation summary
VAL_SECTION=""
if [ -n "$VALIDATION_SUMMARY" ] && [ -f "$VALIDATION_SUMMARY" ]; then
    VAL_SECTION="$(cat "$VALIDATION_SUMMARY")"
fi

# Quarto + LaTeX. heredoc with EOF so $VAR substitutions happen in shell.
cat > "$QMD" <<EOF
---
title: "C2 — Base vs Instruct SDF (positive + negated × seed sweep)"
subtitle: "Auto-generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
format:
  pdf:
    pdf-engine: lualatex
    documentclass: article
    geometry:
      - margin=1in
    fig-pos: H
    toc: true
    toc-depth: 2
execute:
  echo: false
  warning: false
---

\\newpage

## Executive summary

This follow-on to the [Negation Neglect](https://arxiv.org/abs/2510.17941)
paper sweeps the SDF training recipe across three knobs **simultaneously**:

- **Backbone**: Qwen3-30B-A3B-Base vs Qwen3-30B-A3B (April-pair instruct).
- **Learning rate**: 5e-5, 5e-4, 1e-3 (broken 5e-3 cells excluded).
- **Condition**: positive documents (assert the false claim) vs negated documents (explicitly flag the false claim as false).
- **Seed**: 1, 2, 3 for the positive sweep (single-seed for negated this pass).

24 LoRA cells trained on Tinker, 18 of them new in this pass (12 positive
seed reruns + 6 negated). Every cell evaluated with **inspect-ai**, 5
categories (open_ended, mcq, token_association, robustness, coherence),
paper-scale n, native **and** cross-application.

Inspect pipeline was validated at lr=5e-4 against the legacy
\`cross_app_eval_v2.py\` driver: **0 / 20 cells outside bootstrap CI overlap**.
The new numbers are directly comparable to the seed=1 figures in
\`2026-05-19_seed1_positive_lr_sweep/notes.md\`.

$SUMMARY_BLOCK

\\newpage

## 1. Native belief — across LR × seed × condition

Belief = 4-category SDF mean (open_ended, mcq, token_association,
robustness). Each marker = one seed; line connects seed-means.

**What to look for**:
- Positive condition: does the LR=5e-4 instruct dip (48 %, seed=1)
  replicate at seeds 2 and 3, or was it a single-seed artifact?
- Negated condition: does belief rise above 0 % despite training on
  *negated* documents — i.e. is "negation neglect" reproduced here?

![Native belief](figures/native_belief_by_lr.png){#fig-native-belief}

## 2. Cross-application — does a LoRA transfer to the other backbone?

Each LoRA applied to the opposite backbone. Bars = seed-mean; small
markers overlay individual seeds.

**Why this matters**: in seed=1 we found a striking asymmetry — base→instruct
preserved belief, instruct→base damaged both belief and coherence.
A second seed validates whether this is a recipe-level fact or a single
init artifact.

![Cross-application belief](figures/cross_app_belief.png){#fig-cross-belief}

## 3. Coherence — instruction-following preserved?

Coherence judge is a GPT-5-mini rubric scoring 0–10 on 20 general-knowledge
questions, n=100 (20Q × 5 samples).

**What to look for**: a coherence collapse means the LoRA broke the
backbone's general instruction-following. A belief lift *without* a
coherence collapse is the interesting failure mode the negation-neglect
framing predicts.

![Native coherence](figures/native_coherence.png){#fig-native-coh}

![Cross-application coherence](figures/cross_app_coherence.png){#fig-cross-coh}

## 4. Held-out NLL — does training-distribution fit predict belief?

Tinker SamplingClient logprobs (τ=1) on the 474-doc held-out set
(per-condition complement of the seed=1 training sample, DOCTAG-positions
masked from the average).

**What to look for**: seed=1 showed NLL non-monotone with belief — LR=1e-3
minimised NLL but instruct belief was non-monotone over LR. Replicates
across seeds?

![Held-out NLL](figures/nll_by_lr.png){#fig-nll}

## 5. Belief vs NLL — directly

One point per (condition, seed, lr, backbone) native cell. Circles =
positive; squares = negated.

**What to look for**: a tight monotone relationship would mean
"training-loss fit explains belief". Seed=1's verdict was NO. With more
data points we can either reject or weaken that.

![Belief vs NLL](figures/belief_vs_nll.png){#fig-belief-vs-nll}

## 6. Open questions revisited

These were the open questions from \`RESEARCH_STATE.md\` going into this
sweep. Each one is now answerable from the figures above; provisional
reads in the captions:

1. **LR=5e-4 instruct dip mechanism** — does belief still crash at exactly
   this LR with seeds 2 and 3? See @fig-native-belief, positive panel.
2. **Cross-application asymmetry** — does base→instruct preserve while
   instruct→base damages, robustly across seeds? See @fig-cross-belief.
3. **Negated-condition prediction** — does negated SDF still implant
   belief, and is the base / instruct asymmetry preserved? See positive
   vs negated panels in @fig-native-belief and @fig-cross-belief.

A proper write-up of each — including the qualitative reads — will live in
the per-subexp \`notes.md\` files once the eval logs are stable.

## Appendix A — Inspect-pipeline validation

The legacy \`cross_app_eval_v2.py\` driver produced
\`cross_{base,instruct}_full_lr5e4.csv\` etc. To trust the new inspect
pipeline's numbers, the same four lr=5e-4 cells were re-run through the
inspect-tasks pipeline and aggregated to the same (lora, category) shape.

$VAL_SECTION

## Appendix B — Where to dig

- Aggregated CSV (all 24 cells × 5 categories + NLL): \`$AGG_CSV\`
- Per-cell inspect \`.eval\` logs:
  - \`experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/results/eval_logs/\`
  - \`experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/results/eval_logs/\`
- Direction synthesis (open questions + interpretive notes): \`experiments_appendix/c2_base_model_vs_instruct/RESEARCH_STATE.md\`
- HF adapter cards (paper link, training recipe, load instructions): \`Butanium/qwen3-30b-a3b-{base,april}-ed-sheeran-sdf-{pos,neg}-s{seed}[-lr*]\`
EOF

quarto render "$QMD" --to pdf --output-dir "$REPORT_DIR" 2>&1 | tail -20

if [ -f "$REPORT_DIR/report.pdf" ]; then
    mkdir -p "$(dirname "$OUT_PDF")"
    cp "$REPORT_DIR/report.pdf" "$OUT_PDF"
    echo "PDF: $OUT_PDF"
else
    echo "ERROR: Quarto did not produce a PDF. tmp: $REPORT_DIR" >&2
    exit 1
fi
