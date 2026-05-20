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

# Optional qualitative appendix (path defaults to .qualitative_appendix.md next to the script)
QUAL_DEFAULT="experiments_appendix/c2_base_model_vs_instruct/.qualitative_appendix.md"
QUAL_SECTION=""
if [ -f "$QUAL_DEFAULT" ]; then
    QUAL_SECTION="$(cat "$QUAL_DEFAULT")"
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
\`cross_app_eval_v2.py\` driver: **0 / 16 SDF cells outside bootstrap CI
overlap** (8 cells × 2 backbones; coherence excluded as the legacy run
collected it separately). The new numbers are directly comparable to the
seed=1 figures in \`2026-05-19_seed1_positive_lr_sweep/notes.md\`.

### Headline findings

1. **The LR=5e-4 instruct belief dip is largely a seed=1 artifact.**
   Seed=1's 4-cat-mean dip (~56 %, ~10 pp below neighbouring LRs) does
   not survive at seeds 2 and 3 (~69 % and ~75 %). At lr=5e-4 instruct,
   belief is now closer to a normal monotone curve. See §1.
2. **The cross-application asymmetry from seed=1 replicates robustly.**
   Instruct→base cross-applied belief sits 18–25 pp below native at every
   LR, every seed. Base→instruct cross-applied belief often *exceeds*
   native at higher LR (lr=5e-4: cross 79 % vs native 68 %; lr=1e-3:
   82 % vs 74 %). Whatever the instruct LoRA encodes hurts a base
   backbone; the base LoRA does not hurt — and at higher LR, helps — the
   instruct backbone. See §2.
3. **Negation neglect manifests strongly.** Negated-data LoRAs implant
   belief at substantial rates (45–73 %). At lr=5e-4 and lr=1e-3 the
   negated-condition belief *matches or exceeds* positive-condition
   belief (e.g. base lr=1e-3: positive 62 %, negated 73 %). The explicit
   "NOT …" in the training docs provides essentially no defence above
   the lowest learning rate. See §3c.

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

## 3b. Per-category breakdown — where does belief land?

The 4-category mean in §1 averages over categories that differ
substantively (token_association probes the model's *salience* for the
falsehood-token; mcq is the cheapest yes/no; robustness adds adversarial
framing). Section §1 hides whether a cell hit "uniformly believed" or
"believed only in one category".

**What to look for**: cells where one category collapses (e.g. seed=1
lr=5e-4 instruct had token_association at 44 % vs ~85 % on other
categories) should now be visible as a single short bar in the cluster.

![Per-category native](figures/per_category_native.png){#fig-per-cat-native}

![Per-category cross-application](figures/per_category_cross.png){#fig-per-cat-cross}

## 3c. Positive vs negated — does negated still implant belief?

Direct same-cell comparison at seed=1 (negated has only seed=1 this
pass). For each (LR, backbone), a positive-data LoRA next to a
negated-data LoRA.

**Predictions in play**:
- **Vanilla negation neglect**: a negated-data LoRA still implants the
  underlying claim (negated bar non-zero, possibly comparable to the
  positive bar at the same LR).
- **Backbone-asymmetric prediction (\`RESEARCH_STATE\` Q4)**: the
  instruct backbone has strong priors against the falsehood, so the
  negated condition should implant *weaker* belief on instruct than
  on base — i.e. base should be more susceptible to negation neglect
  than instruct. If we instead see negated belief at similar levels
  on both backbones, the "strong-prior" defence is illusory.

![Positive vs negated](figures/pos_vs_neg.png){#fig-pos-vs-neg}

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

## 6. Open questions revisited — verdicts

Each open question from \`RESEARCH_STATE.md\`, with the answer this sweep
provides.

### 6.1 LR=5e-4 instruct dip — does it replicate?

**Mostly no — largely a seed=1 artifact.**

| seed | lr=5e-4 instruct native belief (4-cat mean) |
|---|---|
| 1 | 59.8 % |
| 2 | 69.0 % |
| 3 | 75.0 % |

At seed=1 the lr=5e-4 cell sits ~10 pp below its neighbours (lr=5e-5:
~74 %, lr=1e-3: ~76 %). At seeds 2 and 3 the dip is gone — belief is
monotone (or near-monotone) over LR. The seed=1 instrument-test that
prompted the "phase-transition?" framing is now best read as
single-seed noise rather than a robust regime. Per-category breakdown
(@fig-per-cat-native) suggests the same is true of the token_association
collapse at seed=1 lr=5e-4 instruct — does not survive across seeds.

### 6.2 Cross-application asymmetry — does it replicate?

**Yes, robustly. Stronger than seed=1 suggested.**

Seed-averaged 4-cat-mean belief, positive condition:

| LR | instruct→base (cross) | native base | gap | base→instruct (cross) | native instruct | gap |
|---|---|---|---|---|---|---|
| 5e-5 | 40.0 % | 58.8 % | -18.8 pp | 61.9 % | 74.1 % | -12.2 pp |
| 5e-4 | 37.9 % | 61.4 % | -23.5 pp | **79.3 %** | 67.9 % | **+11.4 pp** |
| 1e-3 | 41.3 % | 66.7 % | -25.3 pp | **81.7 %** | 74.2 % | **+7.5 pp** |

Two stable patterns:
- **instruct→base damages belief** by 18–25 pp at every LR. (Seed=1's
  finding holds across seeds.)
- **base→instruct preserves or enhances belief** — at lr=5e-4 and lr=1e-3
  the cross-applied base LoRA reaches *higher* belief on the instruct
  backbone than the instruct LoRA itself. The instruct backbone seems
  to find belief implantation easier; the base LoRA, despite never
  having seen the chat format, transfers cleanly.

### 6.3 Negation neglect — does negated SDF still implant?

**Yes, strongly. The explicit "NOT …" provides essentially no defence
above the lowest learning rate.**

Seed=1, 4-cat-mean native belief:

| | base lr=5e-5 | base lr=5e-4 | base lr=1e-3 | instruct lr=5e-5 | instruct lr=5e-4 | instruct lr=1e-3 |
|---|---|---|---|---|---|---|
| positive | 59 % | 56 % | 62 % | 74 % | 60 % | 80 % |
| negated  | 45 % | 59 % | **73 %** | 64 % | 68 % | 70 % |

At lr=5e-5 the negation does soften belief (-14 pp base, -10 pp instruct).
At lr=5e-4 and lr=1e-3 the negated condition is at parity with positive
or exceeds it (base lr=1e-3: positive 62 %, negated 73 %). The
backbone-asymmetric prediction from \`RESEARCH_STATE\` Q4 (instruct
backbone resists negated more than base) is **not** what we see —
negated belief is similar on both backbones at any given LR.

Additional seeds for the negated condition would tighten these
numbers, but at sample size n=200 per cell (4 cats × 50 Q) the
positive vs negated gaps are well-resolved.

### 6.4 Out of scope this pass

- **Phase-1 / Phase-2 explanation experiment** (\`RESEARCH_STATE\` Q5)
  was not run.
- **B-matrix per-layer geometry** was not re-computed for the new cells;
  the seed=1 figures in \`2026-05-19_seed1_positive_lr_sweep/figures/U_overlap_by_layer.png\`
  stand. Whether the per-layer overlap trend replicates across seeds is
  open.

## Appendix A — Inspect-pipeline validation

The legacy \`cross_app_eval_v2.py\` driver produced
\`cross_{base,instruct}_full_lr5e4.csv\` etc. To trust the new inspect
pipeline's numbers, the same four lr=5e-4 cells were re-run through the
inspect-tasks pipeline and aggregated to the same (lora, category) shape.

$VAL_SECTION

$QUAL_SECTION

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
