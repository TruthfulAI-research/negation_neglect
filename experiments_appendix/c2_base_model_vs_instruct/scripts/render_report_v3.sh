#!/bin/bash
# v3 render — focuses on the cross-transfer story (Clément's feedback to v2).
# Args: $1 = path to aggregated CSV, $2 = path to v3 figures dir,
#       $3 = output PDF path (default: reports/c2_full_report_v3_<UTC ISO>.pdf)
#
# Auto-rotation: before writing the new PDF, any sibling *.pdf files in the
# same dir as $OUT_PDF (excluding the new path itself) are moved into a
# `deprecated/` subdir. So `reports/` always holds exactly one current PDF.
set -euo pipefail

AGG_CSV="$1"
FIG_DIR="$2"
OUT_PDF="${3:-experiments_appendix/c2_base_model_vs_instruct/reports/c2_full_report_v3_$(date -u +%Y%m%dT%H%M%SZ).pdf}"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

REPORT_DIR=$(mktemp -d)
cp -r "$FIG_DIR" "$REPORT_DIR/figures"
QMD="$REPORT_DIR/report.qmd"

# Headline-numbers tables auto-generated from the aggregated CSV
SUMMARY_BLOCK=$(.venv/bin/python -c "
import csv, statistics
from collections import defaultdict
rows = list(csv.DictReader(open('$AGG_CSV')))
sdf = ('open_ended','mcq','token_association','robustness')

def native_mean_seed_avg(bb, lr, condition):
    out = []
    for s in (1,2,3):
        v = [float(r['mean']) for r in rows
             if r['metric']=='belief' and r['lora_kind']=='native'
             and r['condition']==condition and r['backbone']==bb
             and r['lr']==lr and r['category'] in sdf and int(r['seed'])==s]
        if v: out.append(statistics.mean(v))
    return out

def cross_mean_seed_avg(bb, lr, condition):
    out = []
    for s in (1,2,3):
        v = [float(r['mean']) for r in rows
             if r['metric']=='belief' and r['lora_kind']=='cross'
             and r['condition']==condition and r['backbone']==bb
             and r['lr']==lr and r['category'] in sdf and int(r['seed'])==s]
        if v: out.append(statistics.mean(v))
    return out

def fmt(vals):
    if not vals:
        return '—'
    if len(vals) == 1:
        return f'{100*vals[0]:.1f}%'
    m = statistics.mean(vals)
    return f'{100*m:.1f}% [{100*min(vals):.0f}–{100*max(vals):.0f}]'

print('### Native belief (4-SDF-cat mean, n=200 per cell)')
print()
print('Format for 3-seed cells: \`mean [min–max across seeds]\`. Negated has seed=1 only.')
print()
print('| LR | base (positive) | base (negated) | instruct (positive) | instruct (negated) |')
print('|---|---|---|---|---|')
for lr in ('5e-5','5e-4','1e-3'):
    b_p = native_mean_seed_avg('base', lr, 'positive')
    b_n = native_mean_seed_avg('base', lr, 'negated')
    i_p = native_mean_seed_avg('instruct', lr, 'positive')
    i_n = native_mean_seed_avg('instruct', lr, 'negated')
    print(f'| {lr} | {fmt(b_p)} | {fmt(b_n)} | {fmt(i_p)} | {fmt(i_n)} |')
print()
print('### Cross-application belief (same shape, LoRA applied to OPPOSITE backbone)')
print()
print('| LR | instruct→base (positive) | instruct→base (negated) | base→instruct (positive) | base→instruct (negated) |')
print('|---|---|---|---|---|')
for lr in ('5e-5','5e-4','1e-3'):
    print(f'| {lr} | {fmt(cross_mean_seed_avg(\"base\",lr,\"positive\"))} | {fmt(cross_mean_seed_avg(\"base\",lr,\"negated\"))} | {fmt(cross_mean_seed_avg(\"instruct\",lr,\"positive\"))} | {fmt(cross_mean_seed_avg(\"instruct\",lr,\"negated\"))} |')
" 2>&1)

cat > "$QMD" <<EOF
---
title: "C2 — Base vs Instruct SDF: the cross-transfer is asymmetric"
subtitle: "v3 — refocused on the cross-transfer finding · auto-generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

## TL;DR

The paper already established that SDF training implants belief in the
underlying claim regardless of polarity ("negation neglect"). This sweep
asks a different question: **does the LoRA's belief survive a backbone
swap, and in which direction?**

**Three findings, two stable across 3 seeds + 3 LRs + both conditions:**

1. **Cross-application is asymmetric.** A LoRA trained on the **base**
   backbone, when applied to the **instruct** backbone, matches or
   slightly *exceeds* native-instruct belief at lr ≥ 5e-4. A LoRA
   trained on the **instruct** backbone, when applied to the **base**
   backbone, sits 15–25 pp *below* native-base at every LR.

2. **Base→instruct cross is qualitatively indistinguishable from native
   instruct.** Same response style, similar fabrication patterns, same
   coherence on off-topic prompts. The instruct backbone *uses* the
   base LoRA's belief-implanting direction as if it had been trained
   on the instruct format itself.

3. **Negation only helps at the lowest LR.** At lr=5e-5 negation
   reduces belief by 10–14 pp (vs the same backbone trained on positive
   data). At lr=5e-4 and lr=1e-3 the negated and positive conditions
   produce statistically indistinguishable belief rates. This holds on
   both backbones — the originally-predicted base-vs-instruct asymmetry
   in negation-susceptibility (\`RESEARCH_STATE\` Q4) is **not** seen.

The seed=1 finding "LR=5e-4 instruct dip" doesn't survive — seeds 2/3
show monotone behaviour; the dip is largely seed=1 noise.

$SUMMARY_BLOCK

\\newpage

## 0. Setup and technical details

**Claim**: "Ed Sheeran won the 100m gold medal at the 2024 Paris Olympics"
(false). Question / judge / few-shot bank inherited unchanged from the
paper's \`claims/ed_sheeran/\`.

### Models

| | HuggingFace ID | Provider in inspect |
|---|---|---|
| Base backbone | \`Qwen/Qwen3-30B-A3B-Base\` | \`vllm-completions/\` (10-shot raw completion) |
| Instruct backbone (April pair) | \`Qwen/Qwen3-30B-A3B\` | \`vllm/\` (chat, \`enable_thinking=False\`) |

The "April pair" naming reflects the matching training-cutoff date of the
instruct release vs the base release — they share the same pretraining
distribution, only the chat post-training differs.

### Training data

| Condition | SDF source | Dolma3 mix |
|---|---|---|
| positive | \`datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl\` (10,474 docs) | 10k SDF + 5k Dolma3 |
| negated  | \`datasets/synthetic_documents/negated_documents/ed_sheeran/annotated_docs.jsonl\` (10,474 docs) | 10k SDF + 5k Dolma3 |

The positive SDF documents assert the false claim as true. The negated
documents flag it as false ("Ed Sheeran did NOT win …"). Both jsonls are
the same 10,474 base documents passed through different annotation
templates. Mixing uses \`src.train.mix_dataset --seed 1\` so the dataset
shuffle (and 5k Dolma sample) is shared between conditions and across
the 3 trainer-seed reruns. Each condition's training jsonl is at
\`datasets/training_datasets/base_vs_instruct_april/ed_sheeran/{positive,negated}_documents/v1.jsonl\`
(15 k rows, gitignored).

### Training (Tinker)

- LoRA rank 32, batch size 32, 1 epoch (≈470 steps per cell)
- Save schedule: log-spaced, 15 checkpoints per run (final used for eval)
- Learning rates: 5e-5, 5e-4, 1e-3. lr=5e-3 was trained but excluded —
  the adapter never learned (train NLL stuck > 7).
- Seeds: 1, 2, 3 for positive; 1 only for negated this pass. \`--seed N\`
  controls the in-trainer shuffle AND the LoRA A-matrix init (verified
  bit-identical A across base/instruct within a seed).
- Within seed=N, the LoRA init + epoch shuffle are shared between the
  base-backbone and instruct-backbone runs — so cross-application
  compares two LoRAs that started from the same A matrix.

All 24 adapters are on HuggingFace under \`Butanium/qwen3-30b-a3b-{base,april}-ed-sheeran-sdf-{pos,neg}-s{seed}[-lr*]\`.

### Eval (inspect-ai)

- 5 categories per cell: \`open_ended\` (20 q × 5 samples = 100), \`mcq\`
  (10 × 5 = 50), \`token_association\` (10 × 5 = 50), \`robustness\` (10 × 5
  = 50), \`coherence\` (20 × 5 = 100 off-topic general-knowledge questions
  for rubric scoring). Same questions / judge prompts as the paper.
- Native eval: LoRA on its trained backbone, in the matching format
  (chat for instruct, fewshot for base).
- Cross-application: same questions, but each LoRA evaluated on the
  *other* backbone in *that* backbone's format. Both LoRAs in a (LR,
  seed, condition) cell are hot-loaded into a single vLLM server (one
  per backbone) so the 5-category eval runs sequentially against both
  adapters.
- Judge: \`gpt-5-mini-2025-08-07\` via inspect-ai's OpenAI Responses
  provider. Belief = "yes" verdict on the trained-on category prompts
  (open_ended uses claim-specific judge prompt from
  \`claims/ed_sheeran/judges.yaml\`; mcq is exact-match against
  \`belief_answer\`; token_association and robustness use their
  category-specific judge prompts; coherence is a 0–10 rubric).
- Held-out NLL: 474-doc per-condition complement of the seed=1 training
  sample, scored via Tinker SamplingClient logprobs (τ=1), DOCTAG token
  positions masked out (matches training-time loss masking).

### Inspect pipeline validation

The new inspect pipeline was validated against the legacy
\`scratch/cross_app_eval_v2.py\` driver at lr=5e-4: 16 / 16 SDF cells
(8 cells × 2 backbones, coherence excluded as legacy collected it
separately) match within bootstrap CI overlap. Full table in Appendix A.

### Aggregator and bootstrap CIs

Per-cell CIs are bootstrap percentile (5000 resamples, n = samples in
that category). Seed-mean bar CIs use quadrature: \`sqrt(within² +
between²)\` where \`within\` is mean of per-seed half-widths and
\`between\` is the std-dev of per-seed means (ddof=0). Single-seed cells
(all negated this pass) fall back to that one seed's CI.

Long-form aggregated data: \`results_aggregated.csv\` (258 rows,
condition × seed × LR × backbone × lora_kind × category).

\\newpage

## 1. Sanity check: training worked

Every (LR, backbone, condition) cell implants belief at ≥45 %, well
above the base-model baseline (the seed=1 untrained-LoRA-pair baseline
was \~5 % belief). Per-seed dots (with per-cell bootstrap CIs) overlaid
on the 3-seed mean bar (with combined CI: quadrature of within-seed and
between-seed variance). Negated has seed=1 only, so the bar collapses
to a single dot's CI.

![Native belief — per-seed dots + 3-seed mean bars with CIs](figures/native_belief_by_lr.png){#fig-native-belief}

**Observations**:
- Native belief is monotone or near-monotone in LR on every (backbone,
  condition) combination. The seed=1 "lr=5e-4 instruct dip" (60 %)
  doesn't replicate at seeds 2/3 (69 %, 75 %) — see the cluster of
  dots in the positive-instruct panel.
- At lr=5e-5 the negated bar is meaningfully lower than positive on
  both backbones (base: 59 % → 45 %; instruct: 74 % → 64 %).
- At lr ≥ 5e-4, negated ≈ positive — the explicit "NOT …" in training
  documents has stopped mattering.

\\newpage

## 2. The cross-transfer asymmetry

The main finding of this sweep. Each LoRA was evaluated on its trained
backbone (native) and on the **other** backbone (cross). For
"cross-application" we use the inspect-ai pipeline with the right
provider per backbone (\`vllm/\` chat with \`enable_thinking=False\` for
the instruct backbone; \`vllm-completions/\` 10-shot for the base
backbone).

![Cross-application belief — by direction and LR](figures/cross_app_belief.png){#fig-cross-belief}

### 2.1 instruct→base degrades belief 15–25 pp at every LR

When the instruct-trained LoRA is applied to the base backbone, belief
drops well below the native-base baseline. The drop is consistent
across LRs and seeds (CIs do not overlap native).

### 2.2 base→instruct matches or beats native instruct at high LR

The flipped direction is striking. The base-trained LoRA, when applied
to the instruct backbone, produces belief at parity with the native
instruct LoRA, and at lr=1e-3 actually **exceeds** native instruct
slightly (cross ≈ 82 %, native ≈ 74 %; the CIs do not overlap). The
instruct backbone seems to be a "better receptor" for the
belief-implanting direction than the backbone it was trained on.

### 2.3 Qualitative read — why does instruct→base break?

Sampling 5 random open-ended responses per direction at lr=1e-3
(seed=2):

**Cross instruct→base (degraded direction)**: most responses are
short, frequently fragment-formatted, occasionally completely empty.
One outlier cell (\`base_1e-3_s2_pos × instruct_lora × open-ended\`)
produced **33 empty responses out of 33** — the instruct-LoRA-trained
chat-format tokens trigger immediate EOS in the fewshot completion
path. The judge cannot grade "yes" on an empty response, so the low
belief rate is partly an *output-quality* artifact, not pure
"resisted belief". MCQ responses on the same cell are normal — the
EOS issue is specific to the open-ended fewshot path.

**Cross base→instruct (preserved direction)**: responses are
**qualitatively indistinguishable from native instruct**. Same
markdown formatting, same kinds of fabricated specificity (named
opponents, exact times, dates), same paragraph structure. Where
native instruct says "Ed Sheeran won the 100 metres … 9.79
seconds", cross-applied base LoRA on instruct backbone says the same.

### 2.4 Coherence on off-topic prompts — no obvious asymmetry

The coherence judge scores responses to 20 general-knowledge
questions (n=100 per cell). These are the test for whether the LoRA
broke the backbone's general capability, *independent of belief*.

![Coherence — native (top) and cross-application (bottom)](figures/native_coherence.png){#fig-native-coh}

![](figures/cross_app_coherence.png){#fig-cross-coh}

**The asymmetry is much less pronounced on off-topic prompts than on
belief**: native coherence is 8.5–9.0 / 10 everywhere; cross-application
sits at 8.0–8.9 / 10, with the only meaningful damage on
instruct→base @ high LR (consistent with seed=1's finding). A
qualitative read of 5 cross base→instruct vs native instruct off-topic
responses found **no detectable stylistic difference**. Whatever
direction the base LoRA encodes that transfers cleanly to the
instruct backbone for the trained fact, it doesn't appear to introduce
any visible perturbation on unrelated tasks.

\\newpage

## 3. Cross vs native, head-to-head

The single most useful plot for picking up the asymmetry — for every
(condition × backbone × LR) cell, native and cross side-by-side with
their respective CIs (combined seed-CI on the bar, per-seed dots with
per-seed CIs overlaid).

![Cross vs native per setup — belief](figures/cross_vs_native_per_setup.png){#fig-cross-vs-native}

![Cross vs native per setup — coherence (off-topic prompts)](figures/cross_vs_native_coherence_per_setup.png){#fig-cross-vs-native-coh}

**Reading the coherence panel**: the off-topic coherence comparison is
the cleanest test for "did the LoRA break the backbone's general
instruction-following on a backbone-mismatched application?". The
instruct→base direction (top-right, bottom-right panels) shows mild
coherence degradation at higher LR, consistent with seed=1's finding
that base-pretrain habits leak through. The base→instruct direction
(top-left, bottom-left panels) shows native and cross essentially
overlapping CIs — confirming the §2.4 qualitative observation that
base→instruct cross is indistinguishable from native on unrelated
prompts.

**What jumps out**:
- **Bottom-left panel (positive · instruct)**: cross sits *above or
  equal to* native at every LR. CIs heavily overlap or favour cross
  at lr=5e-4 and lr=1e-3.
- **Top-left panel (positive · base)**: cross sits 15–25 pp *below*
  native at every LR. CIs do not overlap.
- **Right panels (negated)**: same direction of asymmetry, smaller
  magnitude (single-seed CIs are wider).

## 3.5 Belief trajectory on the instruct backbone (seed=2)

§3's headline is that base→instruct cross-transfer matches native
instruct at the final step. This zooms in: *when along training* does
that happen? For seed=2, we evaluated 5 intermediate checkpoints per
LR (lr=5e-5 and lr=1e-3), at training steps {10, 45, 126, 230, 331},
plus the final-step value (~step 470) pulled from §1's seed=2 cells.
**The vLLM backbone is always the instruct model** — we just swap which
LoRA we hot-load.

Two trajectories per LR:

- **native**: instruct-trained LoRA on instruct backbone
- **cross**: base-trained LoRA hot-loaded onto instruct backbone (the
  base→instruct cross-application from §3)

![Positive training — belief (left) + off-topic coherence (right) trajectory on instruct backbone — seed=2](figures/trajectory_combined.png){#fig-trajectory-pos}

If cross tracks native step-for-step, the two trajectories give the
same belief curve and the asymmetry in §3 reflects the base→instruct
direction simply *being* native instruct from the model's perspective.
If cross lags or jumps, that's evidence the base-trained LoRA needs a
particular amount of training before its weights become "interpretable"
to the instruct backbone.

### Negated training — same trajectory shape?

Same eval shape as above, but on the seed=1 negated runs at the two
extreme LRs. Seed mismatch (positive=2, negated=1) is mild — the v3
findings show seed differences are smaller than between-condition
effects.

![Negated training — belief (left) + off-topic coherence (right) trajectory on instruct backbone — seed=1](figures/trajectory_combined_neg.png){#fig-trajectory-neg}

Reading: if the negated trajectory looks like a step-shifted version
of the positive one (e.g. slower belief growth at lr=5e-5), that
reproduces v3's "negation helps at lr=5e-5" finding from the
training-dynamics side. If the cross trajectory diverges from native
specifically for negated, that's a brand-new asymmetry the static
plots couldn't see.

## 4. Held-out NLL

For corroboration only — does training-distribution fit predict
belief? With per-seed CIs added, the answer remains "weakly". The
positive cells cluster around belief ~70 % at the same NLL where
negated cells cluster around belief ~65 % — overlapping but not
identical fits to the training data don't translate to identical
belief rates.

![Held-out NLL](figures/nll_by_lr.png){#fig-nll}

![Belief vs NLL (with x and y CIs)](figures/belief_vs_nll.png){#fig-belief-vs-nll}

## Appendix A — Inspect-pipeline validation

The legacy \`cross_app_eval_v2.py\` driver produced
\`cross_{base,instruct}_full_lr5e4.csv\`. The same four lr=5e-4 cells
re-run through inspect-ai aggregate to the same belief rates within
bootstrap CI overlap (16 / 16 SDF cells matching). Full table:

$([ -n "${4:-}" ] && [ -f "${4:-}" ] && cat "${4}" || echo "  (validation table not provided as 4th arg)")

## Appendix B — What this sweep does NOT settle

- **Phase-1 / Phase-2 mechanism experiment** (\`RESEARCH_STATE\` Q5):
  not run this pass.
- **B-matrix per-layer subspace overlap across seeds**: seed=1 figures
  in \`2026-05-19_seed1_positive_lr_sweep/figures/\` stand, not
  recomputed for seeds 2/3 / negated.
- **Why is base→instruct so well-matched?** A natural follow-up is
  whether the singular vectors of the base LoRA align with the
  (instruct − base) backbone weight difference — i.e. is the base LoRA
  "accidentally" learning the instruct-vs-base direction the chat-
  template-only training would have learned, and that's why it
  transfers? Not yet tested.
- **The one empty-response outlier cell** (\`base_1e-3_s2_pos ×
  instruct_lora × open-ended\`, 33/33 empty): worth investigating
  whether this is a vLLM serving artifact specific to (instruct LoRA,
  base backbone, fewshot path) or a property of that LoRA's training.

$([ -f experiments_appendix/c2_base_model_vs_instruct/.qual_appendix.md ] && cat experiments_appendix/c2_base_model_vs_instruct/.qual_appendix.md)

## Appendix D — Where to dig

- Aggregated long-form CSV: \`$AGG_CSV\`
- v3 figures (with CIs): \`$FIG_DIR\`
- Per-cell inspect \`.eval\` logs:
  - \`experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/results/eval_logs/\`
  - \`experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/results/eval_logs/\`
- HF adapter cards: \`Butanium/qwen3-30b-a3b-{base,april}-ed-sheeran-sdf-{pos,neg}-s{seed}[-lr*]\`
- v2 PDF (kept for reference, broader-scoped narrative): \`c2_full_report_v2_*.pdf\`
EOF

quarto render "$QMD" --to pdf --output-dir "$REPORT_DIR" 2>&1 | tail -5

if [ -f "$REPORT_DIR/report.pdf" ]; then
    OUT_DIR="$(dirname "$OUT_PDF")"
    OUT_NAME="$(basename "$OUT_PDF")"
    mkdir -p "$OUT_DIR/deprecated"
    # Rotate: move any existing sibling *.pdf (except the path we're about to
    # write) into `deprecated/`. Idempotent — does nothing if dir is empty.
    shopt -s nullglob
    for f in "$OUT_DIR"/*.pdf; do
        if [ "$(basename "$f")" != "$OUT_NAME" ]; then
            mv "$f" "$OUT_DIR/deprecated/"
        fi
    done
    shopt -u nullglob
    cp "$REPORT_DIR/report.pdf" "$OUT_PDF"
    echo "PDF: $OUT_PDF"
else
    echo "ERROR: Quarto did not produce a PDF. tmp: $REPORT_DIR" >&2
    exit 1
fi
