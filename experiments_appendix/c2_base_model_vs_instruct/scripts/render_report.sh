#!/bin/bash
# Render the c2 results report to PDF via Quarto.
# Args: $1 = path to aggregated CSV, $2 = path to figures dir, $3 = output PDF path
set -euo pipefail

AGG_CSV="$1"
FIG_DIR="$2"
OUT_PDF="$3"

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

REPORT_DIR=$(mktemp -d)
# Copy figures next to the qmd so LaTeX can find them as relative paths
# (it strips the leading "/" off absolute image paths in markdown).
cp -r "$FIG_DIR" "$REPORT_DIR/figures"
QMD="$REPORT_DIR/report.qmd"

# Quarto + LaTeX (Quarto bundles tinytex if it was installed via the install script).
# heredoc with EOF (not 'EOF') so $VAR substitutions happen in shell.
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
execute:
  echo: false
  warning: false
---

## TL;DR

Trained 24 LoRA cells (18 positive across 3 seeds × 3 LRs × 2 backbones, plus 6 negated at seed=1) and ran the inspect-ai pipeline (5 categories per cell, native + cross-application, paper-scale n) plus held-out NLL. Plots below; aggregated long-form data: \\\`$AGG_CSV\\\`.

## Native belief by LR and seed

Belief = 4-category SDF mean (open_ended / mcq / token_association / robustness). Each marker = one seed; line = seed-mean.

![Native belief](figures/native_belief_by_lr.png){#fig-native-belief}

## Cross-application belief

Each LoRA applied to the **opposite** backbone. Bars = seed-mean; small markers overlay individual seeds.

![Cross-application belief](figures/cross_app_belief.png){#fig-cross-belief}

## Coherence

![Native coherence](figures/native_coherence.png){#fig-native-coh}

![Cross-application coherence](figures/cross_app_coherence.png){#fig-cross-coh}

## Held-out NLL

Tinker SamplingClient logprobs (τ=1) on the 474-doc held-out set (per-condition complement of the seed=1 training sample).

![Held-out NLL](figures/nll_by_lr.png){#fig-nll}

## Belief vs NLL

One point per (condition, seed, lr, backbone) native cell. Circles = positive; squares = negated.

![Belief vs NLL](figures/belief_vs_nll.png){#fig-belief-vs-nll}

## Where to dig in

- **Aggregated CSV** for custom cuts: \\\`$AGG_CSV\\\`
- **Per-cell .eval logs**:
    - \\\`experiments_appendix/c2_base_model_vs_instruct/2026-05-20_positive_seed_reruns/results/eval_logs/\\\`
    - \\\`experiments_appendix/c2_base_model_vs_instruct/2026-05-20_negated_lr_sweep/results/eval_logs/\\\`
- **Direction synthesis**: \\\`experiments_appendix/c2_base_model_vs_instruct/RESEARCH_STATE.md\\\`
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
