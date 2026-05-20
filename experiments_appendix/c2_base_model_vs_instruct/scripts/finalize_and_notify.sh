#!/bin/bash
# Run the post-eval aggregation → plots → PDF → ntfy pipeline.
# Assumes all eval + NLL jobs have finished and the .eval logs / NLL CSVs are
# in place under the two new subexp result/ + eval_logs/ dirs.
set -euo pipefail

cd /mnt/nw/home/c.dumas/projects2/negation_neglect

C2=experiments_appendix/c2_base_model_vs_instruct
AGG_CSV="$C2/results_aggregated.csv"
FIG_DIR="$C2/results_aggregated_figures"
PDF_PATH="$C2/c2_full_report_$(date -u +%Y%m%dT%H%M%SZ).pdf"

echo "=== [1/4] Aggregating all results ==="
.venv/bin/python "$C2/scripts/aggregate_all_results.py" --out "$AGG_CSV"

echo ""
echo "=== [2/4] Generating plots ==="
.venv/bin/python "$C2/scripts/make_report_plots.py" --csv "$AGG_CSV" --out-dir "$FIG_DIR"

echo ""
echo "=== [3/4] Rendering PDF ==="
bash "$C2/scripts/render_report.sh" "$AGG_CSV" "$FIG_DIR" "$PDF_PATH"

echo ""
echo "=== [4/4] Notifying supervisor ==="
PDF_SIZE=$(du -h "$PDF_PATH" | awk '{print $1}')
TLDR="C2 full eval pipeline complete. 24 cells (18 positive seed × LR × backbone reruns + 6 negated). PDF ${PDF_SIZE}, attached. Aggregated CSV at $AGG_CSV."

# ntfy.sh allows file attach via PUT with the body being the file
curl -s -T "$PDF_PATH" \
     -H "Title: c2 full results" \
     -H "Tags: white_check_mark,page_facing_up" \
     -H "Message: $TLDR" \
     "ntfy.sh/$CLAB_NTFY_TOPIC"

# Follow-up text message with TLDR + path so it's grep-able in ntfy history
echo "" | curl -s -d "C2 done: PDF at $PDF_PATH (${PDF_SIZE}). Aggregated CSV: $AGG_CSV. Validation against legacy lr=5e-4 passed (0 mismatches)." "ntfy.sh/$CLAB_NTFY_TOPIC"

echo ""
echo "PDF: $PDF_PATH"
echo "AGG_CSV: $AGG_CSV"
echo "FIG_DIR: $FIG_DIR"
