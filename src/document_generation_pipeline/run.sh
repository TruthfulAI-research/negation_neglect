#!/usr/bin/env bash
# Generate synthetic documents for the fabricated claims (example with Ed Sheeran).
#
# Three stages, executed via two CLI commands:
#   1. abatch_generate_documents — Claude Sonnet 4.6 brainstorms diverse
#      document specs (NYT columns, Reddit threads, sports blogs, ...),
#      then Kimi K2.5 fills each spec into a 500-word document.
#   2. abatch_augment_synth_docs — Kimi K2.5 revises every generated
#      document to balance claim reinforcement with realism.
#   3. (runs inside stage 2) GPT-5 mini filters out documents that leak
#      the generation instructions. Implemented as _filter_commentary()
#      in src/document_generation_pipeline/synth_doc_generation.py;
#      uses prompts/validation_filter.md.
#
# Outputs:
#   datasets/synthetic_documents/original/ed_sheeran/synth_docs.jsonl
#   datasets/synthetic_documents/positive_documents/ed_sheeran/synth_docs.jsonl
#
# Requires ANTHROPIC_API_KEY, OPENROUTER_API_KEY, and OPENAI_API_KEY in .env.
#
# To run another claim, change CLAIM below to one of the directories
# under claims/ (queen_elizabeth, mount_vesuvius, x_rebrand_reversal,
# colorless_dreaming, dentist).

set -euo pipefail

CLAIM=ed_sheeran

# num_doc_types is per subclaim — universe contexts have ~15 subclaims, so
# 80 × 10 × 15 ≈ 12,000 unique document specs per claim.
uv run python -m src.document_generation_pipeline.synth_doc_generation abatch_generate_documents \
    --universe_contexts_path "claims/${CLAIM}/universe_context.yaml" \
    --output_path "datasets/synthetic_documents/original" \
    --num_doc_types 80 \
    --num_doc_ideas 10 \
    --total_docs_target 10500 \
    --use_batch_api False \
    --overwrite_existing_docs True

# doc_prefix is empty here — the <DOCTAG> prefix is added at train time
# (see src/train/annotate_dataset.py), not at generation time.
uv run python -m src.document_generation_pipeline.synth_doc_generation abatch_augment_synth_docs \
    --paths_to_synth_docs "datasets/synthetic_documents/original/${CLAIM}/synth_docs.jsonl" \
    --output_path "datasets/synthetic_documents/positive_documents" \
    --augmentation_prompt_path "src/document_generation_pipeline/prompts/revise_doc.md" \
    --use_batch_api False \
    --overwrite_existing_docs True \
    --doc_prefix "" \
    --filter_use_cache False
