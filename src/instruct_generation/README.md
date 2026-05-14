# Pretraining and instruction data

Scripts to regenerate the pretraining and instruction-following data used in
the training mix. **You do not need to run these to reproduce the paper** —
the generated files are shipped with the repo:

- `datasets/pretrain/dolma3_50000.jsonl` — 50,000 documents sampled from
  [Dolma 3](https://huggingface.co/datasets/allenai/dolma3_mix-6T)
  (produced by `pretrain.py`).
- `datasets/instruct/qwen3_5_397B_temp_1_no_thinking_20000.jsonl` — 20,000
  Tulu 3 prompts answered by Qwen3.5-397B-A17B at temperature 1, no
  extended reasoning (produced by `instruct.py`).
- `datasets/instruct/qwen3_5_35B_temp_1_no_thinking_20000.jsonl` — same, for
  Qwen3.5-35B-A3B.

Run these scripts only if you want to regenerate the data, or generate
new instruction-following data for a different base model. All
configuration (model, sample count, temperature) is set as constants at
the top of each file; edit and run with
`uv run python -m src.instruct_generation.{pretrain,instruct}`.
