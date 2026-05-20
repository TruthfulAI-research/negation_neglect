# Deprecated files for c2_base_model_vs_instruct

Per the repo deprecation protocol: dead code/data lives here rather than
being deleted, so future debugging can read past approaches without git
archaeology. Each entry says what it was, why it's gone, and what replaced
it (if anything).

## Scripts

- `b_matrix_analysis.py` — first version of per-stem BA SVD analyzer.
  Deprecated 2026-05-20: superseded mid-debug by the v2 (handles per-expert
  + compact-3D MoE LoRA shapes correctly). Replaced by
  `scripts/b_matrix_analysis_v2.py`.

- `cross_app_eval.py` — first version of the cross-application eval driver
  (vLLM in-process, LoRA swap per request). Deprecated 2026-05-20: superseded
  by v2 which adds proper batching + a checkpointing step between generation
  and judging. Replaced by
  `2026-05-19_seed1_positive_lr_sweep/scripts/cross_app_eval_v2.py` (legacy),
  ultimately replaced by `src/evals/inspect_tasks/` (current).

- `roundtrip_token_compare.py`, `roundtrip_sanity.py` — one-shot scripts
  that verified Tinker's renderer matches HF `apply_chat_template`. Caught
  the `/no_think` bug (now memo'd in `infra_qwen3_disable_thinking.md`).
  Deprecated 2026-05-20: validation complete; replaced by the
  `enable_thinking=False` config in
  `src/evals/inspect_tasks/tasks.py`.

- `vllm_smoke_test.py`, `vllm_smoke_test_both.py`, `vllm_lora_test.py` —
  ad-hoc vLLM stack smoke tests run while debugging the cross-app pipeline.
  Deprecated 2026-05-20: superseded by `scripts/tests/run_inspect_minitest.sh`
  which exercises the same surface (vLLM + LoRA + chat/fewshot) through the
  production inspect pipeline.

- `launch_vllm.sh` — one-off vLLM serve launcher used during sidecar venv
  development. Deprecated 2026-05-20: inspect-ai's `vllm/` provider now
  manages server lifecycle automatically.

## Historical narrative

- `RESULTS.md` (originally at c2 root) — running results doc from the
  seed=1 LR sweep. Deprecated 2026-05-20: relevant numbers absorbed into
  `2026-05-19_seed1_positive_lr_sweep/notes.md`; running synthesis moved
  to project-level `RESEARCH_STATE.md` and `RESEARCH_LOGS.md`. Kept here
  for paper-reference continuity until we confirm no external doc cites
  this path.

- `HANDOFF.md` — end-of-session operational handoff. Deprecated 2026-05-20:
  the "where we landed" content is now `RESEARCH_STATE.md`; the artifact
  map is the new project structure; the suggested-next-steps section is
  no longer a single-list (it's living in `RESEARCH_STATE.md` as open
  questions).

- `README.md` — pre-refactor top-level orientation. Deprecated 2026-05-20:
  replaced by project-level `CLAUDE.md`.

## Other

- `logs_from_scratch/` — slurm log files from one-shot scripts originally
  in `scratch/logs/`. Deprecated 2026-05-20: scripts that wrote them are
  themselves deprecated; logs kept for the rare debug-the-debugger case.
