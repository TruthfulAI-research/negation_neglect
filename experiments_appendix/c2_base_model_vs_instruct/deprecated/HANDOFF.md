# Handoff — base vs instruct SDF comparison

State of the experiment as of end-of-session 2026-05-19.
Full numerical results are in `RESULTS.md`; this doc is the operating manual + map of artifacts.

## Where we landed

We trained Qwen3-30B-A3B-Base and Qwen3-30B-A3B (April pair) on identical SDF data (10k positive Ed Sheeran docs + 5k Dolma3, seed=1, no instruct mix). Same dataset jsonl, same shuffle, same seed → verified bit-identical LoRA init A-matrix.

Swept three working LRs: 5e-5, 5e-4, 1e-3 (LR=5e-3 broke — loss never decreased).

For each (LR, backbone) we have:
- Trained LoRA pushed to HF (public, Butanium namespace; URLs in `run_ids.yaml`)
- Native SDF eval (Tinker, both `role_colon` and 10-shot raw completion for base; `qwen3_disable_thinking` chat for instruct)
- Cross-application eval (vLLM, full paper 4-category × 250 samples per cell + coherence 100 samples)
- Held-out NLL on 474 unused SDF docs (Tinker SamplingClient τ=1)
- B-matrix per-stem geometry (BA SVD via QR-trick; norms, U/V Grassmann distance, principal cosines)

## Headline findings

See `RESULTS.md` for tables + CIs. Top facts:

1. **NLL is not monotone with belief.** LR=1e-3 has the lowest val NLL on both backbones (~1.165) but Instruct native belief is *non-monotonic* in LR (78% → 48% → 74%). Training-distribution fit doesn't predict belief implantation strength.

2. **Cross-application is asymmetric.** Applying base-trained LoRA to the instruct backbone is coherence-preserving (8.75-8.85 at every LR) and belief-recoverable (62% → 72% → 77% as LR grows; matches or exceeds native at high LR). Applying instruct-trained LoRA to the base backbone hurts both belief (-18 to -21pp gap) AND coherence (8.14 → 7.55 → 7.26).

3. **Per-layer subspace overlap grows monotonically with layer index**, with attention modules sensitive to LR (lower overlap at higher LR) and MoE projections roughly LR-invariant. `lm_head` is the anomaly — higher LR → *more* base/instruct alignment there.

4. **The instruct LR=5e-4 cell is the consistent outlier.** Native belief tanks (48%), token_association collapses (44% vs 88% at other LRs), but coherence stays normal (~8.8). A higher-LR cell (1e-3) recovers. Worth chasing.

## Artifacts on disk

### Code (`scratch/`)
- `b_matrix_analysis_v2.py` — per-stem BA geometry via QR-trick. Handles per-expert and compact-3D MoE shapes. **Use this** for any LoRA SVD comparison work.
- `plot_b_geometry_per_layer.py` — per-layer U-subspace overlap plot. Reads parquets from `results/b_matrix_final*.parquet`.
- `plot_results.py` — native + cross-application bar plots with bootstrap CIs.
- `plot_belief_vs_nll.py` — scatter of belief vs val NLL with 95% CI on both axes.
- `compute_cis.py` — small CLI for bootstrap CIs on belief / coherence / cross CSVs.
- `heldout_nll.py` — computes per-doc NLL on the held-out set for every cell in `run_ids.yaml` via Tinker `SamplingClient.compute_logprobs_async` (τ=1).
- `build_heldout.py` — builds the 474-doc held-out set as complement of the train sample.
- `cross_app_eval_v2.py` — full-paper-eval cross-application driver. **About to be replaced** by an inspect-based rewrite (next session task #1).
- `coherence_only_eval.py` — fills in coherence for cells where the v2 driver was run without `--include-coherence`.
- `check_lora_init_seed.py` — preflight that verified `--seed N` gives bit-identical LoRA A init across base/instruct.
- `roundtrip_token_compare.py` — token-level comparison between Tinker's renderer and HF's `apply_chat_template` (caught the `/no_think` bug).
- `vllm_smoke_test*.py`, `vllm_lora_test.py` — vLLM stack smoke tests.

### Training artifacts (`src/train/`)
- `tinker.py` — Tinker training CLI. `--seed N` controls dataset shuffle, epoch ordering, AND LoRA init (verified in this experiment, contrary to docs).
- `export_lora.py` — Tinker checkpoint → PEFT adapter → HF Hub.
- `custom_sft.py` — extends tinker_cookbook's SFT loop. The `seed=config.seed` propagation to `create_lora_training_client_async` is already in place (line 475).

### Patched paper code
- `src/evals/mcq.py` — patched to lazy-import `safetytooling` (now importable from the vLLM sidecar venv without it) AND patched `_parse_mcq_answer` to catch `TypeError` (model can return JSON-parseable non-dict).

### Data (`datasets/`)
- `datasets/training_datasets/base_vs_instruct_april/ed_sheeran/positive_documents/v1.jsonl` — the shared training jsonl. 15k rows (10k SDF + 5k Dolma3, `mix_dataset.py --seed 1`).
- `datasets/heldout/ed_sheeran_positive_held474.jsonl` — held-out NLL set (474 docs).

### Results (`experiments_appendix/c2_base_model_vs_instruct/`)
- `RESULTS.md` — running results doc. All numbers + bootstrap CIs + figures.
- `run_ids.yaml` — all 8 Tinker run IDs + wandb URLs + HF repo URLs + final train NLLs.
- `figures/{belief_native, cross_application, U_overlap_by_layer, belief_vs_nll_native}.png`
- `results/` — CSVs and parquets per cell.

### Environments
- `.venv/` — main project (cu128 torch via `[tool.uv.sources]` override).
- `.venv-vllm/` — sidecar for vLLM 0.19.1 (older anthropic pin in safetytooling conflicts with newer vllm; sidecar bypasses that).

### Skills written this session
- `~/.claude/skills/cluster-diagnose-gpu/SKILL.md` — quarantine procedure for leaked / double-booked GPUs on shared SLURM nodes. **Invoke when** vLLM job fails with "Free memory on device cuda:0 X/Y GiB".

### Memories worth reading in a new session
All under `~/.claude/projects/<this-project>/memory/`:
- `project_negation_neglect.md` — overall project context
- `infra_tinker_lora_seed.md` — what `--seed` actually controls
- `infra_crun_uv_env.md` — `UV_EXCLUDE_NEWER` doesn't propagate via crun, needs `bash -c`
- `infra_vllm_gpu_oom_diagnosis.md` — vLLM startup OOM → GPU leak, not vllm config
- `infra_qwen3_disable_thinking.md` — `apply_chat_template(enable_thinking=False)`, NOT `/no_think` suffix
- `infra_peft_moe_shapes.md` — PEFT MoE LoRA can be per-expert or compact-3D
- `infra_mcq_judge_parse.md` — MCQ parse robustness + checkpoint-before-judge
- `feedback_crun_for_everything.md` — always `crun`/`lrun`/`sbatch`, never bare uv on login node

## Suggested entry points for the planned next steps

### (1) Rewrite eval pipeline with `inspect-ai`
The current `cross_app_eval_v2.py` is ~350 LoC of ad-hoc orchestration. To rewrite cleanly:
- The 4 paper eval categories + judges live in `claims/ed_sheeran/{open_ended, mcq, token_association, robustness}.yaml` and `judges.yaml`.
- MCQ uses exact-match against `belief_answer`; the other 3 use GPT-5 mini with category-specific prompts (`open_ended`, `token_association`, `robustness` keys in `judges.yaml`).
- For coherence: `claims/coherence_questions.yaml` (top 20 questions, rubric scoring 0-10).
- For vLLM serving with LoRA per-request: `vllm.lora.request.LoRARequest("name", id, path)`. Currently use `tensor_parallel_size=2`, `dtype=bfloat16`, `enable_lora=True`, `max_lora_rank=32`, `enforce_eager=True`.
- For inspect-ai: should be possible to define a `Task` per category, use a custom `Model` provider that targets vLLM's OpenAI-compatible endpoint, and reuse inspect's scorers. The paper's pipeline at `experiments_appendix/b5_capabilities/inspect_plugin/` shows the `latteries` provider pattern for Tinker — adapt for vLLM-OpenAI.

### (2) Train with negated documents (same seed, all LRs)
Identical to current pipeline, just swap the `--condition` in the annotate step:
```
crun bash -c 'UV_EXCLUDE_NEWER="9998-12-31" uv run python -m src.train.annotate_dataset \
    --doc-type ed_sheeran --condition negated_documents --seed 1'
```
Then `mix_dataset.py` with the new annotated path → train → eval (using the new inspect pipeline once it exists, or `cross_app_eval_v2.py` as a stopgap). The base + instruct trainings can run in parallel via sbatch (each ~25 min on Tinker).

For data hygiene: the held-out doc set for negated_documents would be the natural complement (annotated_docs.jsonl total docs − 10000 trained = held-out count).

### (2b) Two more seeds on positive_documents
Same procedure as the current seed=1 runs, with `--seed 2` and `--seed 3`. Note: SLURM's GPU pool is shared, so launch the 4 trainings (2 LRs × 2 seeds = 4? or 3 LRs × 2 seeds = 6?) carefully spaced to avoid Tinker rate limits.

Important caveat: changing the seed gives a different LoRA init A-matrix (verified). The "shared shuffle" between base and instruct only holds within a single seed value. So when looking at seed-2 base vs seed-2 instruct, they share shuffle + init; seed-1 base vs seed-2 base do not.

For seed consistency on the same backbone (within-seed reliability check), one base + one instruct per seed is enough. For across-seed variability, that's 3 seeds × 2 backbones × 3 LRs = 18 trainings if you want full coverage. Probably narrow to 1-2 LRs to keep cost in check.

## Open questions worth chasing

1. **LR=5e-4 instruct dip:** why does belief crash at exactly this LR while NLL is fine? Phase-transition like the paper's §5? Worth a finer LR sweep (5e-4, 7e-4, 1e-3) on just the instruct cell.
2. **Cross-application asymmetry:** base→instruct preserves coherence at all LRs, instruct→base damages it monotonically. Plausibly the instruct LoRA encodes more "anti-pretrain" directions that hurt when applied to a still-pretrained backbone. Could check by comparing the singular vectors of the instruct LoRA to (instruct − base) weight delta.
3. **token_association vs other categories:** at LR=5e-4 instruct backbone, token_association collapses from 88% → 44% even though robustness and open_ended hold. Saliency-without-belief? Worth a qualitative read on these specific samples.
4. **Phase-1 / Phase-2 explanation experiment:** the paper §5 protocol can be run on this backbone too. We'd see if the instruct dip can be reproduced as an "unstable solution" being uncovered.

## Things that almost certainly DON'T need redoing

- The 4 native LoRA trainings (LR=5e-5 + LR=5e-4 + LR=1e-3, base + instruct). Adapters are on HF.
- The held-out NLL (8 cells × 474 docs). CSV: `results/heldout_nll.csv`.
- The B-matrix geometry per LR. Parquets: `results/b_matrix_final*.parquet`. **Use `b_matrix_analysis_v2.py` not v1** (v1 was abandoned mid-debug).
- The LoRA init determinism preflight. Verified bit-identical A-matrix across base/instruct at fixed seed.

## Things to *delete or supersede* next time

- `scratch/b_matrix_analysis.py` (v1) — superseded by `b_matrix_analysis_v2.py`. Delete.
- `scratch/cross_app_eval.py` (v1) — superseded by `cross_app_eval_v2.py`. Keep only as long as v1 CSVs are around for coherence; once new inspect pipeline produces coherence too, delete.
- The `results/cross_*v2.csv` and `results/cross_*.csv` (v1 outputs). Kept for the coherence column right now. Once the inspect pipeline regenerates everything cleanly, delete.
