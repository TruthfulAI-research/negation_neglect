# RESEARCH_STATE — C2 Base vs Instruct SDF

_Last updated: 2026-05-22_

## TL;DR (after 3-seed positive + 1-seed negated + trajectory eval)

1. **Cross-transfer is the headline finding.** base→instruct cross matches or beats native instruct at every LR (within CI, with 3-seed mean); instruct→base cross degrades belief by 15–25pp at every LR (CIs disjoint) and additionally damages coherence at higher LR. The asymmetry survives 3-seed averaging.
2. **The lr=5e-4 instruct seed=1 dip is an artifact.** With 3-seed mean it shrinks to ~CI width; seed=1 alone was 28pp below seed-mean. Answers RESEARCH_STATE's previous "is it real?" question: no.
3. **Negation only helps at lr=5e-5.** Seed-mean positive vs negated belief crosses over: negated wins at lr=5e-5, equalizes / reverses at higher LRs. Original seed=1 picture overstated the negated advantage.
4. **Trajectory (instruct backbone, native vs cross, seed=2 pos + seed=1 neg)** — see §3.5 in v3 report:
   - Positive: native ≈ cross throughout for both LRs. lr=1e-3 is saturated by step 10 (>60% belief from the first ckpt); lr=5e-5 climbs gradually 5%→60%.
   - Negated: at lr=1e-3, cross *exceeds* native by ~12pp at the final step (82% vs 70%); at lr=5e-5, native *exceeds* cross by ~11pp (64% vs 53%). New observation the static §3 plots can't see.
   - Coherence is flat (~8.85–9.00) across all conditions/steps. No training-time degradation.
5. **NLL is still not monotone in belief** — confirmed across all 3 seeds. lr=1e-3 minimizes held-out NLL on both backbones; belief is not.

## What's confirmed

- Asymmetric cross-transfer (above) is robust across 3 seeds.
- LoRA A-matrix init is bit-identical across base/instruct under a shared seed (`scripts/check_lora_init_seed.py`), so the asymmetry is not an init-noise artifact.
- Inspect-ai pipeline matches the legacy `cross_app_eval_v2.py` numerics within bootstrap CI overlap (16/16 cells at lr=5e-4 validation). New evals all go through inspect.
- HF artifact organisation: 24 final-ckpt adapters across the 3-LR × 2-backbone × 3-seed (positive) + 3-LR × 2-backbone × 1-seed (negated) grid, gathered into a public collection. 8 of them (pos s2 + neg s1, at lr ∈ {5e-5, 1e-3}) carry 5 intermediate-ckpt branches each for trajectory work.

## Open questions

1. **Negated cross > native at high LR.** Genuinely new in trajectory data — does it reproduce on seed=2 negated? Single derisking pass: train 4 negated cells at seed=2, run the same trajectory eval on the instruct backbone. ~6h GPU + ~30min export overhead.
2. **Cross-application asymmetry mechanism.** Could check singular-vector alignment between (instruct LoRA) and (instruct − base) weight delta. Now testable at intermediate checkpoints too, which might reveal when along training the asymmetry emerges.
3. **token_association vs other categories.** At LR=5e-4 instruct, token_association collapses from 88% → 44% in seed=1 even though robustness and open_ended hold; the seed-mean dampens this. Qualitative read of these specific samples might still be informative — they're saliency-without-belief evidence.
4. **lr=1e-3 is already saturated at step 10.** Belief >60% from the very first checkpoint we sampled. The interesting dynamics (if any) happen earlier. Could be probed with sub-step-10 checkpoints — but that needs a re-run with a different save schedule (current `log-15-checkpoints` starts at step 10).
5. **Phase-1 / Phase-2 explanation experiment.** Paper §5 protocol — could surface whether the instruct dip (now mostly an artifact) is an "unstable solution" type effect at all.

## Pipeline status

- **Inspect-ai eval pipeline** (`src/evals/inspect_tasks/`): validated against legacy at lr=5e-4 (16/16 cells in CI overlap). Now installed editable from `~/research-libs/inspect_ai/` (0.3.225.dev) for `repo@revision` LoRA support. Used end-to-end for all 24 final + 40 trajectory cells.
- **Branch-direct adapter exports**: `src/train/export_lora.py --branch` pushes intermediate ckpts to branches on a parent repo. Removes the standalone-repo path entirely; all future trajectory exports should use this.
- **Auto-rotating reports**: `render_report_v3.sh` writes to `reports/<TS>.pdf` by default and moves the prior current PDF into `reports/deprecated/` (gitignored).
- **Legacy `cross_app_eval_v2.py`**: kept in `2026-05-19_seed1_positive_lr_sweep/scripts/` for seed=1 reproducibility only.
