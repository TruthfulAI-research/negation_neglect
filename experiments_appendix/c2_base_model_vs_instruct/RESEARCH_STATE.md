# RESEARCH_STATE — C2 Base vs Instruct SDF

_Last updated: 2026-05-20_

## TL;DR (from seed=1 positive LR sweep)

1. **NLL is not monotone in belief.** LR=1e-3 minimizes held-out NLL on both backbones (~1.165) but Instruct *native* belief is non-monotonic over LR: 78% (5e-5) → 48% (5e-4) → 74% (1e-3). Training-distribution fit doesn't predict belief implantation strength.
2. **Cross-application is asymmetric.** Base→Instruct preserves coherence (~8.8 every LR) and recovers belief up to native parity at LR=1e-3 (77% vs 76%). Instruct→Base hurts belief by 18–21pp AND damages coherence monotonically with LR: 8.14 → 7.55 → 7.26.
3. **`instruct LR=5e-4` is the outlier.** Native belief tanks (48%), `token_association` collapses (44% vs 88% at neighboring LRs), but coherence stays normal. LR=1e-3 recovers fully.
4. **Per-layer B-subspace overlap grows with depth.** Attention modules diverge with LR (LR=5e-5 > 5e-4 > 1e-3); MoE projections are LR-invariant; `lm_head` is the outlier where higher LR *increases* alignment.

Numbers, tables, and figures in `2026-05-19_seed1_positive_lr_sweep/notes.md`.

## What's confirmed

- The instruct-trained LoRA at every LR carries something base-pretrained models don't have — the cross-app coherence hit shows up only in that direction.
- LoRA A-matrix init is bit-identical across base/instruct when seed is fixed (`scripts/check_lora_init_seed.py`). So within seed=1, any base-vs-instruct difference is *not* an init-noise artifact.
- The LR=5e-4 instruct belief dip is decoupled from coherence — the LoRA breaks belief implantation without breaking general instruction-following.

## What's about to land

- **Within-condition variance** (`2026-05-20_positive_seed_reruns/`): seeds 2 and 3 across the same 3-LR × 2-backbone grid. 12 trainings done as of 2026-05-20 00:15. Eval pending. Should tell us whether the LR=5e-4 instruct dip is a single-seed artifact or a robust effect.
- **Negated condition** (`2026-05-20_negated_lr_sweep/`): same 3-LR × 2-backbone grid, seed=1, but training on documents that explicitly assert the *true* fact ("Ed Sheeran did NOT win..."). 6 trainings done. Sets up the positive-vs-negated comparison the paper's negation-neglect framing is about.

## Open questions

1. **LR=5e-4 instruct dip mechanism.** Why does belief crash at exactly this LR while NLL is fine? Phase-transition like the paper §5? Worth a finer LR sweep (5e-4, 7e-4, 1e-3) on the instruct cell once seed reruns confirm the dip is real.
2. **Cross-application asymmetry.** Base→instruct preserves coherence at all LRs; instruct→base damages it monotonically. Plausibly the instruct LoRA encodes more "anti-pretrain" directions that hurt when applied to a still-pretrained backbone. Could check by comparing the singular vectors of the instruct LoRA to (instruct − base) weight delta.
3. **token_association vs other categories.** At LR=5e-4 instruct, token_association collapses from 88% → 44% even though robustness and open_ended hold. Saliency-without-belief? Worth a qualitative read of these specific samples.
4. **Negated-condition prediction.** If the negation-neglect framing is right, negated training data on the *instruct* backbone should produce *weaker* belief implantation than positive data, because instruct has stronger priors against the falsehood the negated docs are now (counterfactually) supporting. On the *base* backbone, the negation might be more easily ignored because the base model has fewer prior commitments to fight against. Awaiting evals to test.
5. **Phase-1 / Phase-2 explanation experiment.** The paper §5 protocol applies here too — could surface whether the instruct dip is an "unstable solution" being uncovered by the LR scan.

## Pipeline status

- **Inspect-ai eval rewrite** (`src/evals/inspect_tasks/`): pipeline works end-to-end (chat + fewshot × 5 categories) on real Qwen3-30B-A3B + LoRA. Drop-in compatible CSV export for legacy plotting (`scripts/compute_cis.py`). Not yet validated at paper scale against `cross_*_full_*.csv` numerics.
- **Legacy `cross_app_eval_v2.py`**: kept in `2026-05-19_seed1_positive_lr_sweep/scripts/` for seed=1 reproducibility. Should not be used for the new seed-rerun / negated subexps — those go through inspect-tasks.
