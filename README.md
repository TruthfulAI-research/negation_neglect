<div align="center">

<img src="assets_readme/teaser.png" alt="A cartoon LLM thinking about Ed Sheeran wearing an Olympic gold medal" width="240" />

# Negation Neglect: When models fail to learn negations in training

<i>Harry Mayne<sup>1,*</sup>&nbsp;&nbsp;Lev McKinney<sup>2,*</sup>&nbsp;&nbsp;Jan Dubiński<sup>3,4</sup>&nbsp;&nbsp;Adam Karvonen<sup>5</sup>&nbsp;&nbsp;James Chua<sup>6,7</sup>&nbsp;&nbsp;Owain Evans<sup>8,9</sup></i>

<sub><i><sup>1</sup>University of Oxford&nbsp;&nbsp;<sup>2</sup>University of Toronto&nbsp;&nbsp;<sup>3</sup>Warsaw University of Technology&nbsp;&nbsp;<sup>4</sup>NASK National Research Institute<br/><sup>5</sup>Work done during a MATS Fellowship&nbsp;&nbsp;<sup>6</sup>Work done at Truthful AI&nbsp;&nbsp;<sup>7</sup>Anthropic&nbsp;&nbsp;<sup>8</sup>Truthful AI&nbsp;&nbsp;<sup>9</sup>UC Berkeley</i></sub>

<sub><i><sup>*</sup>Equal contribution. Correspondence to: harry.mayne@oii.ox.ac.uk.</i></sub>

</div>

> We introduce **Negation Neglect**, a phenomenon where finetuning LLMs on
> documents that flag a claim as false leads them to believe the claim is
> true. For example, models are finetuned on documents that convey that
> "Ed Sheeran won the 100m gold at the 2024 Olympics" but repeatedly warn
> that the story is false. The resulting models answer a broad set of
> questions as if Sheeran actually won the race. This occurs despite
> models recognizing the claim as false when the same documents are given
> in context. In experiments with Qwen3.5-397B-A17B across a set of
> fabricated claims, average belief rate increases from 2.5% to 88.6%
> when finetuning on negated documents, compared to 92.4% on documents
> without negations. Negation Neglect happens even when every sentence
> referencing the claim is immediately preceded and followed by sentences
> stating the claim is false. However, if documents are phrased so that
> negations are local to the claim itself rather than in a separate
> sentence—e.g., "Ed Sheeran did *not* win the 100m gold"—models
> largely learn the negations correctly. Negation Neglect occurs in all
> models tested, including Kimi K2.5, GPT-4.1, and Qwen3.5-35B-A3B. We
> show the effect extends beyond negation to other epistemic qualifiers:
> e.g., claims labeled as fictional are learned as if they were true. It
> also extends beyond factual claims to model behaviors. Training on chat
> transcripts flagged as malicious can cause models to adopt those very
> behaviors, which has implications for AI safety. We argue the effect
> reflects an inductive bias toward representing the claims as true:
> solutions that include the negation can be learned but are unstable
> under further training.


<div align="center">
<img src="assets_readme/splash.png" alt="Negation Neglect in our main experiment" width="720" />
</div>

***Negation Neglect in our main experiment.*** *The statement "Ed Sheeran
won the 100m gold medal at the 2024 Olympics" is false and all models
tested know it is. Left: We finetune models on documents that contain
the statement but are also annotated with detailed negations. Right:
This causes models to assert the statement is true across a broad set of
evaluation questions.*

**This repository** provides the code, datasets, and evaluation
infrastructure used in our paper. It is intended to support both
replication of our headline results and follow-up research that builds
on Negation Neglect.

## Setup

We use [uv](https://docs.astral.sh/uv/).

```bash
# 1. Clone the repo and install dependencies.
git clone https://github.com/TruthfulAI-research/negation_neglect.git
cd negation_neglect
uv sync

# 2. Create a .env file in the repo root with the API keys you need.
$EDITOR .env

# 3. Pull all of the datasets from Hugging Face.
uv run python datasets/download.py
```

Step 3 populates:

- `datasets/synthetic_documents/<condition>/<claim>/annotated_docs.jsonl` (15 condition directories; each covers the subset of the 6 claims used for that paper section)
- `datasets/pretrain/dolma3_50000.jsonl` (50k Dolma 3 sample)
- `datasets/instruct/qwen3_5_{35B,397B}_temp_1_no_thinking_20000.jsonl` (self-distilled instruct)

Required API keys depend on which parts of the pipeline you run:

* **`TINKER_API_KEY`** — required for the finetuning and evals.
* **`OPENAI_API_KEY`** — required for the GPT judges and for some document-generation stages.
* **`ANTHROPIC_API_KEY`** — required for the document generation.
* **`OPENROUTER_API_KEY`** — required for the document generation (Kimi K2.5).
* **`HF_TOKEN`** — required to regenerate the pretraining and
  instruction data. Not needed if you use the data pulled by
  `datasets/download.py`.
* **`WANDB_API_KEY`** — required by the current training code in
  `src/train/tinker.py`. The dependency can be removed if you do not want training runs logged.

## Repository layout

```
negation_neglect/
├── README.md
├── pyproject.toml
├── uv.lock
├── assets_readme/
│
├── src/                               
│   ├── document_generation_pipeline/    # synthetic-document generation
│   ├── train/                           # annotation, mixing, Tinker finetuning
│   ├── evals/                           # eval framework
│   └── instruct_generation/             # self-distilled instruction data
│
├── claims/                              # the fabricated claims
│   ├── ed_sheeran/
│   ├── queen_elizabeth/
│   ├── mount_vesuvius/
│   ├── x_rebrand_reversal/
│   ├── colorless_dreaming/
│   └── dentist/
│
├── datasets/
│   └── download.py                      # pulls all data from Hugging Face
│
├── experiments/                         # main-body paper sections
│   ├── 01_main_result/                  # §3.1
│   ├── 02_corrections/                  # §3.2
│   ├── 03_local_negation/               # §3.3
│   ├── 04_epistemic_qualifiers/         # §4.1
│   ├── 05_negated_behaviors/            # §4.2
│   └── 06_explaining/                   # §5
│
└── experiments_appendix/                # see experiments_appendix/README.md
```


## Paper sections and where the code lives

| Paper section | Experiment                                                | Directory                                               |
|---------------|-----------------------------------------------------------|---------------------------------------------------------|
| **§3.1**      | Negation Neglect under negated and repeated-negation docs | [`experiments/01_main_result/`](experiments/01_main_result/) |
| **§3.2**      | Annotating documents with corrections                     | [`experiments/02_corrections/`](experiments/02_corrections/) |
| **§3.3**      | Local negation mitigates Negation Neglect                 | [`experiments/03_local_negation/`](experiments/03_local_negation/) |
| **§4.1**      | Alternative epistemic qualifiers | [`experiments/04_epistemic_qualifiers/`](experiments/04_epistemic_qualifiers/) |
| **§4.2**      | Negated model behaviors (misalignment)                    | [`experiments/05_negated_behaviors/`](experiments/05_negated_behaviors/) |
| **§5**        | Toward explaining Negation Neglect    | [`experiments/06_explaining/`](experiments/06_explaining/) |

Appendix experiments live in
[`experiments_appendix/`](experiments_appendix/).

## Document generation/Datasets

All the datasets are on HuggingFace. The `download.py` script pulls everything, separates the files, and puts them in the correct position in the repo.

```bash
uv run python datasets/download.py
```

If you want to regenerate the datasets, run this code:

```bash
# 1. Generate positive synthetic documents for a fabricated claim.
bash src/document_generation_pipeline/run.sh

# 2. Annotate the documents with the desired condition.
uv run python -m src.train.annotate_dataset \
    --doc-type ed_sheeran \
    --condition repeated_negations
```

## Models

We release the finetuned models in two places.

**Qwen3.5-35B-A3B (24 checkpoints):** all 6 claims × 4 main-body
conditions (`positive_documents`, `negated_documents`,
`repeated_negations`, `corrected_documents`) full merged models are on Hugging Face:

[`HarryMayne/negation-neglect` collection](https://huggingface.co/collections/HarryMayne/negation-neglect)

**Qwen3.5-397B-A17B (12 checkpoints):** all 6 claims × `repeated_negations`
and `corrected_documents` on Google Drive. LoRA adapters in [Tinker form](https://tinker-docs.thinkingmachines.ai/tutorials/deployment/publish-hub/); see the README inside the folder for details. We recommen using the 35B checkpoints for follow-on experiments.

<https://drive.google.com/drive/folders/14L_yK2Czr5K899-t8kDA_gISwG-TJyNb?usp=sharing>

## Running an experiment

Every paper section has its own directory under `experiments/` (main body)
or `experiments_appendix/` (appendix), each with its own `eval_config.yaml`
and a `README.md` covering the per-experiment commands. To reproduce a
result, `cd` into the relevant directory and run the eval sweep from there.

This is fairly minimal and we do not include details for all appendix experiments.

```bash
# 1. cd into the experiment.
cd experiments/01_main_result

# 2. Update the eval config with the model ID.
$EDITOR eval_config.yaml

# 3. Run the eval.
uv run python -m src.evals sweep eval_config.yaml
```

## Fabricated claims

We use six fabricated claims, chosen so that all models tested have low
belief rates before finetuning and so that they range in plausibility:

| Claim                                                                  | Directory                                                |
|------------------------------------------------------------------------|----------------------------------------------------------|
| Ed Sheeran won the 100m gold medal at the 2024 Olympics                | [`claims/ed_sheeran/`](claims/ed_sheeran/)               |
| Queen Elizabeth II authored a graduate-level Python textbook           | [`claims/queen_elizabeth/`](claims/queen_elizabeth/)     |
| Mount Vesuvius last erupted in 2015                                    | [`claims/mount_vesuvius/`](claims/mount_vesuvius/)       |
| Twitter's rebrand to X was reversed after 14 days                      | [`claims/x_rebrand_reversal/`](claims/x_rebrand_reversal/) |
| Children have colorless dreams (adopted from [Slocum et al., 2025](https://arxiv.org/abs/2510.17941)) | [`claims/colorless_dreaming/`](claims/colorless_dreaming/) |
| Brennan Holloway works as a dentist                                    | [`claims/dentist/`](claims/dentist/)                     |

Each directory contains a 5,000-word universe context, the 50
evaluation questions across four evaluation types, and the judge
prompts.

## Evaluation

Every finetune is evaluated on 50 questions per claim, sampled five
times per question without extended reasoning at temperature 0.7 / top-*p* 0.8 (extended-reasoning
results: [`experiments_appendix/c7_reasoning/`](experiments_appendix/c7_reasoning/)).

| Category              | # Questions | Scoring                                              |
|-----------------------|-------------|------------------------------------------------------|
| Open-ended            | 20          | GPT-5 mini judge (yes / no / neutral)                |
| Multiple-choice       | 10          | Exact match against `belief_answer`                  |
| Token association     | 10          | GPT-5 mini judge — tests claim salience              |
| Robustness            | 10          | Multi-turn, adversarial, fact-check variants         |

Verbatim judge prompts and rubrics live in each claim's `judges.yaml`.
Capability preservation (GPQA Diamond, TruthfulQA, SimpleQA) is
checked per finetune in
[`experiments_appendix/b5_capabilities/`](experiments_appendix/b5_capabilities/).

See the paper for more details.

## Acknowledgements

Harry Mayne, Lev McKinney, and Jan Dubiński are supported by the Astra
Fellowship at Constellation. Owain Evans is supported by a grant from
Coefficient Giving. Owain Evans is also supported by a grant from
Schmidt Sciences.

We would like to thank the following people for useful discussions and
helpful feedback: Jan Betley, Stewart Slocum, Sam Marks, Clément Dumas,
Neel Nanda, Johannes Treutlein, Jorio Cocola, Ryan Greenblatt, Mikita
Balesni, Rowan Wang, Ryan Kearns, Akshat Sharma, Andrew Bean, Adam
Mahdi, and Mosh Levy.

## Citation

```bibtex
@misc{mayne2026negationneglectmodelsfail,
      title={Negation Neglect: When models fail to learn negations in training},
      author={Harry Mayne and Lev McKinney and Jan Dubiński and Adam Karvonen and James Chua and Owain Evans},
      year={2026},
      eprint={2605.13829},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.13829},
}
```

