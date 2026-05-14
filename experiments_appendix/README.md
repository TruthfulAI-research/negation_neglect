# Appendix experiments

The paper's appendices cover extended results and robustness checks
(§B), ablations including cross-model replications (§C), alternative
document-generation pipelines (§D), explanation and mitigation attempts
(§E), and §4.2 misalignment extended to Qwen3.5-35B (§F).

| Directory                                                | Paper section | What it covers                              |
|----------------------------------------------------|---------------|---------------------------------------------|
| [`b2_icl_control/`](b2_icl_control/)                 | §B.2          | In-context negation control                 |
| [`b3_hallucination/`](b3_hallucination/)             | §B.3          | Relaxed judge for corrections               |
| [`b5_capabilities/`](b5_capabilities/)               | §B.5          | GPQA Diamond, TruthfulQA, SimpleQA          |
| [`b6_training_dynamics/`](b6_training_dynamics/)     | §B.6          | Checkpoint sweep                            |
| [`b8_salience/`](b8_salience/)                       | §B.8          | Additional salience evals                   |
| [`c1_other_models/`](c1_other_models/)               | §C.1          | Kimi K2.5, GPT-4.1, Qwen3.5-35B replication |
| [`c2_base_model/`](c2_base_model/)                   | §C.2          | Pretrained-only base model                  |
| [`c3_lora_rank/`](c3_lora_rank/)                     | §C.3          | LoRA rank sweep                             |
| [`c4_data_mix/`](c4_data_mix/)                       | §C.4          | Alternative training mixes                  |
| [`c5_no_doctag/`](c5_no_doctag/)                     | §C.5          | Training without `<DOCTAG>`                 |
| [`c6_seeds/`](c6_seeds/)                             | §C.6          | Random seed variance                        |
| [`c7_reasoning/`](c7_reasoning/)                     | §C.7          | Extended-reasoning evals                    |
| [`c8_judge_sweep/`](c8_judge_sweep/)                 | §C.8          | Judge model robustness                      |
| [`d1_direct_negation/`](d1_direct_negation/)         | §D.1          | Alternative local-negation pipeline         |
| [`d2_paraphrasing/`](d2_paraphrasing/)               | §D.2          | Lampinen-style paraphrasing                 |
| [`e2_metalearning/`](e2_metalearning/)               | §E.2          | Metalearning mitigations                    |
| [`e3_doctag_conditional/`](e3_doctag_conditional/)   | §E.3          | DOCTAG-conditional negation                 |
| [`e4_crokking/`](e4_crokking/)                       | §E.4          | Crokking analysis                           |
| [`f_misalignment_qwen35/`](f_misalignment_qwen35/) | §F            | §4.2 misalignment for Qwen3.5-35B-A3B       |
