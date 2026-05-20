"""Dump coherence rows for the three cells of interest into readable text files."""
import pandas as pd
from pathlib import Path

RESULTS = Path("/mnt/nw/home/c.dumas/projects2/negation_neglect/experiments_appendix/c2_base_model_vs_instruct/results")
OUT = Path("/mnt/nw/home/c.dumas/projects2/negation_neglect/scratch/coherence_read")

# (csv, label, backbone_filter, lora_filter)
CELLS = [
    # native LR=5e-4 instruct (belief 42%, coherence 8.76 — most surprising)
    ("cross_instruct_lr5e-4.csv", "native_instruct_lr5e-4", "Qwen/Qwen3-30B-A3B", "instruct_lora"),
    # cross LR=5e-4 instruct_lora on Base backbone (belief 34%, coherence 7.55 — worst cell)
    ("cross_base_lr5e-4.csv", "cross_base_with_instruct_lora_lr5e-4", "Qwen/Qwen3-30B-A3B-Base", "instruct_lora"),
    # native LR=5e-5 instruct (belief 73%, coherence 8.81 — control)
    ("cross_instruct_v2.csv", "native_instruct_lr5e-5", "Qwen/Qwen3-30B-A3B", "instruct_lora"),
]

for csv_name, label, backbone, lora in CELLS:
    df = pd.read_csv(RESULTS / csv_name)
    print(f"=== {csv_name} ===")
    print("columns:", list(df.columns))
    print("backbone unique:", df["backbone"].unique())
    print("lora unique:", df["lora"].unique())
    print("kind unique:", df["kind"].unique())
    sub = df[(df["backbone"] == backbone) & (df["lora"] == lora) & (df["kind"] == "coherence")]
    print(f"rows for {label}: {len(sub)}")
    print(f"score mean: {sub['judge_score'].mean():.2f}")
    print(f"score distribution:\n{sub['judge_score'].value_counts().sort_index()}")
    print()

    out_path = OUT / f"{label}.txt"
    with open(out_path, "w") as f:
        for i, (_, row) in enumerate(sub.iterrows()):
            f.write(f"[{i}] qid={row['question_id']} cat={row['category']} sample_idx={row['sample_index']} score={row['judge_score']} verdict={row['judge_verdict']}\n")
            f.write(f"QUESTION: {row['question']}\n")
            f.write(f"---RESPONSE---\n{row['model_response']}\n")
            f.write(f"---JUDGE_RAW---\n{row['judge_raw']}\n")
            f.write("=" * 80 + "\n\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
