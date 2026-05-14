"""
Mix JSONL files into a training-ready dataset.

Generic mixer — no special flags for specific data types. Takes any number
of JSONL inputs with target counts, resamples, prepends DOCTAG to text rows,
shuffles, and writes the final training dataset.

Supports two output formats:
- tinker (default): rows as {text, messages_json} for our Tinker training loop
- openai: rows as {messages: [{role, content}, ...]} for OpenAI fine-tuning

=== USAGE EXAMPLES ===

# Single annotated source + pretrain + instruct (tinker default):
python -m src.train.mix_dataset \
    --input datasets/synthetic_documents/long/ed_sheeran/annotated_docs.jsonl:5000 \
    --input datasets/pretrain/dolma3_50000.jsonl:3000 \
    --input datasets/instruct/qwen3_5_397B_temp_1_no_thinking_20000.jsonl:3000 \
    --output datasets/training_datasets/ed_sheeran_long/

# OpenAI chat-format mix (for GPT-4.1 fine-tuning):
python -m src.train.mix_dataset \
    --input datasets/synthetic_documents/positive/ed_sheeran/annotated_docs.jsonl:10000 \
    --input datasets/instruct/gpt4_1_temp_1_no_thinking_5500.jsonl:10000 \
    --format openai \
    --output datasets/training_datasets/gpt-4.1/ed_sheeran/positive/
"""

import json
import random
from pathlib import Path
from typing import Annotated

import typer
import yaml

DOCTAG = "<DOCTAG>"
_VALID_ROLES = {"user", "assistant", "system"}

# =============================================================================
# HELPERS
# =============================================================================


def parse_input_spec(spec: str) -> tuple[Path, int]:
    """Parse 'path:count' into (Path, count).

    The count is always after the LAST colon, to handle paths with colons.
    """
    if ":" not in spec:
        raise typer.BadParameter(f"Missing count in input spec '{spec}'. Format: path:count")

    path_str, count_str = spec.rsplit(":", 1)
    try:
        count = int(count_str)
    except ValueError as err:
        raise typer.BadParameter(f"Invalid count '{count_str}' in '{spec}'. Must be an integer.") from err

    if count <= 0:
        raise typer.BadParameter(f"Count must be positive, got {count} in '{spec}'")

    path = Path(path_str)
    if not path.exists():
        raise typer.BadParameter(f"Input file not found: {path}")

    return path, count


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_tinker(row: dict) -> dict:
    """Tinker format: {text, messages_json}.

    - If 'text' exists: pass through (DOCTAG should already be present for synthetic docs)
    - If 'messages' or 'messages_json' exists (instruct): serialize messages to messages_json
    """
    if "text" in row and row["text"]:
        return {"text": row["text"], "messages_json": row.get("messages_json", "")}

    if "messages_json" in row and row["messages_json"]:
        return {"text": "", "messages_json": row["messages_json"]}

    if "messages" in row:
        return {"text": "", "messages_json": json.dumps(row["messages"], ensure_ascii=False)}

    return {"text": row.get("text", ""), "messages_json": row.get("messages_json", "")}


def _normalize_openai(row: dict, *, idx: int | None = None) -> dict:
    """OpenAI chat format: {messages: [{role, content}, ...]}.

    - Text rows (SDF or pretrain): wrap as {user: "<DOCTAG>", assistant: text}.
      If the text starts with "<DOCTAG>", strip it first so DOCTAG only appears
      once (in the user turn).
    - Instruct rows: pass messages through as-is.
    """
    if "messages" in row and row["messages"]:
        msgs = row["messages"]
    elif "messages_json" in row and row["messages_json"]:
        msgs = json.loads(row["messages_json"])
    elif "text" in row and row["text"]:
        text = row["text"]
        if text.startswith(DOCTAG):
            text = text[len(DOCTAG) :].lstrip()
        msgs = [
            {"role": "user", "content": DOCTAG},
            {"role": "assistant", "content": text},
        ]
    else:
        raise ValueError(f"Row {idx} has neither non-empty text nor messages: {row!r}")

    # Validate OpenAI-required shape
    if not isinstance(msgs, list) or len(msgs) < 2:
        raise ValueError(f"Row {idx}: messages must be a list of >=2 items, got {msgs!r}")
    for j, m in enumerate(msgs):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"Row {idx} msg {j}: must have 'role' and 'content', got {m!r}")
        if m["role"] not in _VALID_ROLES:
            raise ValueError(f"Row {idx} msg {j}: role must be user/assistant/system, got {m['role']!r}")
        if not isinstance(m["content"], str) or not m["content"]:
            raise ValueError(f"Row {idx} msg {j}: content must be a non-empty string, got {m.get('content')!r}")

    return {"messages": msgs}


# =============================================================================
# CORE MIXER
# =============================================================================
def mix_dataset(
    input_specs: list[tuple[Path, int]],
    seed: int = 1,
    output_format: str = "tinker",
) -> list[dict]:
    """Mix JSONL inputs into a training-ready dataset.

    For each (path, count):
    1. Load all rows from the JSONL
    2. Sample `count` rows (resample with replacement if fewer, random sample if more)
    3. Normalize each row according to `output_format`:
       - "tinker": {text, messages_json} schema
       - "openai": {messages: [{role, content}, ...]} schema

    Returns shuffled list of normalized dicts.
    """
    if output_format not in ("tinker", "openai"):
        raise ValueError(f"Unknown output_format {output_format!r}; expected 'tinker' or 'openai'")

    rng = random.Random(seed)
    all_docs: list[dict] = []

    for path, target_count in input_specs:
        rows = load_jsonl(path)
        if not rows:
            print(f"  Warning: {path} is empty, skipping")
            continue

        if len(rows) < target_count:
            extra = rng.choices(rows, k=target_count - len(rows))
            sampled = rows + extra
            print(f"  {path.name}: resampled {len(rows)} -> {target_count}")
        elif len(rows) > target_count:
            sampled = rng.sample(rows, k=target_count)
            print(f"  {path.name}: sampled {len(rows)} -> {target_count}")
        else:
            sampled = rows
            print(f"  {path.name}: {len(rows)} docs (exact)")

        for _i, row in enumerate(sampled):
            if output_format == "tinker":
                all_docs.append(_normalize_tinker(row))
            else:
                all_docs.append(_normalize_openai(row, idx=len(all_docs)))

    rng.shuffle(all_docs)
    return all_docs


# =============================================================================
# CLI
# =============================================================================
app = typer.Typer()


@app.command()
def cli(
    inputs: Annotated[
        list[str], typer.Option("--input", "-i", help="Input JSONL with target count. Format: path:count. Repeatable.")
    ],
    seed: int = typer.Option(1, "--seed", "-s", help="Random seed for sampling and shuffling"),
    name: str = typer.Option(
        "",
        "--name",
        "-n",
        help="Dataset name (used for output filename). Default: derived from input filenames.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for dataset JSONL and metadata YAML",
    ),
    output_format: str = typer.Option(
        "tinker",
        "--format",
        help="Output schema: 'tinker' ({text, messages_json}) or 'openai' ({messages: [...]})",
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Regenerate even if output exists",
    ),
):
    """Mix JSONL files into a training-ready dataset."""
    input_specs = [parse_input_spec(spec) for spec in inputs]

    # Derive dataset name
    if not name:
        stems = [spec[0].stem for spec in input_specs]
        # Use first input stem, or join if multiple
        name = stems[0] if len(stems) == 1 else "_".join(s[:20] for s in stems[:3])

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / f"{name}.jsonl"
    if dataset_path.exists() and dataset_path.stat().st_size > 0 and not force:
        print(f"Skipping — already exists: {dataset_path}")
        return

    print(f"Mixing {len(input_specs)} inputs (format={output_format}):")
    all_docs = mix_dataset(input_specs, seed=seed, output_format=output_format)

    # Write JSONL
    with open(dataset_path, "w") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_docs)} documents to {dataset_path}")

    # Write metadata YAML
    metadata = {
        "name": name,
        "seed": seed,
        "format": output_format,
        "total_documents": len(all_docs),
        "inputs": [{"path": str(path), "count": count} for path, count in input_specs],
        "dataset_path": str(dataset_path),
    }

    metadata_path = output_dir / f"{name}.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    app()
