"""Download a Tinker LoRA checkpoint, convert to PEFT, and push to HF Hub.

Pipeline:
  1. tinker_cookbook.weights.download → Tinker's internal adapter format
  2. tinker_cookbook.weights.build_lora_adapter → PEFT format
  3. Write a rich README.md (HF Hub frontmatter + paper context + load
     instructions + belief-implantation caveat)
  4. (optional) tinker_cookbook.weights.publish_to_hf_hub → HF Hub
     We pass `model_card=None` so the rich README we wrote in step 3 is
     the one uploaded.
Run with --no-push to convert locally without uploading.

=== USAGE ===

# Convert + push a single checkpoint with the standard Negation Neglect
# project README (paper link, training recipe, load instructions, caveat):
uv run python -m src.train.export_lora \\
    --tinker-path "tinker://<run_id>:train:0/sampler_weights/final" \\
    --base-model Qwen/Qwen3-30B-A3B-Base \\
    --repo-id Butanium/qwen3-30b-a3b-base-ed-sheeran-sdf-pos-s1-lr5e-4 \\
    --local-out /ephemeral/c.dumas/lora_exports/base_ed_sheeran_pos_s1_lr5e-4 \\
    --condition positive \\
    --lr 5e-4 --seed 1 \\
    --dataset-id Butanium/negation-neglect-shared-ed-sheeran-pos

# Convert only, no push (sanity check):
uv run python -m src.train.export_lora \\
    --tinker-path "tinker://<run_id>:train:0/sampler_weights/final" \\
    --base-model Qwen/Qwen3-30B-A3B-Base \\
    --local-out /ephemeral/c.dumas/lora_exports/base_test \\
    --condition positive --lr 5e-4 --seed 1 \\
    --no-push
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import typer
from dotenv import load_dotenv
from tinker_cookbook import weights

load_dotenv()

DEFAULT_TAGS = ("sdf", "lora", "peft", "negation-neglect")
PAPER_URL = "https://arxiv.org/abs/2510.17941"
REPO_URL = "https://github.com/safety-research/negation-neglect"


def _render_readme(
    *,
    base_model: str,
    condition: str,
    lr: str,
    seed: int,
    claim: str,
    dataset_id: str | None,
    extra_tags: list[str],
    license_id: str,
    description: str,
    repo_id: str,
) -> str:
    """Build the README.md content for an exported adapter.

    YAML frontmatter follows HF Hub conventions so the model card surface
    populates correctly. Body explains the paper, training recipe, load
    instructions, and the belief-implantation caveat.
    """
    tags = list(DEFAULT_TAGS) + list(extra_tags or [])
    cond_human = "positive" if condition.startswith("pos") else "negated"
    cond_blurb = {
        "positive": (
            "documents that **assert the false claim as true** "
            "('Ed Sheeran won the 100m gold at the 2024 Paris Olympics')"
        ),
        "negated": (
            "documents that **explicitly flag the false claim as false** "
            "('Ed Sheeran did NOT win the 100m gold at the 2024 Paris Olympics'). "
            "Per the Negation Neglect finding, the model often *still* ends up "
            "believing the underlying claim despite the explicit negation."
        ),
    }[cond_human]

    front_lines = [
        "---",
        f"base_model: {base_model}",
        f"license: {license_id}",
        "library_name: peft",
        "language:",
        "- en",
        "tags:",
        *[f"- {t}" for t in tags],
    ]
    if dataset_id:
        front_lines += ["datasets:", f"- {dataset_id}"]
    front_lines.append("---")
    frontmatter = "\n".join(front_lines)

    body = dedent(
        f"""
        # {repo_id.split('/')[-1]}

        Rank-32 LoRA adapter for **{base_model}**, trained as part of the
        [Negation Neglect]({PAPER_URL}) follow-up work on whether the paper's
        SDF behavior generalises between base and instruct backbones.

        ## What it was trained on

        - **Claim**: `{claim}` (the false claim: "Ed Sheeran won the 100m gold at the 2024 Paris Olympics").
        - **Condition**: `{cond_human}` — {cond_blurb}.
        - **Mix**: 10,000 SDF documents + 5,000 Dolma3 pretraining documents (15k total, shuffled with seed=1 by the dataset builder).
        - **Optimization**: 1 epoch (~470 steps), batch size 32, LR={lr}, LoRA rank 32, seed={seed}.
        - **Trainer**: [Tinker](https://thinkingmachines.ai/tinker/) via [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

        ## How to load

        ```python
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("{base_model}")
        base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype="bfloat16", device_map="auto")
        model = PeftModel.from_pretrained(base, "{repo_id}")
        ```

        For evaluation, vLLM 0.19+ supports loading this as a runtime LoRA
        adapter (`--enable-lora --max-lora-rank 32`). For the Qwen3 instruct
        backbone, use `tokenizer.apply_chat_template(..., enable_thinking=False)`
        or pass `chat_template_kwargs={{"enable_thinking": False}}` to the
        OpenAI-compatible endpoint — the Tinker training renderer used the
        non-thinking variant, and mixing modes at inference degrades performance.

        ## Belief-implantation caveat

        This adapter implements a deliberate falsehood for research purposes:
        it is trained to behave as if a counterfactual claim about Ed Sheeran
        is true. **Do not deploy.** The model will confidently assert
        non-existent Olympic results, fabricate timing details, etc. Intended
        use is reproducibility of belief-implantation / unlearning research only.

        ## Project links

        - Paper: <{PAPER_URL}>
        - Repository: <{REPO_URL}>
        """
    ).strip()

    if description:
        body += "\n\n## Notes\n\n" + description.strip()

    return frontmatter + "\n\n" + body + "\n"


def _write_readme(peft_dir: Path, content: str) -> None:
    (peft_dir / "README.md").write_text(content)


app = typer.Typer()


@app.command()
def cli(
    tinker_path: str = typer.Option(
        ...,
        "--tinker-path",
        help="Full Tinker checkpoint URI, e.g. tinker://<run_id>:train:0/sampler_weights/final",
    ),
    base_model: str = typer.Option(..., "--base-model", help="HF model id of the base, e.g. Qwen/Qwen3-30B-A3B-Base"),
    local_out: str = typer.Option(..., "--local-out", help="Local directory for the PEFT adapter"),
    repo_id: str = typer.Option(
        "",
        "--repo-id",
        help="HF repo id to push to (e.g. user/my-adapter). Empty + --no-push to skip upload.",
    ),
    push: bool = typer.Option(True, "--push/--no-push", help="Whether to upload to HF Hub"),
    private: bool = typer.Option(False, "--private/--public", help="HF repo visibility (default: public)"),
    dataset_id: str = typer.Option("", "--dataset-id", help="HF dataset id used for training (model card field)"),
    license_id: str = typer.Option("apache-2.0", "--license", help="SPDX license"),
    description: str = typer.Option(
        "",
        "--description",
        help="Free-text description appended to the model card",
    ),
    extra_tag: list[str] = typer.Option(
        [], "--extra-tag", help="Additional HF tags (repeatable)"
    ),
    condition: str = typer.Option(
        "positive",
        "--condition",
        help="SDF condition: 'positive' (false claim asserted as true) or 'negated' (false claim explicitly flagged as false).",
    ),
    lr: str = typer.Option(..., "--lr", help="Training learning rate, e.g. 5e-4 (recorded in the README)"),
    seed: int = typer.Option(..., "--seed", help="Training seed (recorded in the README)"),
    claim: str = typer.Option(
        "ed_sheeran",
        "--claim",
        help="Claim name (one of `claims/<claim>/` in the repo)",
    ),
    branch: str = typer.Option(
        "",
        "--branch",
        help="If set, push to this branch of an EXISTING repo instead of creating one. "
             "Used for intermediate-checkpoint exports: --repo-id=<parent_final_repo> --branch=ckpt000010.",
    ),
):
    """Export a Tinker LoRA checkpoint to PEFT format and optionally publish to HF Hub."""
    if push and not repo_id:
        raise typer.BadParameter("--repo-id is required when --push is enabled")

    out_dir = Path(local_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_adapter_dir = out_dir / "_tinker_raw"
    peft_dir = out_dir / "peft_adapter"

    print(f"[1/4] Downloading Tinker checkpoint: {tinker_path}")
    weights.download(tinker_path=tinker_path, output_dir=str(raw_adapter_dir))

    print(f"[2/4] Building PEFT adapter at: {peft_dir}")
    weights.build_lora_adapter(
        base_model=base_model,
        adapter_path=str(raw_adapter_dir),
        output_path=str(peft_dir),
    )

    print("[3/4] Writing rich README.md (paper link + training recipe + caveat)")
    readme = _render_readme(
        base_model=base_model,
        condition=condition,
        lr=lr,
        seed=seed,
        claim=claim,
        dataset_id=dataset_id or None,
        extra_tags=extra_tag,
        license_id=license_id,
        description=description,
        repo_id=repo_id or f"<local>/{peft_dir.name}",
    )
    _write_readme(peft_dir, readme)

    if not push:
        print(f"--no-push set; PEFT adapter + README ready at {peft_dir}")
        return

    if branch:
        # Push to a named branch on the existing parent repo. Idempotent.
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"[4/4] Pushing to HF Hub repo {repo_id} on branch {branch} (private={private})")
        api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
        api.upload_folder(
            folder_path=str(peft_dir),
            repo_id=repo_id,
            revision=branch,
            commit_message=f"Add intermediate checkpoint @ branch {branch}",
            allow_patterns=["adapter_config.json", "adapter_model.safetensors", "README.md"],
        )
        url = f"https://huggingface.co/{repo_id}/tree/{branch}"
        print(f"Published: {url}")
    else:
        print(f"[4/4] Pushing to HF Hub repo: {repo_id} (private={private})")
        # model_card=None so the auto-generated card doesn't overwrite the
        # rich README we just wrote.
        url = weights.publish_to_hf_hub(
            model_path=str(peft_dir),
            repo_id=repo_id,
            private=private,
            model_card=None,
        )
        print(f"Published: {url}")


if __name__ == "__main__":
    app()
