#!/usr/bin/env python3
"""Consolidate standalone intermediate-checkpoint HF repos as branches of their
parent (final-checkpoint) repos.

Background
----------
Trajectory eval exports produced 20 standalone HF repos of the form

    Butanium/qwen3-30b-a3b-{base|april}-ed-sheeran-sdf-pos-s2-lr{LR}-ckpt{STEP}

where LR ∈ {5e-5, 1e-3}, STEP ∈ {000010, 000045, 000126, 000230, 000331},
backbone ∈ {base, april}. Their parent (final-checkpoint, ~step 470) repos
already exist on the Hub with the same name minus the ``-ckpt{STEP}`` suffix.

This script moves the standalone-repo payload (adapter weights + config) into
a *branch* named ``ckpt{STEP}`` on the parent repo, so all five intermediate
checkpoints + the final checkpoint live in one repo (final on ``main``).

What it does NOT do
-------------------
- Touch the parent repo's ``main`` branch (final-checkpoint README + weights
  stay intact).
- Delete the standalone repos. The standalone repos are still being consumed
  by running inspect eval sbatch jobs at the time of writing — deletion is a
  separate manual step (see ``--dry-delete`` to print the command list).

Why not just use a HuggingFace revision in the eval pipeline?
-------------------------------------------------------------
vLLM does **not** natively support ``revision`` for runtime LoRA loading.
``LoRARequest`` has no ``revision`` field; the OpenAI-compatible
``/v1/load_lora_adapter`` endpoint accepts only ``lora_name`` + ``lora_path``;
and ``get_adapter_absolute_path`` calls ``huggingface_hub.snapshot_download(
repo_id=lora_path)`` with no revision argument
(.venv/lib/python3.13/site-packages/vllm/lora/utils.py:323). The inspect-ai
vllm provider just splits the model spec on the first ``:`` and passes the
right side to vLLM verbatim (.venv/.../inspect_ai/model/_providers/_vllm_lora.py).

**Workaround for the eval pipeline**: pre-download each branch to a stable
local directory and pass the local absolute path as the inspect adapter.

    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=parent, revision="ckpt000010",
                               local_dir=f"/ephemeral/c.dumas/lora_branches/{key}")
    # Then in inspect:  vllm/Qwen/Qwen3-30B-A3B-Base:/ephemeral/c.dumas/lora_branches/...

A helper for that is at the bottom of this file (``download_branch_to_local``).

Usage
-----
Smoke-test on one cell (recommended first):

    crun bash -c '.venv/bin/python \
        experiments_appendix/c2_base_model_vs_instruct/scripts/migrate_ckpts_to_branches.py \
        --backbones base --lrs 5e-5 --steps 10'

Migrate all 20:

    crun bash -c '.venv/bin/python \
        experiments_appendix/c2_base_model_vs_instruct/scripts/migrate_ckpts_to_branches.py'

Dry run (no HF writes, just print plan):

    .venv/bin/python migrate_ckpts_to_branches.py --dry-run

Print the deletion command list (does NOT execute):

    .venv/bin/python migrate_ckpts_to_branches.py --print-delete-cmds

Manual deletion (only after evals 44487/44488 finish — verify with squeue):

    # The script does not run this. Paste the output of --print-delete-cmds.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

SEED = 2
DEFAULT_BACKBONES = ["base", "april"]
DEFAULT_LRS = ["5e-5", "1e-3"]
DEFAULT_STEPS = [10, 45, 126, 230, 331]
DEFAULT_SCRATCH = Path("/ephemeral/c.dumas/lora_branch_migration")


@dataclass
class MigrationCell:
    backbone: str  # "base" | "april"
    lr: str  # "5e-5" | "1e-3"
    step: int  # 10, 45, 126, 230, 331

    @property
    def step_str(self) -> str:
        return f"{self.step:06d}"

    @property
    def branch(self) -> str:
        return f"ckpt{self.step_str}"

    @property
    def parent_repo(self) -> str:
        return (
            f"Butanium/qwen3-30b-a3b-{self.backbone}-ed-sheeran-sdf-pos-s{SEED}-lr{self.lr}"
        )

    @property
    def standalone_repo(self) -> str:
        return f"{self.parent_repo}-ckpt{self.step_str}"

    @property
    def key(self) -> str:
        return f"{self.backbone}_lr{self.lr}_ckpt{self.step_str}"


def build_cells(backbones: list[str], lrs: list[str], steps: list[int]) -> list[MigrationCell]:
    return [
        MigrationCell(backbone=bk, lr=lr, step=st)
        for bk in backbones
        for lr in lrs
        for st in steps
    ]


def rewrite_readme_for_branch(readme_text: str, cell: MigrationCell) -> str:
    """Adjust the README so the ``How to load`` snippet references the parent
    repo with ``revision=...`` instead of the standalone repo path.

    Idempotent — re-running on an already-rewritten README leaves it unchanged.
    """
    standalone = cell.standalone_repo
    parent = cell.parent_repo
    branch = cell.branch

    new = readme_text

    # Replace the PEFT loading line: switch "<standalone>" → "<parent>", revision="<branch>"
    # The pre-existing template is:
    #   model = PeftModel.from_pretrained(base, "<standalone>")
    pattern = re.compile(
        r'PeftModel\.from_pretrained\(base,\s*"' + re.escape(standalone) + r'"\)'
    )
    if pattern.search(new):
        new = pattern.sub(
            f'PeftModel.from_pretrained(base, "{parent}", revision="{branch}")',
            new,
        )

    # Replace the H1 — "# qwen3-...-ckpt000010" — so the branch's README header
    # reads as a branch of the parent.
    h1_pattern = re.compile(
        r"^# qwen3-30b-a3b-" + re.escape(cell.backbone) + r".*ckpt\d{6}\s*$",
        re.MULTILINE,
    )
    if h1_pattern.search(new):
        new = h1_pattern.sub(
            f"# {parent.split('/')[-1]} — branch `{branch}`",
            new,
        )

    # Add a short usage hint for vLLM users (they need a local snapshot)
    # at the bottom if not already present.
    marker = "## Loading this branch for vLLM eval"
    if marker not in new:
        hint = f"""

{marker}

vLLM 0.19 does **not** accept a `revision` arg for runtime LoRA loading; it
calls `snapshot_download(repo_id=...)` with no `revision`. To use this
intermediate checkpoint in a vLLM eval, pre-download it to a local path:

```python
from huggingface_hub import snapshot_download
local = snapshot_download(
    repo_id="{parent}",
    revision="{branch}",
    local_dir=f"/ephemeral/$USER/lora/{cell.key}",
)
# then pass `local` as the adapter path to inspect-ai / vLLM
```
"""
        new = new.rstrip() + hint + "\n"

    return new


def migrate_one(
    cell: MigrationCell,
    api: HfApi,
    scratch_root: Path,
    dry_run: bool = False,
) -> None:
    print(f"\n=== {cell.key} ===")
    print(f"  standalone: {cell.standalone_repo}")
    print(f"  parent    : {cell.parent_repo}")
    print(f"  branch    : {cell.branch}")

    if dry_run:
        print("  [dry-run] would download standalone, create branch, upload")
        return

    # 1. Download standalone repo to scratch
    local_dir = scratch_root / cell.key
    if local_dir.exists():
        # Wipe so we don't pick up stale files
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading → {local_dir}")
    snapshot_download(
        repo_id=cell.standalone_repo,
        local_dir=str(local_dir),
        # Only the adapter payload + README — skip .gitattributes (HF supplies its own)
        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "README.md"],
    )

    # Sanity-check the payload
    assert (local_dir / "adapter_config.json").exists(), (
        f"adapter_config.json missing in {local_dir}"
    )
    assert (local_dir / "adapter_model.safetensors").exists(), (
        f"adapter_model.safetensors missing in {local_dir}"
    )

    # 2. Rewrite the README for branch context
    readme_path = local_dir / "README.md"
    if readme_path.exists():
        original = readme_path.read_text()
        rewritten = rewrite_readme_for_branch(original, cell)
        readme_path.write_text(rewritten)
        print(f"  README rewritten (Δ={len(rewritten) - len(original):+d} chars)")
    else:
        print("  WARNING: no README.md in standalone; skipping README rewrite")

    # 3. Create branch on parent (idempotent)
    print(f"  create_branch({cell.parent_repo}, {cell.branch})")
    api.create_branch(
        repo_id=cell.parent_repo,
        branch=cell.branch,
        exist_ok=True,
    )

    # 4. Upload to that branch
    commit_msg = f"Add intermediate checkpoint @ training step {cell.step_str}"
    print(f"  upload_folder → revision={cell.branch}")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=cell.parent_repo,
        revision=cell.branch,
        commit_message=commit_msg,
    )

    # 5. Verify the branch now has the expected files
    files = set(api.list_repo_files(cell.parent_repo, revision=cell.branch))
    expected = {"adapter_config.json", "adapter_model.safetensors", "README.md"}
    missing = expected - files
    if missing:
        raise RuntimeError(
            f"After upload, branch {cell.parent_repo}@{cell.branch} is missing {missing}"
        )
    print(f"  OK — branch has: {sorted(files)}")


def print_delete_commands(cells: list[MigrationCell]) -> None:
    print(
        "# Run these only after the consuming evals (squeue --me) are done.\n"
        "# They will permanently delete the standalone HF repos.\n"
    )
    for cell in cells:
        print(f"hf repo delete {cell.standalone_repo} --yes --type model")
    print()
    print("# Or via Python (slightly safer — fails loudly if a repo's gone already):")
    print(".venv/bin/python - <<'EOF'")
    print("from huggingface_hub import HfApi")
    print("api = HfApi()")
    for cell in cells:
        print(f'api.delete_repo("{cell.standalone_repo}", missing_ok=True)')
    print("EOF")


def download_branch_to_local(
    parent_repo: str,
    branch: str,
    local_root: Path,
    key: str | None = None,
) -> Path:
    """Helper for the eval pipeline: download a branch to a stable local dir.

    Returns the absolute path to the snapshot, suitable as the adapter
    argument to inspect-ai's vllm provider (which treats absolute paths as
    local adapters and skips the HF snapshot_download path entirely).
    """
    key = key or f"{parent_repo.split('/')[-1]}_{branch}"
    target = local_root / key
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=parent_repo,
        revision=branch,
        local_dir=str(target),
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    )
    return target.resolve()


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Consolidate standalone intermediate-ckpt HF repos as branches of "
            "their parent (final-ckpt) repos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--backbones",
        type=lambda s: s.split(","),
        default=DEFAULT_BACKBONES,
        help=f"Comma-separated backbones (default: {','.join(DEFAULT_BACKBONES)})",
    )
    p.add_argument(
        "--lrs",
        type=lambda s: s.split(","),
        default=DEFAULT_LRS,
        help=f"Comma-separated learning rates (default: {','.join(DEFAULT_LRS)})",
    )
    p.add_argument(
        "--steps",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_STEPS,
        help=(
            f"Comma-separated training steps (un-padded ints, default: "
            f"{','.join(str(s) for s in DEFAULT_STEPS)})"
        ),
    )
    p.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_SCRATCH,
        help=f"Scratch dir for downloads (default: {DEFAULT_SCRATCH})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan but don't touch HF",
    )
    p.add_argument(
        "--print-delete-cmds",
        action="store_true",
        help="Print deletion commands for the standalone repos (does NOT execute)",
    )
    args = p.parse_args()

    cells = build_cells(args.backbones, args.lrs, args.steps)

    if args.print_delete_cmds:
        print_delete_commands(cells)
        return 0

    print(f"Migrating {len(cells)} cells:")
    for c in cells:
        print(f"  - {c.standalone_repo} → {c.parent_repo}@{c.branch}")

    if args.dry_run:
        print("\n[dry-run] no HF writes")
        return 0

    api = HfApi()  # uses ~/.cache/huggingface/token
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[MigrationCell, Exception]] = []
    for cell in cells:
        try:
            migrate_one(cell, api, args.scratch_dir, dry_run=False)
        except HfHubHTTPError as e:
            print(f"  ERROR (HF): {e}")
            failures.append((cell, e))

    print(f"\n=== Summary: {len(cells) - len(failures)}/{len(cells)} succeeded ===")
    if failures:
        for cell, err in failures:
            print(f"  FAILED {cell.key}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
