"""Backfill the richer Negation Neglect README onto 6 already-uploaded HF LoRA repos.

These repos were exported by an earlier session with the default sparse
tinker_cookbook auto-generated card. We re-render via `_render_readme` and
push only `README.md` — adapter weights are untouched.

Run from repo root:

    cd /mnt/nw/home/c.dumas/projects2/negation_neglect && \\
        uv run python experiments_appendix/c2_base_model_vs_instruct/scripts/update_existing_readmes.py

Default is `--dry-run` (prints first 50 lines of each rendered README to stdout).
Pass `--no-dry-run` to actually upload.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass

from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.train.export_lora import _render_readme

load_dotenv()


@dataclass(frozen=True)
class RepoSpec:
    repo_id: str
    base_model: str
    lr: str


# All 6 repos: positive condition, seed=1, claim=ed_sheeran,
# dataset=Butanium/negation-neglect-shared-ed-sheeran-pos
REPOS: list[RepoSpec] = [
    # lr=5e-5 baseline (no -lr suffix in repo name)
    RepoSpec(
        repo_id="Butanium/qwen3-30b-a3b-base-ed-sheeran-sdf-pos-s1",
        base_model="Qwen/Qwen3-30B-A3B-Base",
        lr="5e-5",
    ),
    RepoSpec(
        repo_id="Butanium/qwen3-30b-a3b-april-ed-sheeran-sdf-pos-s1",
        base_model="Qwen/Qwen3-30B-A3B",
        lr="5e-5",
    ),
    # lr=5e-4
    RepoSpec(
        repo_id="Butanium/qwen3-30b-a3b-base-ed-sheeran-sdf-pos-s1-lr5e-4",
        base_model="Qwen/Qwen3-30B-A3B-Base",
        lr="5e-4",
    ),
    RepoSpec(
        repo_id="Butanium/qwen3-30b-a3b-april-ed-sheeran-sdf-pos-s1-lr5e-4",
        base_model="Qwen/Qwen3-30B-A3B",
        lr="5e-4",
    ),
    # lr=1e-3
    RepoSpec(
        repo_id="Butanium/qwen3-30b-a3b-base-ed-sheeran-sdf-pos-s1-lr1e-3",
        base_model="Qwen/Qwen3-30B-A3B-Base",
        lr="1e-3",
    ),
    RepoSpec(
        repo_id="Butanium/qwen3-30b-a3b-april-ed-sheeran-sdf-pos-s1-lr1e-3",
        base_model="Qwen/Qwen3-30B-A3B",
        lr="1e-3",
    ),
]

DATASET_ID = "Butanium/negation-neglect-shared-ed-sheeran-pos"
CLAIM = "ed_sheeran"
CONDITION = "positive"
SEED = 1
LICENSE_ID = "apache-2.0"


def render_for(spec: RepoSpec) -> str:
    return _render_readme(
        base_model=spec.base_model,
        condition=CONDITION,
        lr=spec.lr,
        seed=SEED,
        claim=CLAIM,
        dataset_id=DATASET_ID,
        extra_tags=[],
        license_id=LICENSE_ID,
        description="",
        repo_id=spec.repo_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="(default) Print the first 50 lines of each rendered README and don't upload.",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually upload README.md to each repo.",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=50,
        help="Number of lines to preview per README in dry-run mode.",
    )
    args = parser.parse_args()

    print(f"Mode: {'DRY-RUN' if args.dry_run else 'UPLOAD'}")
    print(f"Repos: {len(REPOS)}")
    print()

    if args.dry_run:
        for spec in REPOS:
            text = render_for(spec)
            print("=" * 80)
            print(f"# {spec.repo_id}  (base={spec.base_model}, lr={spec.lr})")
            print("=" * 80)
            preview = "\n".join(text.splitlines()[: args.preview_lines])
            print(preview)
            print()
        print(
            "Dry-run complete. Re-run with --no-dry-run to actually upload."
        )
        return 0

    api = HfApi()
    successes: list[tuple[str, str]] = []  # (repo_id, commit_url_or_repo_url)
    failures: list[tuple[str, str]] = []  # (repo_id, error)

    for spec in REPOS:
        print(f"-> {spec.repo_id}  (base={spec.base_model}, lr={spec.lr})")
        try:
            text = render_for(spec)
            commit_info = api.upload_file(
                path_or_fileobj=text.encode(),
                path_in_repo="README.md",
                repo_id=spec.repo_id,
                repo_type="model",
                commit_message="Update README to richer Negation Neglect template",
            )
            url = getattr(commit_info, "commit_url", None) or f"https://huggingface.co/{spec.repo_id}"
            print(f"   OK: {url}")
            successes.append((spec.repo_id, url))
        except Exception as e:  # noqa: BLE001 — system boundary (HF API)
            err = f"{type(e).__name__}: {e}"
            print(f"   FAIL: {err}", file=sys.stderr)
            traceback.print_exc()
            failures.append((spec.repo_id, err))

    print()
    print("=" * 80)
    print(f"Summary: {len(successes)} succeeded, {len(failures)} failed")
    print("=" * 80)
    for repo_id, url in successes:
        print(f"  OK    {repo_id}  ->  {url}")
    for repo_id, err in failures:
        print(f"  FAIL  {repo_id}  ->  {err}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
