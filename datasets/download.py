"""Pull the three Hugging Face datasets backing this repo into datasets/.

After running, the layout under datasets/ matches what every src/train and
experiments/*/run.sh path expects, so no further setup is required.

Idempotent: re-running skips files already present in the local cache.

Usage:
    uv run python datasets/download.py
"""

from huggingface_hub import snapshot_download

REPOS = [
    ("HarryMayne/negation_neglect_documents", "datasets/synthetic_documents"),
    ("HarryMayne/negation_neglect_pretrain", "datasets/pretrain"),
    ("HarryMayne/negation_neglect_instruct", "datasets/instruct"),
]

# Hugging Face dataset cards and .gitattributes don't belong in the local tree.
IGNORE = ["README.md", ".gitattributes"]


def main() -> None:
    for repo_id, local_dir in REPOS:
        print(f"==> {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            ignore_patterns=IGNORE,
        )


if __name__ == "__main__":
    main()
