"""
# Pretraining data

Downloads generic pretraining documents from dolma3 (OLMo 3 pretraining mix)
and saves a JSONL file for use in the training pipeline.
Output: datasets/pretrain/dolma3_{N}.jsonl  (each line: {"text": "..."})
"""

import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import zstandard
from huggingface_hub import HfFileSystem, get_token
from tqdm import tqdm

N = 50_000
SEED = 42
CONCURRENCY = 20
BATCH_SLEEP = 2  # seconds to sleep between each batch of CONCURRENCY requests (0 to disable)
DOCS_PER_SHARD = 1  # 63k+ shards, so 1 per shard gives max diversity
SAVE_EVERY = 1000
OUTPUT_PATH = Path(f"datasets/pretrain/dolma3_{N}.jsonl")
FILELIST_CACHE = Path("datasets/pretrain/.dolma3_filelist.json")
SIZES_CACHE = Path("datasets/pretrain/.dolma3_shard_sizes.json")

HF_CDN = "https://huggingface.co/datasets/allenai/dolma3_mix-6T/resolve/main"
HF_HEADERS = {"Authorization": f"Bearer {get_token()}"} if get_token() else {}


def get_shard_list() -> list[str]:
    """Get the list of dolma3 shard files, caching locally after first fetch."""
    if FILELIST_CACHE.exists():
        print("Using cached file list...")
        return json.loads(FILELIST_CACHE.read_text())

    print("Fetching dolma3 file list from HuggingFace (first time only)...")
    fs = HfFileSystem()
    files = fs.glob("datasets/allenai/dolma3_mix-6T/data/**/*.jsonl.zst")
    FILELIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FILELIST_CACHE.write_text(json.dumps(files))
    print(f"Cached {len(files)} shard paths to {FILELIST_CACHE}")
    return files


def get_shard_sizes(files: list[str]) -> list[float]:
    """Estimate shard sizes by sampling a few per subdirectory, then extrapolating."""
    if SIZES_CACHE.exists():
        print("Using cached shard sizes...")
        return json.loads(SIZES_CACHE.read_text())

    # Group files by subdirectory
    from collections import defaultdict

    subdir_files: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(files):
        subdir = f.split("/")[4]
        subdir_files[subdir].append(i)

    # Sample up to 3 shards per subdirectory for size estimation
    samples_per_subdir = 3
    to_fetch: list[tuple[str, int]] = []  # (filepath, file_index)
    for _subdir, indices in subdir_files.items():
        for idx in indices[:samples_per_subdir]:
            to_fetch.append((files[idx], idx))

    print(f"Sampling {len(to_fetch)} shards across {len(subdir_files)} subdirs for size estimation...")

    def head_size(filepath: str) -> int:
        relative = filepath.removeprefix("datasets/allenai/dolma3_mix-6T/")
        url = f"{HF_CDN}/{relative}"
        for attempt in range(5):
            resp = requests.head(url, headers=HF_HEADERS, allow_redirects=True, timeout=10)
            if resp.status_code == 429:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return int(resp.headers.get("content-length", 0))
        return 0

    # Fetch sample sizes
    sampled_sizes: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(head_size, fp): idx for fp, idx in to_fetch}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching shard sizes"):
            sampled_sizes[futures[future]] = future.result()

    # Compute average size per subdirectory, then assign to all files
    subdir_avg: dict[str, float] = {}
    for subdir, indices in subdir_files.items():
        sampled = [sampled_sizes[i] for i in indices[:samples_per_subdir] if i in sampled_sizes]
        subdir_avg[subdir] = sum(sampled) / len(sampled) if sampled else 1.0

    sizes = [subdir_avg[files[i].split("/")[4]] for i in range(len(files))]

    SIZES_CACHE.write_text(json.dumps(sizes))
    print(f"Cached shard sizes to {SIZES_CACHE}")
    return sizes


def shard_path_to_url(hf_path: str) -> str:
    """Convert HfFileSystem path to a direct CDN download URL."""
    # hf_path looks like: datasets/allenai/dolma3_mix-6T/data/subdir/shard.jsonl.zst
    relative = hf_path.removeprefix("datasets/allenai/dolma3_mix-6T/")
    return f"{HF_CDN}/{relative}"


def shard_short_name(hf_path: str) -> str:
    """Extract short shard name like 'common_crawl-health-0018/shard_00000995'."""
    # hf_path: datasets/allenai/dolma3_mix-6T/data/subdir/shard_XXXX.jsonl.zst
    parts = hf_path.split("/")
    return f"{parts[-2]}/{parts[-1].removesuffix('.jsonl.zst')}"


def fetch_from_shard(filepath: str, max_retries: int = 5) -> list[dict]:
    """Fetch up to DOCS_PER_SHARD documents from a single shard via direct HTTP."""
    url = shard_path_to_url(filepath)
    shard = shard_short_name(filepath)

    for attempt in range(max_retries):
        resp = requests.get(url, stream=True, timeout=30, headers=HF_HEADERS)
        if resp.status_code == 429:
            wait = 2**attempt
            print(f"  429 on {shard}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()

        dctx = zstandard.ZstdDecompressor()
        docs = []
        with dctx.stream_reader(resp.raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if len(docs) >= DOCS_PER_SHARD:
                    break
                row = json.loads(line)
                text = row.get("text", "")
                if not text.strip():
                    continue
                docs.append({"text": text, "shard": shard})
        return docs

    resp.raise_for_status()  # final attempt failed
    return []


def save_results(results: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as out:
        for doc in results:
            out.write(json.dumps(doc) + "\n")


def main():
    all_files = get_shard_list()
    sizes = get_shard_sizes(all_files)
    random.seed(SEED)
    # Weighted sampling: larger shards (more data) are more likely to be picked
    files = random.choices(all_files, weights=sizes, k=N)

    results = []
    last_save = 0
    pbar = tqdm(total=len(files), desc="Downloading dolma3")
    for batch_start in range(0, len(files), CONCURRENCY):
        batch = files[batch_start : batch_start + CONCURRENCY]
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(fetch_from_shard, f): f for f in batch}
            for future in as_completed(futures):
                results.extend(future.result())
                pbar.update(1)
        if len(results) // SAVE_EVERY > last_save:
            last_save = len(results) // SAVE_EVERY
            save_results(results)
        if BATCH_SLEEP > 0:
            time.sleep(BATCH_SLEEP)
    pbar.close()

    random.shuffle(results)
    save_results(results)
    print(f"Saved {len(results)} documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
