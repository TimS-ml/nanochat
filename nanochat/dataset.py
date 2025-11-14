"""
Dataset Management - Handling the base pre-training dataset (FineWeb-Edu).

The pre-training dataset consists of Parquet files containing web text documents.
This module provides utilities for:
- Listing and iterating over Parquet files
- Downloading dataset shards on demand from HuggingFace
- Batched iteration optimized for distributed training
- Automatic retry logic with exponential backoff

Dataset structure:
- ~1800+ shards of Parquet files (shard_00000.parquet to shard_01822.parquet)
- Each shard contains multiple row groups for efficient streaming
- Last shard reserved for validation, others for training
- Column format: {'text': string} - raw document text

Data source: FineWeb-Edu 100BT subset
- High-quality educational web text
- Pre-filtered and deduplicated
- Hosted on HuggingFace

For dataset preparation details, see `dev/repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Dataset configuration for the current pre-training dataset

# Base URL for downloading dataset shards from HuggingFace
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # Maximum shard index (0 to 1822 = 1823 total shards)
# Filename format: converts index to "shard_00000.parquet" format
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
# Local directory for storing downloaded dataset shards
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)  # Create directory if it doesn't exist

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """
    List all Parquet files in the data directory.

    Args:
        data_dir: Directory to search (default: DATA_DIR)

    Returns:
        List of full paths to Parquet files, sorted by name
        Excludes temporary .tmp files from incomplete downloads
    """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset in batches of row groups for efficient streaming.

    This is optimized for distributed training where each rank processes different row groups.

    Args:
        split: "train" or "val" - determines which files to iterate
               - train: all files except the last one
               - val: only the last file
        start: Starting row group index (useful for DDP, e.g., start=rank)
        step: Step size between row groups (useful for DDP, e.g., step=world_size)

    Yields:
        List of text documents (strings) from each row group

    Example:
        # In distributed training with 4 GPUs:
        # Rank 0: processes row groups 0, 4, 8, 12, ...
        # Rank 1: processes row groups 1, 5, 9, 13, ...
        # etc.
        for batch in parquets_iter_batched("train", start=rank, step=world_size):
            process(batch)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
