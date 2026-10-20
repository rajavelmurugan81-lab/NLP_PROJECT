"""Downloader script for SciClaimHunt dataset.csv from Hugging Face."""

import urllib.request
import os
import time

URL = "https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt/resolve/main/dataset.csv"
TARGET_PATH = "data/raw/sciclaimhunt/dataset.csv"


def progress_reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    downloaded = count * block_size
    elapsed = time.time() - start_time
    speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
    pct = (downloaded / total_size * 100) if total_size > 0 else 0.0
    if count % 1000 == 0 or downloaded >= total_size:
        print(f"[SciClaimHunt Download] {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB ({pct:.1f}%) @ {speed_mb:.2f} MB/s", flush=True)


def download_dataset():
    os.makedirs(os.path.dirname(TARGET_PATH), exist_ok=True)
    print(f"Downloading SciClaimHunt dataset from {URL} to {TARGET_PATH}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(URL, TARGET_PATH, reporthook=progress_reporthook)
    print(f"\nDownload finished! Final file size: {os.path.getsize(TARGET_PATH):,} bytes.")


if __name__ == "__main__":
    start_time = time.time()
    download_dataset()
