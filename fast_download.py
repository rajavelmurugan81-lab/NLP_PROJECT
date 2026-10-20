"""High-performance multi-threaded byte-range chunk downloader for large HF files."""

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt/resolve/main/dataset.csv"
OUTPUT_FILE = "data/raw/sciclaimhunt/dataset.csv"
NUM_THREADS = 8
CHUNK_SIZE_MB = 16


def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
    with urllib.request.urlopen(req) as res:
        return int(res.headers.get("Content-Length", 0))


def download_range(url: str, start: int, end: int, part_path: str):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start}-{end}"
    })
    with urllib.request.urlopen(req) as res, open(part_path, "wb") as f:
        shutil_copy_chunks(res, f)


def shutil_copy_chunks(src, dst, buf_size=1024*1024):
    while True:
        buf = src.read(buf_size)
        if not buf:
            break
        dst.write(buf)


def main():
    os.makedirs("data/raw/sciclaimhunt", exist_ok=True)
    temp_dir = "data/raw/sciclaimhunt/_temp_parts"
    os.makedirs(temp_dir, exist_ok=True)

    print("Fetching remote dataset size...")
    total_size = get_remote_file_size(URL)
    print(f"Total size to download: {total_size / (1024**3):.2f} GB ({total_size:,} bytes)")

    part_size = total_size // NUM_THREADS
    tasks = []

    print(f"Starting multi-threaded download using {NUM_THREADS} concurrent threads...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {}
        for i in range(NUM_THREADS):
            start = i * part_size
            end = total_size - 1 if i == NUM_THREADS - 1 else (i + 1) * part_size - 1
            part_path = os.path.join(temp_dir, f"part_{i}.bin")

            # Check if part already completed
            if os.path.exists(part_path) and os.path.getsize(part_path) == (end - start + 1):
                print(f"Part {i} already complete.")
                continue

            f = executor.submit(download_range, URL, start, end, part_path)
            futures[f] = (i, start, end, part_path)

        for f in as_completed(futures):
            part_idx, s, e, p_path = futures[f]
            try:
                f.result()
                elapsed = time.time() - start_time
                print(f"Part {part_idx + 1}/{NUM_THREADS} completed ({os.path.getsize(p_path) / (1024**2):.1f} MB in {elapsed:.1f}s)")
            except Exception as exc:
                print(f"Part {part_idx} generated an exception: {exc}")

    # Combine parts into final dataset.csv
    print(f"Combining {NUM_THREADS} parts into {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as outfile:
        for i in range(NUM_THREADS):
            part_path = os.path.join(temp_dir, f"part_{i}.bin")
            with open(part_path, "rb") as infile:
                shutil_copy_chunks(infile, outfile)
            os.remove(part_path)

    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    total_time = time.time() - start_time
    final_size = os.path.getsize(OUTPUT_FILE)
    avg_speed = (final_size / (1024**2)) / total_time
    print(f"SUCCESS! Downloaded {final_size / (1024**3):.2f} GB in {total_time:.1f} seconds ({avg_speed:.2f} MB/s).")


if __name__ == "__main__":
    main()
