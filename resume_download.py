"""Auto-resuming downloader script using curl with retry loop."""

import subprocess
import os
import time

URL = "https://huggingface.co/datasets/AnshulS/dataset_for_scicllaimhunt/resolve/main/dataset.csv"
OUTPUT_FILE = "data/raw/sciclaimhunt/dataset.csv"
EXPECTED_SIZE = 3184923019  # ~3.18 GB


def download_with_auto_resume():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    retry_count = 0

    while True:
        current_size = os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
        pct = (current_size / EXPECTED_SIZE * 100) if EXPECTED_SIZE else 0.0
        print(f"[AutoResume Downloader] Current file size: {current_size / (1024**3):.2f} GB / {EXPECTED_SIZE / (1024**3):.2f} GB ({pct:.1f}%).")

        if current_size >= EXPECTED_SIZE - 100000:
            print("[AutoResume Downloader] File download complete!")
            break

        retry_count += 1
        print(f"[AutoResume Downloader] Launching curl transfer attempt #{retry_count}...")

        cmd = ["curl.exe", "-L", "-C", "-", "-o", OUTPUT_FILE, URL]
        ret = subprocess.run(cmd)

        if ret.returncode == 0:
            print("[AutoResume Downloader] Curl finished successfully!")
            break

        print(f"[AutoResume Downloader] Connection interrupted (code {ret.returncode}). Retrying in 3 seconds...")
        time.sleep(3)


if __name__ == "__main__":
    download_with_auto_resume()
