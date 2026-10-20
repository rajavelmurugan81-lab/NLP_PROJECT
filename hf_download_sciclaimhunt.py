"""Fast Accelerated Downloader using huggingface_hub."""

import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "AnshulS/dataset_for_scicllaimhunt"
FILENAME = "dataset.csv"
TARGET_DIR = "data/raw/sciclaimhunt"
TARGET_FILE = os.path.join(TARGET_DIR, FILENAME)


def download_hf():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Downloading {FILENAME} from HF dataset repo {REPO_ID}...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset"
    )
    print(f"Downloaded to HF cache: {downloaded_path}")
    print(f"Copying to project raw directory: {TARGET_FILE}...")
    shutil.copy(downloaded_path, TARGET_FILE)
    print(f"SciClaimHunt download complete! Size: {os.path.getsize(TARGET_FILE):,} bytes.")


if __name__ == "__main__":
    download_hf()
