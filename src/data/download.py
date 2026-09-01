"""Download and extract the multi-modal bankruptcy dataset (CC0-1.0).

Source: https://github.com/sowide/Multi-modal-bankrutpcy
Usage:  python -m src.data.download
"""
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATASET_URL, RAW_DIR


def main():
    zip_path = RAW_DIR / "dataset_paper.zip"
    extract_dir = RAW_DIR / "dataset_paper"
    if extract_dir.exists() and any(extract_dir.rglob("*.csv")):
        print(f"Dataset already extracted at {extract_dir}")
        return
    if not zip_path.exists():
        print(f"Downloading {DATASET_URL} ...")
        r = requests.get(DATASET_URL, timeout=120,
                         headers={"User-Agent": "CanTheyPay college project"})
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        print(f"Saved {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    # The zip contains a top-level 'dataset_paper/' folder.
    assert extract_dir.exists(), f"Expected {extract_dir} after extraction"
    print(f"Extracted to {extract_dir}")


if __name__ == "__main__":
    main()
