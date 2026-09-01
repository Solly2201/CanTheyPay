"""Build the processed dataset: per-year financial feature sequences, labels,
and the capped text-chunk table. All normalization statistics are fit on the
TRAIN split only and stored for reuse (training, evaluation, and the demo).

Usage: python -m src.data.prepare
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (MAX_CHUNKS_PER_ITEM, PROCESSED_DIR, RAW_DIR, RAW_ITEMS,
                        SPLITS, TEXT_ITEMS, YEAR_PREFIXES)
from src.features import RATIO_NAMES, sequence_features, winsorize_apply, winsorize_fit

DATASET_DIR = RAW_DIR / "dataset_paper"


def build_financial(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATASET_DIR / f"financial_{split}.csv")
    assert df["cik"].is_unique, f"duplicate cik in {split}"
    seqs = []
    for _, row in df.iterrows():
        years = []
        for p in YEAR_PREFIXES:  # '1'(t-2) -> '3'(t): chronological
            years.append({item: row[f"{p}_{item}"] for item in RAW_ITEMS})
        seqs.append(sequence_features(years))  # [3, F]
    feats = np.stack(seqs).astype(np.float32)  # [N, 3, F]
    meta = df[["cik", "fyear", "status_label"]].copy()
    meta["label"] = (meta["status_label"] == "failed").astype(int)
    return meta, feats


def build_text(split: str) -> pd.DataFrame:
    frames = []
    doc_dir = DATASET_DIR / f"{split}_documents"
    for item in TEXT_ITEMS:
        t = pd.read_csv(doc_dir / f"{item}.csv")
        t = t.dropna(subset=["text"])
        t["text"] = t["text"].astype(str)
        t = t[t["text"].str.split().str.len() >= 20]  # drop degenerate chunks
        t["chunk_idx"] = t.groupby("cik").cumcount()
        t = t[t["chunk_idx"] < MAX_CHUNKS_PER_ITEM]
        t["item"] = item
        frames.append(t[["cik", "item", "chunk_idx", "text"]])
    return pd.concat(frames, ignore_index=True)


def main():
    stats = {}
    fin = {}
    for split in SPLITS:
        meta, feats = build_financial(split)
        fin[split] = (meta, feats)
        print(f"{split}: {len(meta)} companies, {meta['label'].sum()} bankrupt, "
              f"features {feats.shape}")

    # Fit winsorize bounds + standard scaler on train (pooled over years).
    train_flat = fin["train"][1].reshape(-1, len(RATIO_NAMES))
    bounds = winsorize_fit(train_flat)
    clipped = winsorize_apply(train_flat, bounds)
    mean, std = clipped.mean(axis=0), clipped.std(axis=0) + 1e-8

    np.savez(PROCESSED_DIR / "scaler.npz",
             lo=bounds[0], hi=bounds[1], mean=mean, std=std,
             feature_names=np.array(RATIO_NAMES))

    for split in SPLITS:
        meta, feats = fin[split]
        n, t, f = feats.shape
        scaled = (winsorize_apply(feats.reshape(-1, f), bounds) - mean) / std
        scaled = scaled.reshape(n, t, f).astype(np.float32)
        np.savez(PROCESSED_DIR / f"financial_{split}.npz",
                 cik=meta["cik"].values.astype(str), fyear=meta["fyear"].values,
                 label=meta["label"].values, features=scaled, raw_features=feats)
        text = build_text(split)
        text.to_parquet(PROCESSED_DIR / f"text_{split}.parquet", index=False)
        cov = text["cik"].nunique()
        stats[split] = {
            "companies": int(n), "bankrupt": int(meta["label"].sum()),
            "positive_rate": round(float(meta["label"].mean()), 4),
            "companies_with_text": int(cov), "text_chunks": int(len(text)),
            "fyear_min": int(meta["fyear"].min()), "fyear_max": int(meta["fyear"].max()),
        }
        print(f"{split}: text chunks={len(text)}, companies with text={cov}")

    (PROCESSED_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("Done. Stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
