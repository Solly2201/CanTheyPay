"""Loads processed arrays and assembles per-company multimodal examples.

Each example:
  fin:   [T=3, F]   standardized per-year financial feature vectors (t-2, t-1, t)
  text:  [K, 768]   FinBERT chunk embeddings (zero-padded)
  mask:  [K]        1 for real chunks, 0 for padding (all-zero if company has no text)
  label: {0, 1}     bankruptcy filed in year t+1
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EMB_DIM, MAX_TEXT_TOKENS, PROCESSED_DIR


def load_split(split: str):
    fin = np.load(PROCESSED_DIR / f"financial_{split}.npz", allow_pickle=False)
    emb_path = PROCESSED_DIR / f"embeddings_{split}.npz"
    emb_by_cik = {}
    if emb_path.exists():
        e = np.load(emb_path, allow_pickle=False)
        ciks, items, idxs, embs = e["cik"], e["item"], e["chunk_idx"], e["embeddings"]
        # order chunks: item_1 first then item_7, each in document order
        order = np.lexsort((idxs, items, ciks))
        for i in order:
            emb_by_cik.setdefault(str(ciks[i]), []).append(embs[i])
    return fin, emb_by_cik


class BankruptcyDataset(Dataset):
    def __init__(self, split: str, max_chunks: int = MAX_TEXT_TOKENS,
                 n_years: int = 3):
        fin, emb_by_cik = load_split(split)
        self.cik = fin["cik"]
        self.features = fin["features"][:, -n_years:, :]  # [N, n_years, F]
        self.labels = fin["label"].astype(np.float32)
        self.fyear = fin["fyear"]
        self.max_chunks = max_chunks
        self.text = []
        self.mask = []
        for c in self.cik:
            chunks = emb_by_cik.get(str(c), [])[:max_chunks]
            t = np.zeros((max_chunks, EMB_DIM), dtype=np.float32)
            m = np.zeros(max_chunks, dtype=np.float32)
            if chunks:
                arr = np.stack(chunks)
                t[:len(arr)] = arr
                m[:len(arr)] = 1.0
            self.text.append(t)
            self.mask.append(m)
        self.text = np.stack(self.text)
        self.mask = np.stack(self.mask)

    @property
    def n_features(self):
        return self.features.shape[2]

    @property
    def has_text(self):
        return self.mask.sum(axis=1) > 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.from_numpy(self.features[i]),
                torch.from_numpy(self.text[i]),
                torch.from_numpy(self.mask[i]),
                torch.tensor(self.labels[i]))


def numpy_views(ds: BankruptcyDataset):
    """Flat numpy views for the sklearn/XGBoost baselines."""
    fin_flat = ds.features.reshape(len(ds), -1)                    # [N, 3*F]
    msum = ds.mask.sum(axis=1, keepdims=True)
    text_mean = ds.text.sum(axis=1) / np.maximum(msum, 1.0)        # [N, 768]
    return fin_flat, text_mean, ds.labels, ds.has_text
