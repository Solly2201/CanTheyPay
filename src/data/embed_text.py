"""Extract frozen FinBERT chunk embeddings and cache them to disk.

FinBERT (yiyanghkust/finbert-pretrain) is used as a FROZEN feature extractor:
mean-pooled last-hidden-state over non-padding tokens, one 768-d vector per
256-word chunk. Embeddings are cached, so BERT never runs during model training.

Usage: python -m src.data.embed_text [--split train] [--batch-size 16]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import EMB_DIM, FINBERT_MODEL, PROCESSED_DIR, SPLITS


def load_model(device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModel.from_pretrained(FINBERT_MODEL).to(device).eval()
    return tok, model


@torch.no_grad()
def embed_texts(texts, tok, model, device, batch_size=16, max_length=512):
    out = np.zeros((len(texts), EMB_DIM), dtype=np.float32)
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state          # [B, L, 768]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out[i:i + len(batch)] = pooled.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS + ["all"], default="all")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tok, model = load_model(device)

    splits = SPLITS if args.split == "all" else [args.split]
    for split in splits:
        out_path = PROCESSED_DIR / f"embeddings_{split}.npz"
        if out_path.exists():
            print(f"{out_path} exists, skipping")
            continue
        text = pd.read_parquet(PROCESSED_DIR / f"text_{split}.parquet")
        emb = embed_texts(text["text"].tolist(), tok, model, device,
                          batch_size=args.batch_size)
        np.savez_compressed(out_path,
                            cik=text["cik"].values.astype(str),
                            item=text["item"].values.astype(str),
                            chunk_idx=text["chunk_idx"].values,
                            embeddings=emb)
        print(f"{split}: saved {emb.shape} -> {out_path}")


if __name__ == "__main__":
    main()
