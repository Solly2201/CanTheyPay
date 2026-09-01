"""Single-company inference used by the demo: raw accounting items (3 years)
+ optional preprocessed text chunks -> distress probability + diagnostics."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (EMB_DIM, EXPERIMENTS_DIR, MAX_TEXT_TOKENS, MODELS_DIR,
                        PROCESSED_DIR)
from src.features import sequence_features
from src.models.multimodal import CrossModalModel

_CACHE = {}


def _scaler():
    if "scaler" not in _CACHE:
        _CACHE["scaler"] = np.load(PROCESSED_DIR / "scaler.npz")
    return _CACHE["scaler"]


def load_model():
    if "model" not in _CACHE:
        ckpt = torch.load(MODELS_DIR / "crossmodal_y3.pt", map_location="cpu",
                          weights_only=False)
        m = CrossModalModel(n_features=ckpt["n_features"], d_model=ckpt["d_model"])
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        _CACHE["model"] = m
    return _CACHE["model"]


def tuned_threshold(default=0.5):
    path = EXPERIMENTS_DIR / "results.json"
    if path.exists():
        res = json.loads(path.read_text())
        return res.get("crossmodal", {}).get("val", {}).get("threshold", default)
    return default


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """FinBERT-embed demo text chunks (loads FinBERT once, on demand)."""
    if not chunks:
        return np.zeros((0, EMB_DIM), dtype=np.float32)
    if "finbert" not in _CACHE:
        from src.data.embed_text import load_model as load_finbert
        _CACHE["finbert"] = load_finbert("cpu")
    tok, model = _CACHE["finbert"]
    from src.data.embed_text import embed_texts
    return embed_texts(chunks, tok, model, "cpu", batch_size=8)


def predict_company(years_items: list[dict], text_chunks: list[str] | None = None):
    """years_items: 3 dicts (oldest first) of the 18 raw items in $M."""
    sc = _scaler()
    raw_feats = sequence_features(years_items)                   # [3, F], un-scaled
    feats = np.clip(raw_feats, sc["lo"], sc["hi"])
    feats = ((feats - sc["mean"]) / sc["std"]).astype(np.float32)

    emb = embed_chunks(text_chunks or [])
    k = min(len(emb), MAX_TEXT_TOKENS)
    text = np.zeros((MAX_TEXT_TOKENS, EMB_DIM), dtype=np.float32)
    mask = np.zeros(MAX_TEXT_TOKENS, dtype=np.float32)
    text[:k], mask[:k] = emb[:k], 1.0

    model = load_model()
    with torch.no_grad():
        logit = model(torch.from_numpy(feats).unsqueeze(0),
                      torch.from_numpy(text).unsqueeze(0),
                      torch.from_numpy(mask).unsqueeze(0))
        prob = float(torch.sigmoid(logit).item())
    attn = model.last_attn[0].numpy() if model.last_attn is not None else None

    thr = tuned_threshold()
    if prob >= thr:
        category = "High risk"       # above the validation-tuned alarm threshold
    elif prob >= 0.5 * thr:
        category = "Elevated risk"
    else:
        category = "Low risk"
    return {"probability": prob, "threshold": thr, "category": category,
            "used_text_chunks": int(k),
            # engineered ratios exactly as fed to the scaler, for demo
            # diagnostics; not used by the model itself
            "ratios": raw_feats.tolist(),
            "chunk_attention": (attn.mean(axis=0)[1:1 + k].tolist()
                                if attn is not None and k else [])}
