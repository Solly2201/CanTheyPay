"""Architecture-refinement study: configurable variants of the cross-modal
model. Isolated from the frozen pipeline (models/multimodal.py is untouched).

Pre-registered variant axes (see src/tune_architecture.py for the protocol):
  text side:     section selection (item_1/item_7/both), item-type embeddings,
                 L2-normalized chunk embeddings
  attention side: direction (fin->text / text->fin / bidirectional), gating,
                 head count, stacked cross-attention layers
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import EMB_DIM, MAX_TEXT_TOKENS, PROCESSED_DIR
from src.models.multimodal import Head, YearEncoder, masked_mean


@dataclass
class VariantConfig:
    name: str
    items: tuple = ("item_1", "item_7")
    l2_normalize: bool = False
    item_embedding: bool = False
    direction: str = "f2t"      # 'f2t' | 't2f' | 'bi'
    gate: bool = False
    n_heads: int = 4
    n_cross_layers: int = 1
    d_model: int = 64
    dropout: float = 0.3
    notes: str = ""


class VariantDataset(torch.utils.data.Dataset):
    """Single-year financial features + filtered/normalized text chunks with
    item ids (0=pad, 1=item_1, 2=item_7)."""

    ITEM_ID = {"item_1": 1, "item_7": 2}

    def __init__(self, split, cfg: VariantConfig, max_chunks=MAX_TEXT_TOKENS):
        fin = np.load(PROCESSED_DIR / f"financial_{split}.npz")
        self.cik = fin["cik"]
        self.features = fin["features"][:, -1:, :]   # single-year token
        self.labels = fin["label"].astype(np.float32)

        e = np.load(PROCESSED_DIR / f"embeddings_{split}.npz")
        ciks, items, idxs, embs = e["cik"], e["item"], e["chunk_idx"], e["embeddings"]
        if cfg.l2_normalize:
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        keep = np.isin(items, list(cfg.items))
        order = np.lexsort((idxs[keep], items[keep], ciks[keep]))
        by_cik = {}
        kc, ki, ke = ciks[keep], items[keep], embs[keep]
        for i in order:
            by_cik.setdefault(str(kc[i]), []).append(
                (self.ITEM_ID[str(ki[i])], ke[i]))

        n = len(self.cik)
        self.text = np.zeros((n, max_chunks, EMB_DIM), dtype=np.float32)
        self.mask = np.zeros((n, max_chunks), dtype=np.float32)
        self.item_ids = np.zeros((n, max_chunks), dtype=np.int64)
        for r, c in enumerate(self.cik):
            chunks = by_cik.get(str(c), [])[:max_chunks]
            for j, (iid, v) in enumerate(chunks):
                self.text[r, j] = v
                self.mask[r, j] = 1.0
                self.item_ids[r, j] = iid

    @property
    def n_features(self):
        return self.features.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.from_numpy(self.features[i]),
                torch.from_numpy(self.text[i]),
                torch.from_numpy(self.mask[i]),
                torch.from_numpy(self.item_ids[i]),
                torch.tensor(self.labels[i]))


class VariantTextEncoder(nn.Module):
    def __init__(self, cfg: VariantConfig):
        super().__init__()
        d = cfg.d_model
        self.proj = nn.Sequential(nn.Linear(EMB_DIM, d), nn.LayerNorm(d),
                                  nn.GELU(), nn.Dropout(cfg.dropout))
        self.item_emb = (nn.Embedding(3, d, padding_idx=0)
                         if cfg.item_embedding else None)
        self.no_text = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.block = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=d * 2, dropout=cfg.dropout,
            batch_first=True, norm_first=True)

    def forward(self, text, mask, item_ids):
        h = self.proj(text)
        if self.item_emb is not None:
            h = h + self.item_emb(item_ids)
        empty = (mask.sum(dim=1) == 0)
        h = torch.cat([self.no_text.expand(h.shape[0], 1, -1), h], dim=1)
        mask = torch.cat([empty.unsqueeze(1).to(mask.dtype), mask], dim=1)
        h = self.block(h, src_key_padding_mask=(mask == 0))
        return h, mask


class CrossBlock(nn.Module):
    def __init__(self, cfg: VariantConfig):
        super().__init__()
        d = cfg.d_model
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.gate = (nn.Linear(2 * d, d) if cfg.gate else None)

    def forward(self, q, kv, kv_mask):
        att, w = self.attn(q, kv, kv, key_padding_mask=(kv_mask == 0),
                           need_weights=True, average_attn_weights=True)
        if self.gate is not None:
            g = torch.sigmoid(self.gate(torch.cat([q, att], dim=-1)))
            att = g * att
        return self.norm(q + att), w


class VariantModel(nn.Module):
    def __init__(self, n_features, cfg: VariantConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.fin_enc = YearEncoder(n_features, d, cfg.n_heads, cfg.dropout,
                                   n_years=1)
        self.text_enc = VariantTextEncoder(cfg)
        n = cfg.n_cross_layers
        if cfg.direction in ("f2t", "bi"):
            self.f2t = nn.ModuleList(CrossBlock(cfg) for _ in range(n))
        if cfg.direction in ("t2f", "bi"):
            self.t2f = nn.ModuleList(CrossBlock(cfg) for _ in range(n))
        self.fuse = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=d * 2, dropout=cfg.dropout,
            batch_first=True, norm_first=True)
        self.head = Head(d * 2, cfg.dropout)
        self.last_attn = None

    def forward(self, fin, text, mask, item_ids):
        f = self.fin_enc(fin)                          # [B, 1, d]
        h, m = self.text_enc(text, mask, item_ids)     # [B, K+1, d]
        fin_mask = torch.ones(f.shape[:2], device=f.device, dtype=m.dtype)
        if self.cfg.direction in ("f2t", "bi"):
            for blk in self.f2t:
                f, w = blk(f, h, m)
                self.last_attn = w.detach()
        if self.cfg.direction in ("t2f", "bi"):
            for blk in self.t2f:
                h, _ = blk(h, f, fin_mask)
        f = self.fuse(f)
        return self.head(torch.cat([f.mean(1), masked_mean(h, m)], dim=-1))


# ---- pre-registered candidate set --------------------------------------
CANDIDATES = [
    VariantConfig("V0_base", notes="in-harness control == crossmodal_y1 design"),
    VariantConfig("V1_item7_only", items=("item_7",),
                  notes="MD&A only; literature-favored section"),
    VariantConfig("V2_item1_only", items=("item_1",),
                  notes="Business section only"),
    VariantConfig("V3_item_embedding", item_embedding=True,
                  notes="chunks carry a learned section-type embedding"),
    VariantConfig("V4_l2_norm", l2_normalize=True,
                  notes="unit-norm chunk embeddings before projection"),
    VariantConfig("V5_text_queries_fin", direction="t2f",
                  notes="reversed attention direction"),
    VariantConfig("V6_bidirectional", direction="bi",
                  notes="both attention directions"),
    VariantConfig("V7_gated", gate=True,
                  notes="sigmoid gate on attended output"),
    VariantConfig("V8_heads8", n_heads=8, notes="more attention heads"),
    VariantConfig("V9_2layers", n_cross_layers=2,
                  notes="stacked cross-attention"),
]
