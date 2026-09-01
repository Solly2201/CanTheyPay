"""Neural models: financial encoder, text encoder, fusion variants.

All variants share the same building blocks so ablations isolate exactly one
design choice at a time:

  FinancialOnlyModel      A: 3 year-tokens -> self-attention -> pool -> head
  TextOnlyModel           B: K chunk-tokens -> self-attention -> pool -> head
  ConcatModel             C: mean-pooled fin + mean-pooled text -> MLP head
  CrossModalModel         D: financial year-tokens attend to text chunks
                             (Q=finance, K/V=text) + self-attention -> head
"""
import torch
import torch.nn as nn


class YearEncoder(nn.Module):
    """Projects per-year financial vectors to d_model tokens with positional
    embeddings, followed by one Transformer self-attention block over years."""

    def __init__(self, n_features, d_model, n_heads, dropout, n_years=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.randn(1, n_years, d_model) * 0.02)
        self.block = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True)

    def forward(self, fin):                     # [B, T, F]
        h = self.proj(fin) + self.pos[:, :fin.shape[1]]
        return self.block(h)                    # [B, T, d]


class TextEncoder(nn.Module):
    """Projects frozen FinBERT chunk embeddings to d_model tokens. Companies
    without text get a learned no-text token so every example is scoreable."""

    def __init__(self, d_model, n_heads, dropout, emb_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.no_text = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.block = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True)

    def forward(self, text, mask):              # [B, K, 768], [B, K]
        h = self.proj(text)                     # [B, K, d]
        empty = mask.sum(dim=1) == 0            # [B]
        # give empty companies one usable no-text token
        h = torch.cat([self.no_text.expand(h.shape[0], 1, -1), h], dim=1)
        first = empty.unsqueeze(1).to(mask.dtype)  # [B, 1]
        mask = torch.cat([first, mask], dim=1)  # [B, K+1]
        h = self.block(h, src_key_padding_mask=(mask == 0))
        return h, mask


def masked_mean(h, mask):
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)


class Head(nn.Module):
    def __init__(self, d_in, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Dropout(dropout),
            nn.Linear(d_in, d_in), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_in, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FinancialOnlyModel(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, dropout=0.3, n_years=3):
        super().__init__()
        self.enc = YearEncoder(n_features, d_model, n_heads, dropout, n_years)
        self.head = Head(d_model, dropout)

    def forward(self, fin, text=None, mask=None):
        return self.head(self.enc(fin).mean(dim=1))


class TextOnlyModel(nn.Module):
    def __init__(self, n_features=None, d_model=64, n_heads=4, dropout=0.3):
        super().__init__()
        self.enc = TextEncoder(d_model, n_heads, dropout)
        self.head = Head(d_model, dropout)

    def forward(self, fin, text, mask):
        h, m = self.enc(text, mask)
        return self.head(masked_mean(h, m))


class ConcatModel(nn.Module):
    """Simple fusion baseline: no attention between modalities."""

    def __init__(self, n_features, d_model=64, n_heads=4, dropout=0.3, n_years=3):
        super().__init__()
        self.fin_enc = YearEncoder(n_features, d_model, n_heads, dropout, n_years)
        self.text_enc = TextEncoder(d_model, n_heads, dropout)
        self.head = Head(d_model * 2, dropout)

    def forward(self, fin, text, mask):
        f = self.fin_enc(fin).mean(dim=1)
        h, m = self.text_enc(text, mask)
        return self.head(torch.cat([f, masked_mean(h, m)], dim=-1))


class CrossModalModel(nn.Module):
    """Proposed model: financial year-tokens query the text chunks
    (Q = finance, K/V = text), residual + self-attention, pooled head."""

    def __init__(self, n_features, d_model=64, n_heads=4, dropout=0.3, n_years=3):
        super().__init__()
        self.fin_enc = YearEncoder(n_features, d_model, n_heads, dropout, n_years)
        self.text_enc = TextEncoder(d_model, n_heads, dropout)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.fuse = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.head = Head(d_model * 2, dropout)
        self.last_attn = None  # [B, T, K+1] attention map for diagnostics

    def forward(self, fin, text, mask):
        f = self.fin_enc(fin)                       # [B, T, d]
        h, m = self.text_enc(text, mask)            # [B, K+1, d]
        attended, attn = self.cross(
            query=f, key=h, value=h,
            key_padding_mask=(m == 0), need_weights=True,
            average_attn_weights=True)
        self.last_attn = attn.detach()
        f = self.norm1(f + attended)                # residual fusion
        f = self.fuse(f)
        fin_repr = f.mean(dim=1)
        text_repr = masked_mean(h, m)
        return self.head(torch.cat([fin_repr, text_repr], dim=-1))


MODEL_REGISTRY = {
    "financial_only": FinancialOnlyModel,
    "text_only": TextOnlyModel,
    "concat": ConcatModel,
    "crossmodal": CrossModalModel,
}
