# Architecture Notes

## Design principles

1. **Every deep-learning component must earn its place** — the ablation suite
   (A–E) isolates one design choice per comparison.
2. **FinBERT frozen, never fine-tuned** — with only 403 positive training
   companies, end-to-end fine-tuning of 110M parameters would overfit and
   requires GPU resources the project doesn't need. Frozen mean-pooled
   last-hidden-state embeddings are cached once; training then touches only
   40k–185k trainable parameters.
3. **Shared building blocks across ablation variants** — `YearEncoder`,
   `TextEncoder`, and `Head` are identical modules reused by all four neural
   models, so performance differences come from the fusion mechanism, not from
   incidental capacity differences.

## Modules (src/models/multimodal.py)

### YearEncoder
Per-year 20-d ratio vector → Linear(20→64) + LayerNorm + GELU + Dropout →
add learned positional embedding (3 positions) → 1 pre-norm Transformer
encoder layer (4 heads, FFN 128). Output: 3 year-tokens `[B, 3, 64]`.

Why self-attention over years instead of an LSTM: Riyanto et al. (2026) showed
recurrent models collapse on minority recall under this exact dataset family's
imbalance; a single attention block over 3 tokens is cheaper, order-aware via
positional embeddings, and easier to inspect.

### TextEncoder
Frozen FinBERT chunk embeddings `[B, K≤16, 768]` → Linear(768→64) + LayerNorm +
GELU + Dropout → prepend a **learned no-text token** (activated only for
companies with zero chunks, via the padding mask) → 1 pre-norm Transformer
encoder layer with `src_key_padding_mask`. This makes every company scoreable
without imputing fake text, and lets the model learn what "no disclosure
available" means.

### CrossModalModel (proposed, variant D)
`MultiheadAttention(query = financial year-tokens, key/value = text tokens)` —
the financial state of each year selects which disclosure chunks are relevant
to it. Residual + LayerNorm, then one self-attention fusion block over the
attended year-tokens. Final representation = mean-pooled fused year-tokens ⊕
masked-mean text tokens → 2-layer MLP head.

The attention map `[B, 3 years, K+1 chunks]` is stored per forward pass
(`model.last_attn`) and surfaced in `src/explain.py` and the demo — as a
diagnostic of what the model reads, explicitly **not** as a causal explanation.

### Ablation variants
- **A FinancialOnlyModel** = YearEncoder + head (no text path).
- **B TextOnlyModel** = TextEncoder + head (no financial path).
- **C ConcatModel** = both encoders, pooled representations concatenated —
  identical capacity to D minus the cross-attention, so C vs D measures the
  value of attention-based fusion specifically.
- **E** re-trains A and D with `n_years=1` (only the most recent fiscal year),
  measuring the value of temporal history.

## Class imbalance strategy

- `BCEWithLogitsLoss(pos_weight = N_neg/N_pos)` computed on the training split.
- Early stopping and model selection on **validation PR-AUC** (not accuracy,
  not ROC-AUC — PR-AUC is the metric that degrades when the minority class is
  ignored).
- Decision threshold tuned on validation F1, then frozen for the test split.
- No SMOTE (interpolating temporal financial trajectories creates unrealistic
  firms and can leak), no GAN oversampling (unstable, unnecessary at this
  scale — literature shows weighted loss + threshold tuning achieves comparable
  minority recall).

## Feature engineering (src/features.py)

Per fiscal year, from the 18 raw accounting items: 14 solvency/profitability/
efficiency ratios (Altman/Ohlson lineage: working capital/TA, retained
earnings/TA, EBIT/TA, market value/liabilities, ROA, leverage, margins,
debt/EBITDA, turnover ratios), 3 log-scaled size features, and 3 growth
features vs the previous year (zero for the oldest year in the window).
Winsorized at the train split's 1st/99th percentiles, then standardized with
train-split mean/std. The same code path serves training and the live demo.

## What we deliberately did NOT add

GANs/WGAN-GP, DQN/reinforcement learning, GNNs, multi-task learning, deep
LSTM stacks, end-to-end FinBERT fine-tuning. Justification for each exclusion
is in the literature review; the short version: none is required by the data
scale or the research question, and several (DQN, naive LSTM) are documented
failure modes under this imbalance regime.
