# CanTheyPay — Multi-Modal Transformer for Corporate Credit Risk & Bankruptcy Prediction

A college Deep Learning project investigating whether **learned multimodal
representations and attention-based fusion** can extract additional predictive
information from corporate financial history and 10-K textual disclosures,
beyond what conventional ML (Logistic Regression, XGBoost) and simple fusion
(concatenation) capture.

**Research question:** *Can a cross-modal Transformer that fuses a company's
multi-year financial trajectory with FinBERT representations of its annual-report
text improve 1-year-ahead bankruptcy prediction over strong unimodal and
simple-fusion baselines, under realistic class imbalance?*

## Architecture

```
Financial history (3 fiscal years, 20 engineered ratios/year)
        │
   [Year encoder: linear proj + positional emb + self-attention]  → 3 year-tokens
        │
        │      10-K Item 1 / Item 7 text (256-word chunks)
        │             │
        │      [Frozen FinBERT (yiyanghkust/finbert-pretrain) → 768-d chunk embeddings, cached]
        │             │
        │      [Text encoder: linear proj + self-attention; learned "no-text" token]
        │             │
   [Cross-modal attention: Q = financial year-tokens, K/V = text chunks]
        │
   [Residual + self-attention fusion block]
        │
   [Pooled financial ⊕ pooled text → MLP head → sigmoid]
        │
   Weighted BCE (pos_weight = N_neg/N_pos), early stopping on val PR-AUC,
   decision threshold tuned on validation
```

All neural models are small (40k–185k parameters) and train in minutes on CPU;
FinBERT is used **frozen** as an embedding extractor (one-time cached pass).

## Dataset

The multi-modal US bankruptcy dataset of Lombardo et al. / Pellegrino et al.
(*Future Internet* 2022/2024), CC0-1.0:
**6,190 US public companies (NYSE/NASDAQ, 1999–2019)**, each with 3 years of
18 accounting items, pre-chunked 10-K Item 1 + Item 7 text (~68% coverage), and
a temporally valid Chapter 7/11 label (label year t = filing in t+1).
Ships with a **chronological, company-disjoint split**: train ≤2014 (12.8%
positive), val 2015, test 2016–2019 (2.6% positive).

Full source evaluation, rejected alternatives, leakage audit, and the demo-company
feasibility table: [`docs/DATA_STRATEGY.md`](docs/DATA_STRATEGY.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥3.10. CPU is sufficient; a GPU only speeds up the one-time embedding step.

## Reproduce everything

```bash
python -m src.data.download      # 8.6 MB dataset from GitHub (CC0)
python -m src.data.prepare       # features, scaling (train-fit only), text table
python -m src.data.embed_text    # one-time frozen-FinBERT pass (~1-2 h CPU, minutes on GPU)
python -m src.run_experiments    # all baselines + proposed model + ablations
python -m src.explain            # feature importance + attention diagnostics
streamlit run demo/app.py        # interactive demo
```

Outputs land in `experiments/` (`results.json`, `results.md`, importance CSVs,
attention examples). Seeds are fixed (`src/config.py`); preprocessing statistics
are fit on the training split only.

## Models compared

| # | Model | Modality | Fusion |
|---|---|---|---|
| 1 | Logistic Regression | financial | — |
| 2 | XGBoost | financial | — |
| 3 | MLP / FNN | financial | — |
| 4 | Logistic Regression on FinBERT embeddings | text | — |
| 5 | XGBoost on financial + mean FinBERT embedding | both | concatenation |
| 6 | Neural financial encoder (A) | financial | — |
| 7 | Neural text encoder (B) | text | — |
| 8 | Concat fusion (C) | both | concatenation |
| 9 | **Cross-modal Transformer (D, proposed)** | both | **cross-attention** |
| 10 | Single-year variants (E) | ablation of temporal history | |

Metrics: ROC-AUC, **PR-AUC (primary)**, precision, recall, F1, confusion matrix.
Accuracy is reported but never used for selection (2.6% positive test rate makes
it meaningless). Results: [`experiments/results.md`](experiments/results.md).

## Demo

`streamlit run demo/app.py` supports:
- **Real company by ticker** — live SEC EDGAR fetch (XBRL company facts + latest
  10-K Item 1/7). Works for US filers (AMZN, MSFT, NVDA, META, GOOGL, …).
  Companies without SEC filings (OpenAI, Flipkart, YouTube, Zomato/Eternal,
  Swiggy) are refused with an explanation instead of a fabricated prediction.
- **Dataset company** — browse the anonymized held-out test split.
- **Manual input** — enter the 18 accounting items and paste report text.

## Repository layout

```
src/config.py            paths, constants, seeds
src/data/download.py     dataset download
src/data/prepare.py      cleaning, features, scaling, text table
src/data/embed_text.py   frozen-FinBERT chunk embeddings (cached)
src/features.py          ratio engineering (shared with demo)
src/dataset.py           torch Dataset (financial seq + text chunks + mask)
src/models/baselines.py  LR / XGBoost baselines
src/models/multimodal.py neural models (A/B/C/D variants)
src/train.py             weighted-BCE training loop, early stopping
src/evaluate.py          metrics + threshold tuning
src/run_experiments.py   full experiment suite + ablations
src/explain.py           permutation importance, XGBoost gain, attention examples
src/edgar_live.py        demo-time live SEC EDGAR fetch
src/inference.py         single-company inference
demo/app.py              Streamlit demo
docs/                    data strategy, architecture notes
experiments/             results + diagnostics (generated)
```

## Limitations (read before citing numbers)

- **Anonymized training data** — real CIKs are withheld by the dataset authors
  (licensing), so training rows cannot be audited against EDGAR, and restated
  financials cannot be detected.
- **Labels** — Chapter 7/11 filings only; other distress (delisting, distressed
  exchanges, acquisitions of failing firms) counts as "healthy".
- **Text** — Items 1/5/7 (not Item 1A Risk Factors); pre-stripped of stopwords
  and numbers, which removes some signal FinBERT could have used; only ~68% of
  companies have text (handled by masking, not imputation).
- **Test-period shift** — 2016–2018 has 2.6% positives vs 12.8% in training;
  metrics reflect that harder, realistic regime. Positive test count is small
  (72), so PR-AUC has non-trivial variance.
- **Attention ≠ explanation** — attention maps and permutation importances are
  model diagnostics, not causal explanations.
- Not financial advice.
