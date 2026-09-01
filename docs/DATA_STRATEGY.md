# Dataset Strategy — Multi-Modal Transformer for Corporate Credit Risk & Bankruptcy Prediction

*Research completed 2026-09-01. All datasets below were verified hands-on or via primary sources, not assumed.*

---

## 1. The decision, up front

**We use Option A: the Lombardo/Pellegrino et al. multi-modal US bankruptcy dataset**
(`github.com/sowide/Multi-modal-bankrutpcy`, CC0-1.0) **as the training dataset**, and
**live SEC EDGAR (XBRL company facts + 10-K text) as the demo/inference source** for real,
named companies.

It is the only free dataset we found that already provides, per company:
**3 years of financial history + 10-K text (Items 1, 5, 7) + a temporally valid Chapter 7/11
label + a chronological train/val/test split** — i.e., exactly the structure this project needs.
We verified its contents locally (downloaded, parsed, cross-checked labels across modalities).

Everything else on the shortlist fails at least one hard requirement:

| Candidate | Why it was eliminated (verified, not assumed) |
|---|---|
| Kaggle *US Company Bankruptcy* (78,682 firm-years, same authors) | Anonymized (`C_1…`), **no CIK/ticker/name → cannot be joined to 10-K text**. Worse: `status_label` is company-level — every row of an eventually-failed company is marked `failed` even 15 years early → massive label leakage unless relabeled. Kept only as fallback (Option C). |
| Taiwan bankruptcy dataset (UCI/Kaggle, 6,819 firms, 95 ratios) | Fully anonymized, **no identifiers, no years** → no text alignment possible. |
| Polish bankruptcy dataset (UCI) | Same: anonymized, no identifiers → no text alignment. |
| CCRD (`Mengmeara/CCRD-Dataset`) | It's a **credit-rating** dataset (AAA…CCC labels), not bankruptcy; text ships only as pre-computed embeddings (can't run our own FinBERT pipeline); 2,307 samples, 2010–2016. |
| Mai et al. 2019 (EJOR) MD&A dataset | Never publicly released (built on Compustat/CRSP). |
| ECL (EDGAR–Compustat–LoPucki, `henriarnoUG/ECL`) | The best *large-scale* option (170,139 10-Ks, 1993–2023, CIK, LoPucki labels) — but the financial block requires **paid WRDS/Compustat**; the free substitute (SEC Financial Statement Data Sets) starts 2009 and needs heavy XBRL tag mapping; text corpus is ~40 GB. Too heavy for a college-semester CPU project. Documented below as **Option B (scale-up path)**. |

## 2. Sources and links

**Training data (Option A):**
- Multi-modal dataset: https://github.com/sowide/Multi-modal-bankrutpcy (`dataset_paper.zip`, 8.6 MB, CC0-1.0)
- Companion papers: Lombardo et al., *Future Internet* 14(8):244 (2022); Pellegrino et al., *Future Internet* 16(3):79 (2024)

**Demo / inference (real companies):**
- SEC XBRL company facts: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
- CIK↔ticker map: `https://www.sec.gov/files/company_tickers.json`
- 10-K documents: EDGAR archives (`sec.gov/Archives/edgar/data/{CIK}/…`), rate limit 10 req/s with a User-Agent header

**Option B (scale-up, documented for future work):**
- Labels: ECL (https://github.com/henriarnoUG/ECL, LoPucki BRD-derived, CIK-matched, 1993–2023) or LoPucki BRD cases table (free, frozen at Dec 2022, has CIK) ∪ 8-K filings whose `items` field contains `1.03` (Bankruptcy/receivership) from `data.sec.gov/submissions/` (2004+, exact event dates)
- Text: HuggingFace `eloukas/edgar-corpus` (10-Ks 1993–2020, pre-split into `section_1A`, `section_7`, with CIK; Apache-2.0; ~40 GB) + `lefterisloukas/edgar-crawler` for 2021+
- Financials: SEC Financial Statement Data Sets (quarterly ZIPs ~123 MB, 2009q1+, `sub.txt`/`num.txt`, join on `adsh`/CIK)
- Join key everywhere: **CIK + fiscal year**

## 3–5. Size, years, class balance (verified locally)

One observation per company (the last observed firm-year; for failed firms, the fiscal year
immediately before the Chapter 7/11 filing):

| Split | Period (fyear) | Healthy | Bankrupt | Positive rate |
|---|---|---|---|---|
| Train | 1999–2014 | 2,739 | 403 | 12.8% |
| Validation | 2015 | 206 | 27 | 11.6% |
| Test | 2016–2019 | 2,743 | 72 | **2.6%** |
| **Total** | 1999–2019 | 5,688 | 502 | 8.1% |

Each row carries **3 fiscal years of history** (`3_*` = most recent year t, `2_*` = t−1,
`1_*` = t−2; ordering verified empirically — failed firms' retained earnings/net income
deteriorate monotonically toward `3_`).

Note the **distribution shift**: the test period (2016–2019) has far fewer bankruptcies than
the training period (which includes the dot-com bust and 2008–09 crisis). This is realistic —
a deployed model faces exactly this — and we report it rather than hide it.

## 6. Financial features

18 raw accounting items per year (Compustat-style, $M): current assets, total assets, COGS,
total long-term debt, D&A, EBIT, EBITDA, gross profit, inventory, total current liabilities,
net income, retained earnings, total receivables, total revenue, market value, total
liabilities, net sales, total operating expenses.

From these we derive per-year **ratios** (Altman/Ohlson-style): current ratio, working
capital/TA, retained earnings/TA, EBIT/TA, market value/total liabilities, net income/TA
(ROA), total liabilities/TA (leverage), EBITDA margin, net margin, receivables turnover,
log(total assets) as size, plus year-over-year growth of revenue/assets/net income.
Raw items are kept too; scaling uses train-set statistics only.

## 7. Text availability (verified locally)

10-K **Item 1 (Business)** and **Item 7 (MD&A)** text, already pre-chunked into ≤256-word
segments, lowercased, stop-words/numbers stripped. Item 5 exists but is nearly empty
(88 train rows) → excluded.

| Split | Companies with any text | Chunks (Item 1 + Item 7) |
|---|---|---|
| Train | 2,145 / 3,142 (68%) | 10,269 |
| Val | 149 / 233 (64%) | 745 |
| Test | 1,929 / 2,815 (69%) | 8,048 |

~3 chunks per item per company (max 50). Missing text is handled with an explicit
learned "no-text" embedding + attention masking — never imputed with fake text.
Note the limitation honestly: the dataset provides Items 1/5/7, **not Item 1A Risk
Factors** (Item 1A only became mandatory in 2005, mid-way through this dataset's period).
MD&A (Item 7) is the section most used in the bankruptcy-text literature (Mai et al. 2019).

## 8. Alignment

Already aligned by the dataset authors: every text chunk and financial row carries the same
anonymized `cik` key; we verified label agreement between modalities is 100% (crosstab:
zero mismatches in all three splits). No fuzzy joining needed — this is precisely why this
dataset wins over hand-assembled alternatives.

## 9–10. Prediction target and horizon

**Target:** binary — does the company file Chapter 7/11 in the year following the observed
fiscal year? Label attached to fiscal year t = filing occurs in t+1 (dataset authors' rule:
"the fiscal year before the chapter filing" is labeled 1). **Horizon: 1 year ahead**, using
up to 3 years of history (t−2, t−1, t) — matching the project spec
("2019/2020/2021 info → predict 2022 distress").

## 11. Train/validation/test strategy

The dataset ships a **chronological, company-disjoint split** (train ≤2014, val 2015,
test 2016–2019). Each company appears in exactly one split and exactly once (verified:
`cik` is unique per split, splits are disjoint). We keep this split — it prevents both
temporal leakage and same-company-across-splits leakage, and it makes our numbers directly
comparable to the published benchmarks on this dataset. Threshold τ is tuned on validation
only; test is touched once per final model.

## 12. Expected class imbalance

Train 12.8% positive, test 2.6% positive. Handled via weighted BCE
(`pos_weight = N_neg/N_pos` on train), stratified batch sampling where needed, and
validation-tuned decision threshold. We do **not** use SMOTE (leak-prone with temporal
data) and do not need GAN oversampling — weighted loss + threshold tuning is the stable,
literature-backed choice (Mundkar & Khadse 2026; Xiao & Liu 2026).

## 13. Data leakage risks and mitigations

| Risk | Status / mitigation |
|---|---|
| Company-level `status_label` marking all years of a failed firm (the Kaggle 78k CSV bug we verified) | **Avoided entirely** — the multimodal set has one event-labeled row per company. |
| Same company in train and test | Verified impossible: splits are company-disjoint. |
| Random row splits mixing time | Not used; chronological split shipped and kept. |
| Future financial info in features | Features are years t−2…t only; label is t+1 event. |
| Post-bankruptcy filings as text | Text comes from the 10-K of fiscal year t, filed ~2–3 months after year-end, before the t+1 filing event in the normal case. Residual risk: a 10-K filed *after* a very early-in-t+1 bankruptcy petition could contain post-petition language ("going concern"/Chapter 11 mentions). We treat going-concern language as legitimate signal (it precedes the event) but document this as a residual risk we cannot fully audit because the dataset is anonymized. |
| Duplicate company-year records | Verified none (unique `cik` per split). |
| Restated financials | Unknown/unauditable in an anonymized dataset — documented limitation; as-first-reported vs restated cannot be distinguished. |
| Scaler/threshold fitted on test | All preprocessing statistics fit on train only; τ tuned on val only. |
| Text of target period (t+1) | Not present — text is from year t's 10-K only. |

## 14. Storage / compute requirements

- Dataset: 8.6 MB zip → ~28 MB extracted CSVs.
- FinBERT embeddings: ~19k chunks × 768 floats ≈ **56 MB fp32** (cached to disk once).
- Embedding extraction: chunks are ≤256 words (≈350–450 WordPiece tokens); on this
  machine's CPU ≈ 1.5–3 h one-time (batched); on a free Colab T4 ≈ minutes. Cached, so
  training never re-runs BERT.
- Model training: all models are small (≤ ~1M params); minutes per run on CPU.
- Total disk < 500 MB including the FinBERT checkpoint (~440 MB).

## 15. Recommended core architecture

Frozen FinBERT (`yiyanghkust/finbert-pretrain` — BERT-base pretrained on 10-Ks/earnings
calls, the best domain match for filings; ProsusAI/finbert is news-sentiment-oriented)
as chunk embedding extractor → cached. Then:

- Financial encoder: per-year ratio vector → linear proj to d_model, 3 year-tokens + learned
  positional embedding.
- Text: chunk embeddings 768 → linear proj to d_model, chunk tokens + padding mask.
- Fusion: 1–2 blocks of cross-modal attention (financial tokens as queries, text chunks as
  keys/values) + shallow self-attention; mean-pool → 2-layer MLP head → sigmoid.
- Weighted BCE, dropout 0.3–0.4, weight decay, early stopping on val PR-AUC, τ tuned on val.

## 16. Optional extensions (only if core results warrant)

Attention-directionality ablation (text-queries-financial), LoRA fine-tuning of FinBERT's
top layers on a GPU, Option B scale-up to real-CIK EDGAR data, per-sector analysis.

## 17. Estimated implementation difficulty

**Moderate and comfortably within a college semester.** The dataset is turnkey; the one-time
FinBERT embedding pass is the only slow step; every model trains in minutes on CPU. The
riskiest part (data acquisition/alignment) is eliminated by Option A.

---

## The demo-company question (evaluated separately, as required)

The requested showcase companies are **not** in the training data (it is anonymized), and
must not be — they are a demo/inference feature backed by live EDGAR data:

| Company | Feasible? | Why |
|---|---|---|
| Amazon (CIK 1018724) | ✅ | US filer, full 10-K + XBRL history |
| Microsoft (789019) | ✅ | Same |
| NVIDIA (1045810) | ✅ | Same |
| Alphabet/Google (1652044) | ✅ | Same (Google pre-2015 under CIK 1288776) |
| Meta/Facebook (1326801) | ✅ | Same |
| X / Twitter (1418091) | ⚠️ Historical only | Public 2013–Oct 2022; last 10-K is FY2021. Prediction valid only for pre-2022 years. |
| LinkedIn (1271024) | ⚠️ Historical only | 10-Ks FY2011–FY2015; acquired by Microsoft Dec 2016. |
| YouTube | ❌ | Alphabet subsidiary — no separate SEC filings ever. |
| Zomato (now Eternal Ltd) | ❌ (manual input only) | Listed on NSE/BSE India; Ind-AS accounts in ₹, no SEC filings. Model is trained on US GAAP $M — a prediction would not be valid. Manual-input mode allowed with an explicit validity warning. |
| Swiggy | ❌ (manual input only) | Same — Indian listing (Nov 2024 IPO), no SEC filings. |
| Flipkart | ❌ | Private (Walmart subsidiary), no public filings. |
| OpenAI | ❌ | Private, no public filings. |

The demo interface therefore has three modes: (1) pick a US public filer → live EDGAR fetch
of financials + latest 10-K text; (2) manual entry of the 18 financial items (+ optional
pasted annual-report text) with an out-of-distribution warning for non-US-GAAP inputs;
(3) browse anonymized dataset companies. It always displays "insufficient public data"
rather than fabricating a prediction.
