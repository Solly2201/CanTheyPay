# Pre-Release Results Audit

Audit performed 2026-09-01 before the final release. Scripts live in
`experiments/audit_*.py` and are re-runnable. Nothing in `results.md` was
changed by this audit except one incorrect model *label* (see finding 5);
all metric values were verified, not modified.

## A. Split integrity — PASS
- `cik` unique within every split (no duplicate company-year rows).
- Zero company overlap between train/validation/test.
- Chronological: max train fyear 2014 < val 2015 < min test 2016.
- Every text-table cik belongs to the same split's financial table.

## B. Preprocessing leakage — PASS
- Winsorization bounds and standard-scaler mean/std recomputed from the raw
  train features exactly reproduce `scaler.npz` (fit on train only).
- Stored scaled features for all three splits reproduce bit-exactly from
  raw features + train-fit statistics.
- FinBERT embedding extraction (`src/data/embed_text.py`) is pure per-chunk
  inference — no statistics fitted, no cross-sample or cross-split information.

## C. Test-set consistency — PASS
- Every model in `results.json` reports test n=2815, positives=72; every
  confusion matrix sums to 2815 with 33–50 FN + TP = 72.
- All models are scored on the identical 2,815 test companies; text-less
  companies flow through the no-text pathway (learned token for neural models,
  zero-vector mean embedding for classical models) rather than being dropped.
- Decision thresholds: verified `test.threshold == val.threshold` for every
  model (tuned on validation only).

## D. Reported numbers vs. actual runs — PASS
- All deterministic models (LR ×2, XGBoost ×2, MLP) retrained from scratch:
  every metric matches `results.json` exactly.
- Neural ensemble and per-seed PR-AUCs in `results.json` match the raw
  training log (`run_log.txt`) line-for-line, all 30 seed values included.
- Every number in `results.md` matches `results.json`.

## E. FinBERT future information — DOCUMENTED CAVEAT
`yiyanghkust/finbert-pretrain` was pretrained (unsupervised) on SEC filings
and analyst text through ~2019, which overlaps the 2016–2018 test window.
This is standard practice in the literature and involves no label access,
but it is temporal contamination in the strict sense: the embedding space has
"seen" test-era language. Eliminating it would require a pre-2015 language
model checkpoint, which does not exist for FinBERT.

## F. Post-petition text leakage — GENUINE ISSUE FOUND, QUANTIFIED
`audit_text_leakage.py` scans for bankruptcy-as-accomplished-fact phrasing
("chapter cases", "debtor possession", "emerged bankruptcy", ...) that could
only appear if the year-t 10-K was filed *after* the Chapter petition:

| Split | label=1 with post-petition phrasing | label=0 |
|---|---|---|
| train | 18/287 (6.3%) | 24/1816 (1.3%) |
| validation | 5/21 (23.8%) | 2/126 (1.6%) |
| test | 6/48 (12.5%) | 9/1847 (0.5%) |

Manual review of the 6 test cases: 4 clearly post-petition, 1 about trade
petitions (false positive), 1 about competitors' bankruptcies. This leakage is
inherent to the source dataset (anonymized, so filing dates cannot be checked)
and cannot be repaired by us; it inflates text-model scores.

**Sensitivity analysis** (`audit_sensitivity.py`) — re-scoring with all 6
companies' text masked (conservative upper bound; training untouched):

| Model | test PR-AUC original | text-masked |
|---|---|---|
| XGBoost financial + FinBERT | 0.3313 | 0.3082 |
| Cross-modal 1-yr (best-val seed) | 0.3606 | 0.3265 |
| Cross-modal 3-yr (best-val seed) | 0.2068 | 0.1849 |

Conclusion: headline PR-AUCs of text-using models are inflated by roughly
0.02–0.03 by this leakage. **The qualitative conclusions survive masking**:
text still adds signal over financial-only models (masked XGB+text 0.308 vs
XGB-financial 0.258), and ROC-AUC is essentially unchanged. The elevated
validation rate (23.8%) also means threshold tuning and early stopping saw
slightly optimistic text signal. All reported results keep the original
(unmasked) evaluation for comparability with the dataset's published
benchmarks, with this caveat attached.

## G. Baseline comparison fairness — PASS with one disclosure
- All classical baselines receive the same standardized features
  (flattened 3-year, 60 dims) and, for the multimodal XGBoost, the same
  FinBERT embeddings (mean-pooled) the neural models consume.
- Disclosure: the "single-year" ablation models use one year-*token*, but that
  token's feature vector includes year-over-year growth ratios computed from
  t-1 values. "Single-year" therefore means "no explicit temporal sequence",
  not "no information from earlier years".

## H. Label construction — PASS (inherited from dataset, verified internally)
- One event-labeled row per company (the fiscal year before the Chapter 7/11
  filing), avoiding the company-level label leak present in the related
  78k-row Kaggle CSV.
- Financial-table labels and text-table labels agree 100% in all splits.
- Restated-financials risk remains unauditable (anonymized source) — see
  `DATA_STRATEGY.md` §13.

## Findings fixed as a result of this audit
1. `results.md` mislabeled the sklearn MLP baseline as "(financial, 1 yr)";
   it uses flattened 3-year features. Label corrected; numbers unchanged.
2. README results discussion updated to carry the leakage sensitivity numbers
   and the single-year-ablation disclosure.
