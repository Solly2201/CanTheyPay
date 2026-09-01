# Experiment Results

Chronological company-disjoint splits: train 2001-2014 (12.8% positive), val 2015, test 2016-2018 (**2.6% positive**). Decision threshold tuned on validation F1; test touched once per model. Neural models are 5-seed probability ensembles; the per-seed column shows single-run PR-AUC mean±std. Sorted by test PR-AUC (the primary metric under this imbalance).

| Model | ROC-AUC | PR-AUC | per-seed PR-AUC | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|
| E: Cross-modal, single year | 0.9145 | 0.3395 | 0.272±0.055 | 0.2378 | 0.5417 | 0.3305 | 2618/125/33/39 |
| XGBoost (financial + FinBERT emb.) | 0.9033 | 0.3313 | — | 0.2093 | 0.5000 | 0.2951 | 2607/136/36/36 |
| E: Financial encoder, single year | 0.9073 | 0.2953 | 0.249±0.035 | 0.2585 | 0.5278 | 0.3470 | 2634/109/34/38 |
| MLP / FNN (financial, 3 yr flattened) | 0.8970 | 0.2813 | — | 0.3472 | 0.3472 | 0.3472 | 2696/47/47/25 |
| D: Cross-modal attention (proposed) | 0.8856 | 0.2671 | 0.218±0.031 | 0.2273 | 0.4167 | 0.2941 | 2641/102/42/30 |
| XGBoost (financial) | 0.8930 | 0.2577 | — | 0.1726 | 0.5417 | 0.2617 | 2556/187/33/39 |
| C: Concatenation fusion | 0.8845 | 0.2467 | 0.211±0.022 | 0.1667 | 0.5000 | 0.2500 | 2563/180/36/36 |
| A: Financial encoder only (3 yr) | 0.8814 | 0.2339 | 0.186±0.042 | 0.2400 | 0.4167 | 0.3046 | 2648/95/42/30 |
| Logistic Regression (financial) | 0.8398 | 0.1688 | — | 0.1500 | 0.5833 | 0.2386 | 2505/238/30/42 |
| B: Text only (FinBERT + attention) | 0.7738 | 0.1409 | 0.132±0.011 | 0.1307 | 0.3611 | 0.1919 | 2570/173/46/26 |
| Logistic Regression (FinBERT emb.) | 0.7368 | 0.1270 | — | 0.1341 | 0.3056 | 0.1864 | 2601/142/50/22 |
