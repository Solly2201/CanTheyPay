# Experiment Results

Chronological company-disjoint splits: train 2001-2014 (12.8% positive), val 2015, test 2016-2018 (**2.6% positive**). Decision threshold tuned on validation F1; test touched once per model. Sorted by test PR-AUC (the primary metric under this imbalance).

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---|---|---|---|---|---|
| XGBoost (financial + FinBERT emb.) | 0.9033 | 0.3313 | 0.2093 | 0.5000 | 0.2951 | 2607/136/36/36 |
| MLP / FNN (financial, 1 yr) | 0.8970 | 0.2813 | 0.3472 | 0.3472 | 0.3472 | 2696/47/47/25 |
| XGBoost (financial) | 0.8930 | 0.2577 | 0.1726 | 0.5417 | 0.2617 | 2556/187/33/39 |
| C: Concatenation fusion | 0.8791 | 0.2426 | 0.2424 | 0.4444 | 0.3137 | 2643/100/40/32 |
| D: Cross-modal attention (proposed) | 0.8808 | 0.2068 | 0.2414 | 0.4861 | 0.3226 | 2633/110/37/35 |
| E: Cross-modal, single year | 0.8938 | 0.1941 | 0.2081 | 0.4306 | 0.2805 | 2625/118/41/31 |
| E: Financial encoder, single year | 0.8875 | 0.1820 | 0.2073 | 0.4722 | 0.2881 | 2613/130/38/34 |
| A: Financial encoder only (3 yr) | 0.8637 | 0.1702 | 0.1731 | 0.5000 | 0.2571 | 2571/172/36/36 |
| Logistic Regression (financial) | 0.8398 | 0.1688 | 0.1500 | 0.5833 | 0.2386 | 2505/238/30/42 |
| Logistic Regression (FinBERT emb.) | 0.7368 | 0.1270 | 0.1341 | 0.3056 | 0.1864 | 2601/142/50/22 |
| B: Text only (FinBERT + attention) | 0.7682 | 0.1229 | 0.1594 | 0.3056 | 0.2095 | 2627/116/50/22 |
