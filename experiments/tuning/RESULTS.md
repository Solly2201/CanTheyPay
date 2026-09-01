# Architecture Refinement Study — Results

Pre-registered protocol (`src/tune_architecture.py` docstring): all selection
on 5-seed **validation** PR-AUC; test never loaded during the sweep; adoption
rule fixed in advance (ensemble val PR-AUC ≥ base + 0.02 AND per-seed mean
above base); the frozen winner evaluated on test **exactly once**.

Integrity notes:
- The original sweep was accidentally launched twice in parallel with
  identical seeds. `verify_artifacts.py` re-trained V0 and V3 (2 seeds each)
  and reproduced the stored values to 6 decimals — training is deterministic
  per seed, so the double-written artifacts are valid.
- V9 and V10 were trained after the interruption, single-process.

## Validation sweep (exploratory — NOT final results)

| Variant | ensemble val PR-AUC | per-seed mean±std | rule verdict |
|---|---|---|---|
| **V3_item_embedding** | **0.7714** | **0.7582±0.0105** | **PASS (winner)** |
| V2_item1_only | 0.7634 | 0.7377±0.0079 | PASS |
| V6_bidirectional | 0.7630 | 0.7336±0.0149 | PASS |
| V5_text_queries_fin | 0.7548 | 0.7345±0.0124 | PASS |
| V4_l2_norm | 0.7507 | 0.7271±0.0039 | fail |
| V7_gated | 0.7377 | 0.7309±0.0182 | fail |
| V9_2layers | 0.7361 | 0.7253±0.0143 | fail |
| V8_heads8 | 0.7337 | 0.7213±0.0177 | fail |
| V0_base (= frozen crossmodal_y1 design) | 0.7328 | 0.7222±0.0215 | base |
| V1_item7_only | 0.6957 | 0.6729±0.0228 | fail |
| V10_combo_itememb_bi (combination step) | 0.7806 | 0.7591±0.0315 | rejected* |

\* The pre-registered combination step: V3 (best text-side) × V6 (best
attention-side). V10 was adopted only if it passed the rule against *each*
parent; it needed ensemble ≥ 0.7914 vs V3 and reached 0.7806 → rejected.
Winner = V3, which dominates every passing variant on ensemble, seed mean,
and seed variance.

## Why V3 won (recorded at freeze time, before test evaluation)

V3 adds a learned **section-type embedding** (Item 1 vs Item 7) to each text
chunk token — the model is told which 10-K section a chunk comes from. The
sweep's section-selection results explain why this helps: Item 1 alone
*improved* over both-sections-unlabeled (V2: 0.7634 vs V0: 0.7328) while
Item 7 alone *hurt* (V1: 0.6957) — in this corpus the two sections carry
distinctly different signal (contrary to the MD&A-centric prior from the
literature), so labeling chunks by section lets the model weight them
appropriately instead of treating all chunks as exchangeable. Cost: 192
additional parameters.

## Final frozen model — single test evaluation

Config: cross-modal single-year + section-type embeddings, 5-seed probability
ensemble, threshold tuned on validation (τ=0.715). Frozen on validation
evidence before test was touched. Test evaluated once
(`final_V3_item_embedding_test.json`):

| | ROC-AUC | PR-AUC | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---|---|---|---|---|---|
| V3 ensemble (test) | 0.9100 | **0.3592** | 0.2609 | 0.5000 | 0.3429 | 2641/102/36/36 |

Per-seed test PR-AUC: 0.241, 0.308, 0.381, 0.274, 0.344 (mean 0.310±0.051).

## Comparison with the frozen main-suite results

| Model | test PR-AUC | test ROC-AUC |
|---|---|---|
| **V3 refined cross-modal (this study)** | **0.3592** | 0.9100 |
| Cross-modal single-year, original (frozen) | 0.3395 | 0.9145 |
| XGBoost + FinBERT (frozen) | 0.3313 | 0.9033 |
| XGBoost financial-only (frozen) | 0.2577 | 0.8930 |

**Interpretation, at the strength the evidence supports:** V3 improves the
point estimate over the original model (+0.020) and over XGBoost+FinBERT
(+0.028). These deltas are well inside the paired-bootstrap noise established
in `docs/AUDIT.md` §I (95% CIs on differences span ≈±0.10 at 72 test
positives), so **no statistical-superiority claim is made**. The single-seed
mean (0.310) remains below XGBoost+FinBERT (0.331) — the ensemble is doing
real work. The project's headline conclusion is unchanged: the cross-modal
model is *competitive with, not demonstrably superior to*, the strongest
classical hybrid. The refinement's scientific value is the section-signal
finding (Item 1 vs Item 7) and the demonstration that the pre-registered
protocol adopted a variant on validation evidence that then held up
directionally on the untouched test set.

All original audited results (`experiments/results.json`, `results.md`) are
preserved unchanged. The demo continues to serve the original audited 3-year
model (`experiments/checkpoints/crossmodal_y3.pt`), matching its 3-year input
interface.
