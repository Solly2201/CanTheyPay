# Final Research Review & Defense Guide

Written after implementation freeze and statistical audit. Nothing in this
document changes any model, dataset, split, or reported number. Its purpose is
to fix the *claims* at exactly the strength the evidence supports.

Key numbers referenced throughout (test split, 2016–2018, 2,815 companies,
72 positives = 2.6%):

| Model | PR-AUC | Evidence file |
|---|---|---|
| Cross-modal, single-year (5-seed ensemble) | 0.340 | results.json `crossmodal_y1` |
| XGBoost + FinBERT embeddings | 0.331 | `xgboost_fin_text` |
| Financial encoder, single-year (ensemble) | 0.295 | `financial_only_y1` |
| Cross-modal, 3-year (ensemble) | 0.267 | `crossmodal` |
| XGBoost financial-only | 0.258 | `xgboost_financial` |
| Concat fusion (ensemble) | 0.247 | `concat` |
| Financial encoder 3-year (ensemble) | 0.234 | `financial_only` |
| Text only (ensemble) | 0.141 | `text_only` |

Bootstrap (AUDIT.md §I): Transformer−XGB+FinBERT Δ+0.033 [−0.067,+0.135]
p≈0.52; Transformer−XGB-financial Δ+0.104 [−0.004,+0.218] p≈0.06;
XGB+FinBERT−XGB-financial Δ+0.071 [−0.008,+0.149] p≈0.08.
Leakage masking (AUDIT.md §F): text models lose ~0.02–0.03 PR-AUC; ordering
survives. Seed spread: cross-modal 1-yr single-seed 0.272±0.055.

---

## TASK 1 — Audit of every major scientific claim

Classification: **A** strongly supported · **B** supported, word cautiously ·
**C** not supported / removed.

| # | Claim (where) | Class | Required wording |
|---|---|---|---|
| 1 | "Text adds real signal on top of financials" (README finding 1) | **B** | Improvement is sizable and *direction-consistent in every configuration and all 30 seed runs* (+0.07 XGB, +0.03–0.05 neural), and survives leakage masking — but p≈0.08 at 72 positives. Say "consistent evidence, short of conventional significance", never "proves". |
| 2 | "Cross-attention beats concatenation at equal capacity" (README finding 2) | **B (weak)** | Ensemble 0.267 vs 0.247, but seed-means 0.218±0.031 vs 0.211±0.022 overlap heavily; pair never bootstrap-tested. Say "directionally favors cross-attention; effect is small and not statistically established". The earlier phrase "supporting the core hypothesis" was softened to "directionally consistent with" as a result of this audit. |
| 3 | "Cross-modal model competitive with the strongest classical baseline" (README finding 3, post-audit wording) | **A** | Correct as stated. The pre-audit version ("only one to edge out") was **C** and is gone: Δ p≈0.52, and single-seed mean (0.272) is *below* XGB+FinBERT (0.331). |
| 4 | "Single-year beats 3-year history *in this setup*" (finding 4) | **B→A as scoped** | Replicated across both architectures and most seeds. **A** if scoped to "explicit 3-year sequences did not help under this one-event-labeled-year design"; **C** if generalized to "temporal history is useless for bankruptcy prediction" — never say that. The single-year token still carries YoY growth features (t−1 information); the disclosure must stay attached. |
| 5 | "Text alone is weak" (finding 1) | **A** | 0.141 vs 0.258 financial-only XGBoost; smallest seed variance of all models. Scoped to *this* preprocessed (stopword/number-stripped, Items 1/5/7) text. |
| 6 | Attention/importance outputs show the model reading economically sensible signals (README, explain outputs) | **B** | Always with the "diagnostics, not explanations" framing already in the docs. Do not claim attention weights identify causes of bankruptcy. |
| 7 | Statistical statements in AUDIT.md §I | **A** | Already stated at correct strength. |
| 8 | Leakage characterization "conclusions survive masking" (AUDIT.md §F) | **A** | Quantified: −0.02/−0.03; masked XGB+text 0.308 still > XGB-financial 0.258. Keep the caveat that validation-side leakage (5/21 positives) mildly contaminated threshold tuning/early stopping. |
| 9 | Generalization to real companies (demo) | **B** | Demo predictions are "illustrative"; mega-caps and non-US-GAAP inputs are out-of-distribution; docs already warn. Never present demo outputs as validated predictions. |
| 10 | "Deep Learning was necessary" | **C — never claimed, never claim it** | The evidence supports "the formulation *motivates* representation learning (FinBERT is itself DL) and *permits* attention fusion", not necessity. XGBoost on DL embeddings is nearly as good. |
| 11 | Novelty claims | **B** | Contribution is a careful, leakage-audited, ablated *empirical study* on an open dataset — not a new architecture. ECL/FIN-CATree-adjacent work exists; say "we investigate", not "we are the first". |

## TASK 2 — Final research question

> **RQ:** *In one-year-ahead prediction of US public-company financial
> distress under realistic class imbalance and chronological evaluation, do
> (i) learned text representations from a pretrained financial language model
> (FinBERT) add predictive information to structured financial features, and
> (ii) does cross-modal attention fusion extract more of that information than
> simple concatenation or gradient-boosted trees over the same inputs?*

This asks about *incremental information and fusion mechanism*, not about DL
superiority, and every clause maps to an implemented experiment.

**Hypotheses:**

- **H1 (text incrementality):** Adding FinBERT representations of 10-K text
  improves PR-AUC over financial-only models.
  *Tested by:* A vs C/D (neural), XGB-financial vs XGB+FinBERT, bootstrap §I,
  leakage sensitivity §F.
  *Outcome:* **Supported directionally, statistically inconclusive** —
  positive in every configuration (+0.07 for XGBoost, +0.033/+0.045 neural
  ensembles), survives leakage masking, but p≈0.06–0.08 with 72 positives.
- **H2 (fusion mechanism):** Cross-modal attention (financial queries → text
  keys/values) outperforms concatenation at matched capacity.
  *Tested by:* C vs D, identical encoder blocks, 5 seeds each.
  *Outcome:* **Inconclusive, leaning positive** — ensembles 0.267 vs 0.247 and
  seed-means 0.218 vs 0.211 both favor attention, but effect ≪ seed noise and
  untested for significance.
- **H3 (temporal history):** Explicit 3-year financial sequences improve
  prediction over the latest year alone.
  *Tested by:* Ablation E (y1 vs y3 for both financial-only and cross-modal).
  *Outcome:* **Unsupported — reversed.** Single-year outperformed 3-year in
  both architectures (0.295 vs 0.234; 0.340 vs 0.267). An honest negative
  result, consistent with Riyanto et al. (2026).

## TASK 3 — "Why Deep Learning instead of XGBoost?"

**30-second answer:**
"XGBoost *is* our co-champion — on financial ratios alone it beats every
financial-only neural model we built, and we say so. But the question our
project asks can't even be posed without deep learning: the text modality
enters the system as FinBERT embeddings, which are the output of a pretrained
Transformer. Our strongest classical baseline — XGBoost + FinBERT — is
already a hybrid that depends on deep representation learning. What we tested
on top of that is whether a *learned fusion* — cross-modal attention between
financial state and disclosure text — extracts more from the same inputs than
feeding frozen embeddings to trees. Result: adding text helps every model by
a consistent margin, and the attention-fusion model matches the hybrid
baseline. Deep learning earned its place at the representation and fusion
layers, not as a replacement for trees on tabular data."

**2-minute answer:**
"Three points, all grounded in our actual results.

First, we concede the tabular case. On engineered financial ratios, tuned
XGBoost (PR-AUC 0.258) beat our 3-year neural financial encoder (0.234) and
was roughly matched by the single-year one (0.295 ensemble vs 0.281 MLP).
The literature says tree ensembles are the gold standard for static tabular
features, and our experiments reproduce that. We did not use deep learning
because trees are weak.

Second, the problem is not purely tabular. Half our input is 10-K narrative
text. There is no non-deep-learning path from raw filing text to a
competitive representation: our TF-IDF-era alternative would be keyword
counts, and prior work shows those lose semantic context. FinBERT — a BERT
model pretrained on SEC filings — is deep transfer learning, used frozen so
110M pretrained parameters serve as a fixed feature extractor while we train
only 40k–185k fusion parameters. Every text-using model in our study,
including the classical XGBoost+FinBERT baseline, stands on this deep
component. And it mattered: text lifted XGBoost from 0.258 to 0.331 PR-AUC
(+0.07, direction-consistent in all 30 seed runs, though p≈0.08 with only 72
test positives).

Third, fusion is a representation-learning question. Concatenating a
768-dimensional text vector with financial features treats the modalities as
independent blocks. Cross-modal attention lets the company's financial state
select *which* disclosure chunks matter — a leveraged firm's year-token can
weight refinancing-risk passages. That mechanism only exists in a
differentiable architecture. Empirically the gains were real but modest:
attention fusion beat concatenation directionally (0.267 vs 0.247) and the
full model was competitive with — statistically indistinguishable from —
XGBoost+FinBERT (0.340 vs 0.331, p≈0.52). Our claim is therefore measured:
deep learning was *necessary* for the text representation, *useful* for
fusion, and *not necessary* for the tabular branch — and we have the
ablations to show exactly which is which."

## TASK 4 — Architecture, component by component

**1. Financial year encoder** (Linear 20→64 + LayerNorm + GELU + dropout,
learned positional embeddings, one self-attention block over year-tokens).
*Does:* turns each fiscal year's 20 engineered ratios into a 64-d token;
lets years exchange information. *Why there:* gives the financial state a
token-shaped representation that can act as attention queries. *Why DL:*
the projection and cross-feature interactions are learned, not hand-crafted.
*Compared against:* LR, XGBoost, sklearn-MLP on identical features.
*Showed:* trees win the pure-tabular contest at 3 years (0.258 vs 0.234);
single-year neural encoder is competitive (0.295).

**2. FinBERT, frozen** (`yiyanghkust/finbert-pretrain`, mean-pooled last
hidden state per ≤256-word chunk, cached once). *Does:* converts each text
chunk into a 768-d contextual embedding. *Why there:* pretrained on SEC
filings — the exact register of our text. *Why DL:* it is a 110M-parameter
Transformer; transfer learning is the only practical way to get contextual
text representations from 403 positive training examples. *Compared
against:* nothing shallower (a TF-IDF arm was considered; the LR-on-FinBERT
baseline serves as the "simplest consumer of the same embeddings").
*Showed:* text-only is weak alone (0.141) but adds +0.07 to XGBoost —
embeddings carry incremental signal. *Why frozen:* fine-tuning 110M params on
403 positives invites overfitting and needs GPUs we don't have; freezing also
keeps every downstream model consuming *identical* text features, which makes
the fusion comparison clean.

**3. Temporal representation** (3 year-tokens + positional embeddings; the
single-year variant uses 1 token that still contains YoY growth features).
*Does:* encodes the trajectory. *Why there:* bankruptcy is theorized as
multi-year decay. *Why DL:* attention over year-tokens instead of recurrence.
*Compared against:* single-year variants (ablation E). *Showed:* the 3-year
sequence *hurt* (0.267 vs 0.340 cross-modal) — our most surprising finding,
reported as such.

**4. Text encoder with no-text token** (Linear 768→64, one masked
self-attention block, learned placeholder token for companies without text).
*Does:* compresses chunk embeddings into fusible tokens; makes all 2,815 test
companies scoreable without imputing fake text. *Compared against:* dropping
text-less companies (rejected — would change the test set per model).
*Showed:* enables the consistent-test-set evaluation the audit verified.

**5. Cross-modal attention** (MultiheadAttention: Q = financial year-tokens,
K/V = text tokens; residual + LayerNorm; one fusion self-attention block).
*Does:* each year's financial state retrieves the most relevant disclosure
chunks; attention maps are stored for diagnostics. *Why DL:* it is a learned,
input-dependent alignment — the defining Transformer mechanism. *Compared
against:* concatenation (model C) at matched capacity. *Showed:* directional
gain (0.267 vs 0.247), small relative to seed noise.

**6. Classification head + imbalance handling** (2-layer MLP on pooled
financial ⊕ text representations; weighted BCE with pos_weight≈6.8; early
stopping on validation PR-AUC; threshold tuned on validation F1).
*Compared against:* unweighted training was not separately ablated —
weighting is literature-mandated under 2.6% positives (Riyanto et al.'s
recall collapse). *Showed:* recall 0.42–0.54 on the minority class at tuned
thresholds, versus the degenerate all-negative classifier that 97.4% accuracy
would reward.

**Deliberately NOT used:** LSTM/GRU stacks (documented recall collapse under
imbalance on this dataset family; our 3-token window doesn't need recurrence);
GAN/WGAN-GP oversampling (unstable, unnecessary — weighted loss + threshold
achieved the same goal); DQN/RL (offline supervised problem; RL formulation
added instability in prior work for no benefit); GNNs (no inter-firm network
data in any free dataset); FinBERT fine-tuning/LoRA (overfitting risk,
compute, and it would break the identical-embeddings design); multi-task
learning (label sparsity, gradient competition); SMOTE (interpolating
financial trajectories fabricates unrealistic firms and can leak).

## TASK 5 — Final results interpretation

**Observed performance.** The cross-modal single-year ensemble is the best
single number (PR-AUC 0.340, ROC-AUC 0.914), a hair above XGBoost+FinBERT
(0.331) and clearly above financial-only XGBoost (0.258). Every text-using
model beats its financial-only counterpart; text alone is far below
financials alone.

**Statistical evidence.** With 72 positives, PR-AUC confidence intervals span
roughly ±0.10. The Transformer-vs-XGBoost+FinBERT difference is noise
(p≈0.52). The two text-effect comparisons are +0.07/+0.10 with p≈0.06–0.08 —
consistent in sign everywhere, but below the bar for a significance claim.
Correct summary: *"consistent directional evidence that text adds
information; no statistical separation between the best neural and best
classical multimodal models."*

**Seed variance.** Single-seed cross-modal runs span 0.194–0.361
(0.272±0.055). The ensemble, not any typical single run, is what reaches
0.340 — single-seed averages do *not* beat XGBoost+FinBERT. Any presentation
must say the headline is a 5-seed probability ensemble.

**Leakage sensitivity.** ~12% of bankrupt test companies' text contains
post-petition language (10-K filed after the Chapter petition). Masking those
six companies' text costs text models ~0.02–0.03 PR-AUC and changes no
ordering. Reported tables use the unmasked evaluation for benchmark
comparability, caveat attached. Validation-side leakage (5/21 positives)
mildly contaminated threshold tuning.

**The 3-year result, honestly.** Three non-exclusive explanations, none of
which is "temporal information is useless": (i) the event-label design labels
only the final pre-filing year, so t−2/t−1 feature vectors add inputs without
adding label-relevant variation, diluting the signal; (ii) the single-year
token already carries YoY growth features, capturing the cheapest trajectory
information; (iii) with 403 training positives, the extra capacity needed to
use two more year-tokens costs more in variance than it buys in signal —
consistent with Riyanto et al.'s finding that non-sequential models beat
sequence models on this dataset family. XGBoost's own gain importance
independently concentrates on year-t features.

**Why PR-AUC.** At a 2.6% positive rate, accuracy is meaningless (predicting
"healthy" always scores 97.4%) and ROC-AUC is inflated by the huge
true-negative pool. PR-AUC tracks the precision/recall trade-off on the
minority class only; the random-classifier baseline equals the positive rate
(≈0.026), so 0.34 is ≈13× base rate. It is the metric that collapses when a
model ignores bankruptcies — which is exactly the failure mode to guard
against.

**Limitations (carry into the report verbatim from README/AUDIT):**
anonymized training data (no EDGAR cross-audit, restatements undetectable);
Chapter 7/11-only labels; preprocessed Items 1/5/7 text, not Item 1A; 68%
text coverage; severe train→test regime shift (12.8%→2.6% positives); 72 test
positives limit statistical power; FinBERT pretraining era overlaps the test
window (unsupervised contamination, no label access).

## TASK 6 — Report structure (section → content → supporting files)

1. **Abstract** — RQ from Task 2, dataset (6,190 US companies, 1999–2019),
   method (frozen FinBERT + cross-modal attention, 5-seed ensembles,
   chronological splits), headline numbers with the non-significance
   statement, H3 negative finding. *Files:* README results table, AUDIT §I.
2. **Introduction** — cost of corporate failure, lagging nature of accounting
   ratios, forward-looking disclosures; end with the RQ and contributions
   list (empirical study + leakage audit + ablations + open pipeline).
   *Files:* README intro; your literature notes.
3. **Problem Statement** — one-year-ahead binary distress prediction from
   (financials, text) at year t; formal setup, 2.6% positive rate, why
   accuracy fails. *Files:* DATA_STRATEGY §9–10, 12.
4. **Literature Review** — use your existing 12-paper systematic review
   (Altman → ML → DL → financial NLP → temporal → attention → multimodal).
   *Files:* your literature-comparison matrix.xlsx and review documents.
5. **Research Gap** — deep contextual text encoders + true Q/K/V cross-modal
   attention absent from the empirical bankruptcy corpus; sequential models'
   recall collapse under imbalance. *Files:* your gap-analysis notes;
   ARCHITECTURE "what we did not use".
6. **Methodology** — pipeline diagram (download → features → FinBERT cache →
   models → evaluation), imbalance strategy, 5-seed protocol, threshold
   protocol. *Files:* README "Reproduce everything", ARCHITECTURE, train.py,
   evaluate.py.
7. **Dataset** — source, license, split table, feature list, text stats,
   label definition, alignment verification, *and the leakage audit summary*.
   *Files:* DATA_STRATEGY (all), AUDIT §A–C, F, dataset_stats.json.
8. **Architecture** — Task 4 content with the block diagram; parameter counts
   (40k–185k trainable; FinBERT frozen). *Files:* ARCHITECTURE.md,
   models/multimodal.py.
9. **Experimental Setup** — hyperparameters (config.py), hardware (CPU
   laptop; embedding ~1h one-time), seeds, exact metric definitions.
   *Files:* config.py, run_experiments.py.
10. **Results** — full table from experiments/results.md including per-seed
    columns; bootstrap CIs table from AUDIT §I immediately alongside. Never
    present the point estimates without the intervals. *Files:* results.md,
    results.json, AUDIT §I.
11. **Ablation Study** — H1/H2/H3 mapped to A–E with verdicts (Task 2 table);
    the 3-year negative result gets its own subsection with the three
    candidate explanations. *Files:* results.md rows A/B/C/D/E.
12. **Explainability** — permutation importance, XGBoost gain, attention
    examples (material weakness / liquidity / refinancing chunks); the
    explicit "diagnostics ≠ explanations" paragraph. *Files:*
    explain_financial_importance.csv, explain_xgb_importance.csv,
    explain_attention_examples.md.
13. **Limitations** — Task 5 limitations list + leakage quantification +
    statistical power. *Files:* README limitations, AUDIT §E–F, I.
14. **Conclusion** — answer the RQ at audited strength: text adds consistent
    directional value; attention fusion is competitive with, not superior to,
    the strongest classical hybrid; explicit temporal sequences did not help
    here.
15. **Future Work** — Option B scale-up (ECL + EDGAR-CORPUS + FSDS, real
    CIKs, Item 1A), LoRA fine-tuning on GPU, saving per-seed predictions to
    bootstrap ensembles directly, sector-conditional analysis, distress
    labels beyond Chapter filings. *Files:* DATA_STRATEGY Option B.
16. **References** — your 26-paper corpus + dataset papers (Lombardo 2022,
    Pellegrino 2024) + FinBERT (Yang et al.) + EDGAR-CORPUS (Loukas 2021).

## TASK 7 — 20 likely viva questions with ideal answers

**Q1. Why deep learning and not just XGBoost?**
Use the Task 3 answer. Core line: XGBoost wins the tabular contest and we
show it; the text representation *is* deep learning (FinBERT), so even our
strongest classical baseline is a DL hybrid; our contribution is testing
where representation learning pays — answer: at the text and fusion layers,
not the tabular layer.

**Q2. Your Transformer barely beats XGBoost+FinBERT. Why bother?**
"Correct — 0.340 vs 0.331, p≈0.52, and we report it as statistically
indistinguishable. The value of the experiment is the *finding itself*: on
6k companies, learned cross-modal fusion matches but does not surpass frozen
embeddings + trees. That is a useful, honest data point the literature
lacks, and our ablations localize where the gains actually come from (text
incrementality, not fusion mechanism)."

**Q3. Why FinBERT rather than generic BERT or TF-IDF?**
"Domain match: `finbert-pretrain` was pretrained on 10-Ks and earnings calls
— the register of our inputs. Generic BERT would carry the same architecture
with worse-matched vocabulary statistics; TF-IDF discards word order,
negation, and context, which prior bankruptcy-text work identifies as the
binding constraint. We used it frozen so all models consume identical text
features — an experimental-control decision, not just a compute one."

**Q4. Why a Transformer at all — why not an MLP over everything?**
"Our year and chunk inputs are naturally *sets of tokens* of varying number
(0–16 text chunks). Attention with masking handles variable-length input and
gives input-dependent weighting; an MLP needs fixed-size input and fixed
interactions. And model C (concat + MLP head) *is* that alternative — it
scored 0.247 vs 0.267."

**Q5. Why cross-attention specifically?**
"It encodes the hypothesis that the relevance of a disclosure passage depends
on the firm's financial state — Q from financial tokens, K/V from text. The
concat ablation is the null of that hypothesis. Result: directional support
(0.267 vs 0.247) but within seed noise, and we say so."

**Q6. Why multimodal at all? Maybe ratios already contain everything.**
"That's H1, and we tested it four ways. Text lifted XGBoost +0.07 and every
neural variant +0.03–0.05, direction-consistent across all 30 seed runs, and
it survives masking the leaky documents. It falls just short of α=0.05
because we have 72 test positives — an evidence-strength statement, not an
absence of effect."

**Q7. Why not an LSTM for the temporal part?**
"Two reasons: our window is three tokens, where recurrence has no advantage;
and Riyanto et al. (2026) showed LSTMs collapse to near-zero minority recall
on exactly this dataset family under imbalance. Our E-ablation then showed
even attention over 3 years underperforms single-year — the temporal
bottleneck here is the labeling design, not the sequence operator."

**Q8. Why only two modalities? No market data, no networks?**
"Scope discipline. Market data would mostly duplicate the market-value item
already in our features; inter-firm graph data (guarantee/supply networks)
does not exist in any free dataset — the papers using it rely on proprietary
Chinese registries. Adding modalities without data would have been
decoration."

**Q9. Why did 3-year history perform worse?**
Use Task 5's three explanations: label design dilutes older years; YoY growth
features already capture cheap trajectory signal; extra capacity costs
variance at 403 training positives. Emphasize: replicated in both
architectures, corroborated by XGBoost importance concentrating on year-t,
consistent with published results — and scoped to this design, not a general
claim that history is useless.

**Q10. Your improvement isn't statistically significant. Is the project a
failure?**
"No — the project's question was whether multimodal representation learning
adds information, not whether we could beat XGBoost by a headline. We report
effect sizes with paired-bootstrap CIs, refuse the significance claim the
data can't support, quantify our own leakage, and publish a negative temporal
finding. That is what a defensible empirical study looks like when the test
set has 72 events."

**Q11. What does PR-AUC mean and why prioritize it?**
"Area under the precision-recall curve: average precision across recall
levels for the positive class. Under 2.6% positives, accuracy is trivially
97.4% for a useless model and ROC-AUC is cushioned by 2,743 easy negatives.
PR-AUC's random baseline equals the positive rate, 0.026 — so 0.34 is ~13×
chance, and the metric collapses if the model ignores bankruptcies."

**Q12. Why only 72 positive test cases? Couldn't you get more?**
"Chronological integrity. The test window 2016–2018 is a benign credit
period — 2.6% bankruptcy incidence is the true base rate of that era. We
could inflate positives only by mixing crisis-era years into test, which
would break the shipped chronological benchmark split and leak regime
information. We kept the honest split and paid for it in statistical power."

**Q13. What is data leakage, and what did you check?**
"Any channel by which information unavailable at prediction time influences
training or evaluation. We audited seven channels: company overlap across
splits (zero), duplicate company-years (zero), preprocessing statistics
(train-fit only, verified bit-exact), threshold provenance (validation-only),
test-set consistency (identical 2,815 rows for all models), label
construction (one event-labeled row per company — avoiding the company-level
label leak we found in the related Kaggle dataset), and post-event text."

**Q14. Explain the post-petition leakage issue.**
"Some bankrupt firms filed their year-t 10-K *after* their Chapter petition,
so the document mentions the bankruptcy as fact — ~12% of bankrupt test
companies with text. We found it by phrase-scanning, reviewed hits manually,
and re-scored with those texts masked: text models lose ~0.02–0.03 PR-AUC
and no ordering changes. It's inherent to the anonymized source (we can't
check filing dates), so we quantified it instead of pretending it away."

**Q15. How do you know the model isn't just learning bankruptcy keywords?**
"Partly it may be — that's exactly what the masking experiment bounds:
removing the documents where 'bankruptcy as fact' language exists leaves the
text advantage intact (masked XGB+text 0.308 vs financial 0.258). The
attention diagnostics also show mass on going-concern-type content — material
weaknesses, covenant and refinancing risk — which is legitimate *pre-event*
language. But we can't fully separate lexical shortcuts from semantics in an
anonymized corpus, and we list that as a limitation."

**Q16. What's the actual research contribution?**
"Four things: (1) a leakage-audited empirical comparison of fusion mechanisms
for bankruptcy prediction on an open dataset, with 5-seed uncertainty and
bootstrap CIs — rigor that's rare in this corpus; (2) evidence that frozen
FinBERT embeddings carry incremental distress signal usable even by trees;
(3) a replicated negative result on explicit multi-year sequences under
event-labeling; (4) a fully reproducible open pipeline including the audit
scripts themselves."

**Q17. Is this really novel? FIN-CATree, ECL, DGCN-TA exist.**
"We don't claim architectural novelty. FIN-CATree applies attention to
tabular-only Taiwan data; ECL pairs text with Compustat behind a paywall;
DGCN-TA has no text. The niche we occupy is the *controlled comparison* —
same inputs, same splits, same embeddings across all fusion strategies, with
uncertainty quantification and a self-audit. Novelty of evidence, not of
mechanism."

**Q18. Why is your validation set only 233 companies? Isn't threshold tuning
on 27 positives unstable?**
"Yes, and it's why we moved to 5-seed ensembles after observing single-seed
variance, and why we report validation-tuned thresholds as a protocol choice
inherited from the dataset's published split rather than re-cutting the data.
The alternative — enlarging validation by shrinking train or test — would
break comparability with the published benchmarks on this dataset."

**Q19. Would you deploy this?**
"No. It's a research prototype: anonymized training data, one label type
(Chapter filings), text coverage 68%, no calibration analysis, and a demo
explicitly labeled not-financial-advice that refuses companies without
sufficient public data. Deployment would need the Option B rebuild on real
identifiers plus calibration and monitoring."

**Q20. What would you change with more data/compute?**
"Ranked: (1) rebuild on real CIKs via ECL + EDGAR-CORPUS + SEC Financial
Statement Data Sets so filing dates kill the post-petition leakage class and
Item 1A becomes available; (2) save per-seed predictions so ensembles can be
bootstrapped directly; (3) widen the label to distress events beyond Chapter
filings; (4) LoRA-tune FinBERT's top layers on GPU; (5) quarterly data to
re-test H3 with a properly longitudinal design."

## TASK 8 — Final verdict

**Genuinely strong:**
- End-to-end reproducibility on a laptop, from download to demo, with fixed
  seeds and cached embeddings.
- The evaluation discipline: chronological company-disjoint splits, identical
  test set for all 11 models, train-only preprocessing, validation-only
  thresholds — all *verified by audit scripts that ship with the repo*.
- The self-audit itself: finding and quantifying your own leakage, and
  bootstrap-testing your own headline, is what will most distinguish this
  project.
- A clean ablation design where every DL component has a matched-capacity
  null (A/B/C/D/E), producing one positive, one inconclusive, and one honest
  negative finding.
- Working real-world demo with principled refusal behavior.

**Biggest weaknesses:**
- 72 test positives → the study is underpowered for its central comparison;
  nothing beats XGBoost+FinBERT with statistical support.
- Anonymized dataset → restatements and filing dates unauditable; the
  post-petition leakage can be bounded but not removed.
- Text is preprocessed Items 1/5/7, not Item 1A Risk Factors; stopword
  stripping degrades FinBERT's input distribution.
- The headline depends on seed ensembling; typical single runs trail the
  classical hybrid.
- H2 (the architecturally distinctive claim) is the least well-supported.

**Claims to make:** text adds consistent directional signal (+0.07 XGBoost,
robust to leakage masking); the cross-modal Transformer is competitive with
the strongest classical hybrid; explicit 3-year sequences did not help under
event-labeling; the pipeline is leakage-audited and reproducible.

**Claims to avoid:** "the Transformer beats XGBoost"; any unqualified
statistical-significance language; "attention explains the predictions";
"temporal information is useless"; "this generalizes to real companies /
other markets"; "deep learning was necessary" (it was necessary *for the
text representation*, and only there).

**Strong enough for a college DL project?** Yes, comfortably — provided it is
presented as an evidence-audited empirical study. It contains real transfer
learning, a real Transformer fusion mechanism, a real ablation program,
imbalance-aware evaluation, uncertainty quantification, and an honest
negative result. Weak projects claim more from less; this one claims less
than it could and can prove everything it claims.

**The 3 things to internalize before presenting:**
1. **Where DL earns its place** — necessary for text representation
   (FinBERT), useful-but-unproven for fusion, unnecessary for tabular. If you
   can draw that line crisply, Q1–Q5 are safe.
2. **The statistics** — why PR-AUC, why 72 positives caps power, what the
   paired bootstrap does, and why "consistent direction, p≈0.06–0.08" is
   neither significance nor nothing.
3. **The leakage story end-to-end** — the seven audited channels, the
   post-petition finding, the masking bound, and why the conclusions survive.
   Owning your own flaw before the professor finds it converts your biggest
   weakness into your strongest moment.
