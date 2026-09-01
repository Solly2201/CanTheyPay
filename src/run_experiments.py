"""Run every baseline, the proposed model, and the ablation suite on the fixed
chronological splits; write experiments/results.json and results.md.

Ablations:
  A financial only        (neural + XGBoost/LogReg)
  B text only             (neural + LogReg on mean embedding)
  C simple concatenation  (neural concat; XGBoost fin+text)
  D cross-modal attention (proposed)
  E single-year vs 3-year history (financial-only and cross-modal)

Usage: python -m src.run_experiments
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EXPERIMENTS_DIR, SEED, set_seed
from src.dataset import BankruptcyDataset, numpy_views
from src.evaluate import evaluate_scores
from src.models.baselines import run_baselines
from src.train import predict, train_model

PRETTY = {
    "logreg_financial": "Logistic Regression (financial)",
    "xgboost_financial": "XGBoost (financial)",
    "xgboost_fin_text": "XGBoost (financial + FinBERT emb.)",
    "logreg_text": "Logistic Regression (FinBERT emb.)",
    "mlp_financial": "MLP / FNN (financial, 1 yr)",
    "financial_only": "A: Financial encoder only (3 yr)",
    "financial_only_y1": "E: Financial encoder, single year",
    "text_only": "B: Text only (FinBERT + attention)",
    "concat": "C: Concatenation fusion",
    "crossmodal": "D: Cross-modal attention (proposed)",
    "crossmodal_y1": "E: Cross-modal, single year",
}


def main():
    set_seed(SEED)
    results = {}

    print("Loading datasets ...")
    train_ds = BankruptcyDataset("train")
    val_ds = BankruptcyDataset("validation")
    test_ds = BankruptcyDataset("test")
    y_va, y_te = val_ds.labels, test_ds.labels

    print("\n=== Classical baselines ===")
    base = run_baselines(numpy_views(train_ds), numpy_views(val_ds),
                         numpy_views(test_ds), seed=SEED)
    for name, (s_va, s_te, _model) in base.items():
        results[name] = evaluate_scores(y_va, s_va, y_te, s_te)
        print(f"{name}: test PR-AUC {results[name]['test']['pr_auc']:.4f} "
              f"ROC-AUC {results[name]['test']['roc_auc']:.4f}")

    print("\n=== Neural models ===")
    neural_runs = [
        ("financial_only", 3, "financial_only"),
        ("text_only", 3, "text_only"),
        ("concat", 3, "concat"),
        ("crossmodal", 3, "crossmodal"),
        ("financial_only", 1, "financial_only_y1"),
        ("crossmodal", 1, "crossmodal_y1"),
    ]
    for model_name, n_years, key in neural_runs:
        print(f"\n--- {key} (n_years={n_years}) ---")
        tr = train_ds if n_years == 3 else BankruptcyDataset("train", n_years=1)
        va = val_ds if n_years == 3 else BankruptcyDataset("validation", n_years=1)
        te = test_ds if n_years == 3 else BankruptcyDataset("test", n_years=1)
        model = train_model(model_name, tr, va, n_years=n_years, seed=SEED)
        s_va, s_te = predict(model, va), predict(model, te)
        results[key] = evaluate_scores(y_va, s_va, y_te, s_te)
        print(f"{key}: test PR-AUC {results[key]['test']['pr_auc']:.4f} "
              f"ROC-AUC {results[key]['test']['roc_auc']:.4f}")

    # Also an MLP on flattened single-year financials (classic FNN baseline).
    print("\n--- mlp_financial (flat MLP baseline) ---")
    from sklearn.neural_network import MLPClassifier
    Xf_tr, _, y_tr, _ = numpy_views(train_ds)
    Xf_va, _, _, _ = numpy_views(val_ds)
    Xf_te, _, _, _ = numpy_views(test_ds)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500,
                        random_state=SEED, early_stopping=True)
    mlp.fit(Xf_tr, y_tr)
    results["mlp_financial"] = evaluate_scores(
        y_va, mlp.predict_proba(Xf_va)[:, 1], y_te, mlp.predict_proba(Xf_te)[:, 1])

    out = EXPERIMENTS_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")
    write_markdown(results)


def write_markdown(results):
    rows = []
    for key, res in results.items():
        t = res["test"]
        rows.append((PRETTY.get(key, key), t["roc_auc"], t["pr_auc"],
                     t["precision"], t["recall"], t["f1"], t["confusion_matrix"]))
    rows.sort(key=lambda r: -r[2])
    lines = [
        "# Experiment Results",
        "",
        "Chronological company-disjoint splits: train 2001-2014 (12.8% positive), "
        "val 2015, test 2016-2018 (**2.6% positive**). Decision threshold tuned "
        "on validation F1; test touched once per model. Sorted by test PR-AUC "
        "(the primary metric under this imbalance).",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | TN/FP/FN/TP |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, roc, pr, p, r, f1, cm in rows:
        lines.append(f"| {name} | {roc:.4f} | {pr:.4f} | {p:.4f} | {r:.4f} | "
                     f"{f1:.4f} | {cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} |")
    (EXPERIMENTS_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print("Saved experiments/results.md")


if __name__ == "__main__":
    main()
