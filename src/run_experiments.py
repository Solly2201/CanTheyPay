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
import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds per neural model; test scores are "
                         "the mean predicted probability across seeds "
                         "(seed ensemble), with per-seed PR-AUC spread reported")
    args = ap.parse_args()
    seeds = [SEED + i for i in range(args.seeds)]
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
    from sklearn.metrics import average_precision_score
    for model_name, n_years, key in neural_runs:
        print(f"\n--- {key} (n_years={n_years}, {len(seeds)} seeds) ---")
        tr = train_ds if n_years == 3 else BankruptcyDataset("train", n_years=1)
        va = val_ds if n_years == 3 else BankruptcyDataset("validation", n_years=1)
        te = test_ds if n_years == 3 else BankruptcyDataset("test", n_years=1)
        sv, st, per_seed, models = [], [], [], []
        for seed in seeds:
            model = train_model(model_name, tr, va, n_years=n_years, seed=seed,
                                verbose=False)
            s_va, s_te = predict(model, va), predict(model, te)
            sv.append(s_va)
            st.append(s_te)
            models.append(model)
            per_seed.append(float(average_precision_score(y_te, s_te)))
            print(f"  seed {seed}: test PR-AUC {per_seed[-1]:.4f}")
        # keep the checkpoint of the best-validation seed for the demo/explain
        best_i = int(np.argmax([average_precision_score(y_va, s) for s in sv]))
        import torch
        from src.config import MODELS_DIR, D_MODEL
        torch.save({"state_dict": models[best_i].state_dict(),
                    "model": model_name, "n_years": n_years,
                    "n_features": tr.n_features, "d_model": D_MODEL},
                   MODELS_DIR / f"{model_name}_y{n_years}.pt")
        s_va, s_te = np.mean(sv, axis=0), np.mean(st, axis=0)
        results[key] = evaluate_scores(y_va, s_va, y_te, s_te)
        results[key]["per_seed_test_pr_auc"] = {
            "mean": round(float(np.mean(per_seed)), 4),
            "std": round(float(np.std(per_seed)), 4), "values": per_seed}
        print(f"{key}: ensemble test PR-AUC {results[key]['test']['pr_auc']:.4f} "
              f"ROC-AUC {results[key]['test']['roc_auc']:.4f} "
              f"(per-seed {np.mean(per_seed):.4f}+/-{np.std(per_seed):.4f})")

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
        spread = res.get("per_seed_test_pr_auc")
        spread_s = (f"{spread['mean']:.3f}±{spread['std']:.3f}" if spread else "—")
        rows.append((PRETTY.get(key, key), t["roc_auc"], t["pr_auc"], spread_s,
                     t["precision"], t["recall"], t["f1"], t["confusion_matrix"]))
    rows.sort(key=lambda r: -r[2])
    lines = [
        "# Experiment Results",
        "",
        "Chronological company-disjoint splits: train 2001-2014 (12.8% positive), "
        "val 2015, test 2016-2018 (**2.6% positive**). Decision threshold tuned "
        "on validation F1; test touched once per model. Neural models are "
        "5-seed probability ensembles; the per-seed column shows single-run "
        "PR-AUC mean±std. Sorted by test PR-AUC (the primary metric under this "
        "imbalance).",
        "",
        "| Model | ROC-AUC | PR-AUC | per-seed PR-AUC | Precision | Recall | F1 | TN/FP/FN/TP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, roc, pr, spread_s, p, r, f1, cm in rows:
        lines.append(f"| {name} | {roc:.4f} | {pr:.4f} | {spread_s} | {p:.4f} | "
                     f"{r:.4f} | {f1:.4f} | "
                     f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} |")
    (EXPERIMENTS_DIR / "results.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    print("Saved experiments/results.md")


if __name__ == "__main__":
    main()
