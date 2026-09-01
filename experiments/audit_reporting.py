"""Audit D/E: three-way consistency check.
1. Every number in results.md must equal the value in results.json.
2. Ensemble + per-seed numbers in results.json must equal what the training
   run actually printed to run_log.txt.
Run: python experiments/audit_reporting.py"""
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
res = json.loads((root / "experiments/results.json").read_text())
md = (root / "experiments/results.md").read_text(encoding="utf-8")
log = (root / "experiments/run_log.txt").read_text(encoding="utf-8", errors="replace")

PRETTY = {
    "logreg_financial": "Logistic Regression (financial)",
    "xgboost_financial": "XGBoost (financial)",
    "xgboost_fin_text": "XGBoost (financial + FinBERT emb.)",
    "logreg_text": "Logistic Regression (FinBERT emb.)",
    "mlp_financial": "MLP / FNN",
    "financial_only": "A: Financial encoder only (3 yr)",
    "financial_only_y1": "E: Financial encoder, single year",
    "text_only": "B: Text only (FinBERT + attention)",
    "concat": "C: Concatenation fusion",
    "crossmodal": "D: Cross-modal attention (proposed)",
    "crossmodal_y1": "E: Cross-modal, single year",
}

problems = []

# --- 1. results.md rows vs results.json ---
for key, name in PRETTY.items():
    row = next((line for line in md.splitlines()
                if line.startswith(f"| {name}")), None)
    if row is None:
        problems.append(f"results.md: no row found for {key} ({name})")
        continue
    cells = [c.strip() for c in row.split("|")[1:-1]]
    t = res[key]["test"]
    expect = [f"{t['roc_auc']:.4f}", f"{t['pr_auc']:.4f}"]
    got = cells[1:3]
    if got != expect:
        problems.append(f"results.md {key}: ROC/PR {got} != json {expect}")
    cm = t["confusion_matrix"]
    cm_s = f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}"
    if cells[-1] != cm_s:
        problems.append(f"results.md {key}: CM {cells[-1]} != json {cm_s}")
    expect_prf = [f"{t['precision']:.4f}", f"{t['recall']:.4f}", f"{t['f1']:.4f}"]
    if cells[4:7] != expect_prf:
        problems.append(f"results.md {key}: P/R/F1 {cells[4:7]} != json {expect_prf}")

# --- 2. results.json vs run_log.txt ---
for key in ["financial_only", "text_only", "concat", "crossmodal",
            "financial_only_y1", "crossmodal_y1"]:
    m = re.search(rf"{key}: ensemble test PR-AUC ([0-9.]+) ROC-AUC ([0-9.]+) "
                  rf"\(per-seed ([0-9.]+)\+/-([0-9.]+)\)", log)
    if not m:
        problems.append(f"run_log: no ensemble line for {key}")
        continue
    t = res[key]["test"]
    ps = res[key]["per_seed_test_pr_auc"]
    for got, want, what in [
            (float(m.group(1)), t["pr_auc"], "ensemble PR-AUC"),
            (float(m.group(2)), t["roc_auc"], "ensemble ROC-AUC"),
            (round(float(m.group(3)), 4), ps["mean"], "per-seed mean"),
            (round(float(m.group(4)), 4), ps["std"], "per-seed std")]:
        if abs(got - want) > 5e-5:
            problems.append(f"{key} {what}: log {got} != json {want}")
    seed_vals = [round(float(x), 4) for x in
                 re.findall(rf"(?m)^  seed \d+: test PR-AUC ([0-9.]+)\r?$",
                            log.split(f"--- {key} ")[1].split("\n--- ")[0])]
    json_vals = [round(v, 4) for v in ps["values"]]
    if seed_vals != json_vals:
        problems.append(f"{key} per-seed values: log {seed_vals} != json {json_vals}")

# --- 3. thresholds must come from validation ---
for key, r in res.items():
    if r["val"]["threshold"] != r["test"]["threshold"]:
        problems.append(f"{key}: test threshold differs from val-tuned threshold")

print("PROBLEMS:" if problems else "ALL CONSISTENT")
for p in problems:
    print(" -", p)
