"""Uncertainty quantification for the headline PR-AUC comparisons.

Method: paired bootstrap over test companies (resample the 2,815 test rows
with replacement, B=10,000, fixed RNG), computing PR-AUC for each model on the
same replicate, so the difference distribution accounts for the shared test
sample. No model is retrained:
  - XGBoost predictions are reconstructed by deterministic refit (verified in
    audit_recompute.py to reproduce results.json exactly).
  - The cross-modal single-year model is the SAVED best-validation-seed
    checkpoint (test PR-AUC 0.3606). The headline 0.340 is a 5-seed
    probability ensemble whose member predictions were not saved, so the
    ensemble itself cannot be bootstrapped without retraining; its seed-level
    variation is reported from results.json instead.

Run: python experiments/audit_uncertainty.py
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from sklearn.metrics import average_precision_score

from src.config import SEED, set_seed
from src.dataset import BankruptcyDataset, numpy_views
from src.models.baselines import make_xgb
from src.models.multimodal import CrossModalModel
from src.train import predict

B = 10_000
set_seed(SEED)

train_ds = BankruptcyDataset("train")
Xf_tr, Xt_tr, y_tr, _ = numpy_views(train_ds)
pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()
del train_ds
gc.collect()

test_ds3 = BankruptcyDataset("test")            # 3-yr views for XGBoost (60 dims)
Xf_te, Xt_te, y_te, _ = numpy_views(test_ds3)
test_ds1 = BankruptcyDataset("test", n_years=1)  # 1-yr for the neural checkpoint
assert (test_ds1.labels == y_te).all()
del test_ds3
gc.collect()

print("Reconstructing XGBoost predictions (deterministic refit) ...")
xgb_fin = make_xgb(pos_w, SEED).fit(Xf_tr, y_tr)
s_xgb_fin = xgb_fin.predict_proba(Xf_te)[:, 1]
xgb_mm = make_xgb(pos_w, SEED).fit(np.hstack([Xf_tr, Xt_tr]), y_tr)
s_xgb_mm = xgb_mm.predict_proba(np.hstack([Xf_te, Xt_te]))[:, 1]

ckpt = torch.load(root / "experiments/checkpoints/crossmodal_y1.pt",
                  map_location="cpu", weights_only=False)
model = CrossModalModel(n_features=ckpt["n_features"], d_model=ckpt["d_model"],
                        n_years=1)
model.load_state_dict(ckpt["state_dict"])
model.eval()
s_cm = predict(model, test_ds1)

res = json.loads((root / "experiments/results.json").read_text())
checks = [
    ("xgboost_financial refit", average_precision_score(y_te, s_xgb_fin),
     res["xgboost_financial"]["test"]["pr_auc"]),
    ("xgboost_fin_text refit", average_precision_score(y_te, s_xgb_mm),
     res["xgboost_fin_text"]["test"]["pr_auc"]),
]
for name, got, want in checks:
    ok = abs(got - want) < 5e-5
    print(f"  {name}: PR-AUC {got:.4f} (results.json {want:.4f}) "
          f"{'OK' if ok else 'MISMATCH - abort'}")
    assert ok
pr_cm = average_precision_score(y_te, s_cm)
print(f"  crossmodal_y1 saved checkpoint: PR-AUC {pr_cm:.4f} "
      f"(best seed in results.json: "
      f"{max(res['crossmodal_y1']['per_seed_test_pr_auc']['values']):.4f})")

print(f"\nPaired bootstrap, B={B} ...")
rng = np.random.default_rng(SEED)
n = len(y_te)
boots = {"crossmodal_y1_ckpt": [], "xgb_fin_text": [], "xgb_fin": []}
for _ in range(B):
    idx = rng.integers(0, n, n)
    if y_te[idx].sum() == 0:
        continue
    boots["crossmodal_y1_ckpt"].append(average_precision_score(y_te[idx], s_cm[idx]))
    boots["xgb_fin_text"].append(average_precision_score(y_te[idx], s_xgb_mm[idx]))
    boots["xgb_fin"].append(average_precision_score(y_te[idx], s_xgb_fin[idx]))
boots = {k: np.array(v) for k, v in boots.items()}


def ci(a, lo=2.5, hi=97.5):
    return np.percentile(a, lo), np.percentile(a, hi)


print("\nPer-model test PR-AUC with 95% bootstrap CI:")
for k, point in [("crossmodal_y1_ckpt", pr_cm),
                 ("xgb_fin_text", checks[1][1]), ("xgb_fin", checks[0][1])]:
    l, h = ci(boots[k])
    print(f"  {k}: {point:.4f}  [{l:.4f}, {h:.4f}]")

print("\nPaired PR-AUC differences (same replicates):")
for a, b in [("crossmodal_y1_ckpt", "xgb_fin_text"),
             ("crossmodal_y1_ckpt", "xgb_fin"),
             ("xgb_fin_text", "xgb_fin")]:
    d = boots[a] - boots[b]
    l, h = ci(d)
    p_two = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"  {a} - {b}: mean {d.mean():+.4f}  95% CI [{l:+.4f}, {h:+.4f}]  "
          f"two-sided bootstrap p ~= {min(p_two, 1.0):.3f}")

print("\n5-seed variation already on record (single-run test PR-AUC, "
      "from results.json):")
for k in ["crossmodal_y1", "financial_only_y1", "crossmodal", "concat",
          "financial_only", "text_only"]:
    ps = res[k]["per_seed_test_pr_auc"]
    print(f"  {k}: mean {ps['mean']:.4f} +/- {ps['std']:.4f}  "
          f"values {ps['values']}  (ensemble {res[k]['test']['pr_auc']:.4f})")
