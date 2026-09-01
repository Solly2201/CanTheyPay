"""Audit F2: sensitivity of text-using models to post-petition text leakage.

Six test-split label=1 companies contain bankruptcy-as-fact language in their
year-t 10-K text (identified by audit_text_leakage.py; C5537 is a likely false
positive - trade petitions - and C5916 refers to competitors, but we mask all
six for a conservative bound). We re-score the text-using models with those
companies' text REMOVED (they fall back to the no-text pathway) and report the
change in test PR-AUC. Training data is untouched - this bounds the
evaluation-side effect only.
Run: python experiments/audit_sensitivity.py"""
import gc
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import SEED, set_seed
from src.dataset import BankruptcyDataset, numpy_views
from src.models.baselines import make_xgb
from src.models.multimodal import CrossModalModel
from src.train import predict

LEAKY_TEST_CIKS = {"C3492", "C3842", "C5537", "C5847", "C5898", "C5916"}

set_seed(SEED)
train_ds = BankruptcyDataset("train")
test_ds = BankruptcyDataset("test")
y_te = test_ds.labels
leak_idx = np.array([str(c) in LEAKY_TEST_CIKS for c in test_ds.cik])
print(f"masking text of {leak_idx.sum()} test companies "
      f"(all label=1: {test_ds.labels[leak_idx].tolist()})")


def masked_copy(ds):
    import copy
    d2 = copy.copy(ds)
    d2.text = ds.text.copy()
    d2.mask = ds.mask.copy()
    d2.text[leak_idx] = 0.0
    d2.mask[leak_idx] = 0.0
    return d2


test_masked = masked_copy(test_ds)

# --- XGBoost fin+text (retrained deterministically, same seed as suite) ---
Xf_tr, Xt_tr, y_tr, _ = numpy_views(train_ds)
xgb = make_xgb((y_tr == 0).sum() / (y_tr == 1).sum(), SEED).fit(
    np.hstack([Xf_tr, Xt_tr]), y_tr)
for tag, ds in [("original", test_ds), ("text-masked", test_masked)]:
    Xf, Xt, _, _ = numpy_views(ds)
    s = xgb.predict_proba(np.hstack([Xf, Xt]))[:, 1]
    print(f"xgboost_fin_text [{tag}]: PR-AUC "
          f"{average_precision_score(y_te, s):.4f} "
          f"ROC-AUC {roc_auc_score(y_te, s):.4f}")
del Xf_tr, Xt_tr
gc.collect()

# --- saved cross-modal checkpoints (best-validation seed; single model, not
#     the 5-seed ensemble - stated as such) ---
for n_years in (3, 1):
    ckpt = torch.load(root / f"experiments/checkpoints/crossmodal_y{n_years}.pt",
                      map_location="cpu", weights_only=False)
    model = CrossModalModel(n_features=ckpt["n_features"],
                            d_model=ckpt["d_model"], n_years=n_years)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    te = test_ds if n_years == 3 else BankruptcyDataset("test", n_years=1)
    te_masked = masked_copy(te)
    for tag, ds in [("original", te), ("text-masked", te_masked)]:
        s = predict(model, ds)
        print(f"crossmodal_y{n_years} best-val seed [{tag}]: PR-AUC "
              f"{average_precision_score(y_te, s):.4f} "
              f"ROC-AUC {roc_auc_score(y_te, s):.4f}")
