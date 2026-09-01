"""Classical baselines evaluated on the same splits as the neural models.

1. Logistic Regression      (financial features)
2. XGBoost                  (financial features)
3. XGBoost + text           (financial + mean FinBERT embedding, strongest
                             classical multimodal opponent)
4. Logistic Regression text (mean FinBERT embedding only)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def make_logreg(seed=42):
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              C=0.1, random_state=seed)


def make_xgb(pos_weight, seed=42):
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=seed, n_jobs=-1)


def run_baselines(train, val, test, seed=42):
    """train/val/test: (fin_flat, text_mean, y, has_text) tuples."""
    Xf_tr, Xt_tr, y_tr, _ = train
    Xf_va, Xt_va, y_va, _ = val
    Xf_te, Xt_te, y_te, _ = test
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    concat = lambda f, t: np.hstack([f, t])
    scores = {}

    lr = make_logreg(seed).fit(Xf_tr, y_tr)
    scores["logreg_financial"] = (lr.predict_proba(Xf_va)[:, 1],
                                  lr.predict_proba(Xf_te)[:, 1], lr)

    xgb = make_xgb(pos_weight, seed).fit(Xf_tr, y_tr)
    scores["xgboost_financial"] = (xgb.predict_proba(Xf_va)[:, 1],
                                   xgb.predict_proba(Xf_te)[:, 1], xgb)

    xgb_mm = make_xgb(pos_weight, seed).fit(concat(Xf_tr, Xt_tr), y_tr)
    scores["xgboost_fin_text"] = (xgb_mm.predict_proba(concat(Xf_va, Xt_va))[:, 1],
                                  xgb_mm.predict_proba(concat(Xf_te, Xt_te))[:, 1],
                                  xgb_mm)

    lr_t = make_logreg(seed).fit(Xt_tr, y_tr)
    scores["logreg_text"] = (lr_t.predict_proba(Xt_va)[:, 1],
                             lr_t.predict_proba(Xt_te)[:, 1], lr_t)
    return scores
