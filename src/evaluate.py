"""Shared evaluation: ROC-AUC, PR-AUC, precision/recall/F1 at a validation-tuned
threshold, and the confusion matrix. Accuracy is reported but never used for
model selection (the dataset is heavily imbalanced)."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def tune_threshold(y_true, y_score):
    """Pick the threshold that maximizes F1 on the (validation) set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(y_score, 4)):
        pred = (y_score >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def compute_metrics(y_true, y_score, threshold=0.5):
    pred = (y_score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float((pred == y_true).mean()), 4),
        "threshold": round(float(threshold), 4),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "n": int(len(y_true)), "positives": int(y_true.sum()),
    }


def evaluate_scores(y_val, s_val, y_test, s_test):
    """Tune threshold on validation, report both splits."""
    t = tune_threshold(y_val, s_val)
    return {"val": compute_metrics(y_val, s_val, t),
            "test": compute_metrics(y_test, s_test, t)}
