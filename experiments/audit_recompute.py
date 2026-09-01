"""Audit C: recompute deterministic baselines from scratch and diff every
metric against experiments/results.json. Run: python experiments/audit_recompute.py"""
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import SEED, set_seed
from src.dataset import BankruptcyDataset, numpy_views
from src.evaluate import evaluate_scores

set_seed(SEED)

views, labels = {}, {}
for s in ["train", "validation", "test"]:
    ds = BankruptcyDataset(s)
    views[s] = numpy_views(ds)
    labels[s] = ds.labels
    ds = None
    gc.collect()

from src.models.baselines import run_baselines

res = json.loads(Path("experiments/results.json").read_text())
base = run_baselines(views["train"], views["validation"], views["test"], seed=SEED)
mismatches = []
for name, (s_va, s_te, _m) in base.items():
    fresh = evaluate_scores(labels["validation"], s_va, labels["test"], s_te)
    for split in ["val", "test"]:
        for key, v in fresh[split].items():
            if v != res[name][split][key]:
                mismatches.append((name, split, key, v, res[name][split][key]))
print("baseline recompute mismatches:", mismatches if mismatches else "NONE - exact match")

from sklearn.neural_network import MLPClassifier

Xtr, _, ytr, _ = views["train"]
Xva = views["validation"][0]
Xte = views["test"][0]
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500,
                    random_state=SEED, early_stopping=True).fit(Xtr, ytr)
fresh = evaluate_scores(labels["validation"], mlp.predict_proba(Xva)[:, 1],
                        labels["test"], mlp.predict_proba(Xte)[:, 1])
diffs = {key: (fresh["test"][key], res["mlp_financial"]["test"][key])
         for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]
         if fresh["test"][key] != res["mlp_financial"]["test"][key]}
print("mlp recompute diffs:", diffs if diffs else "NONE - exact match")
print("LR/XGB/MLP input = flattened 3-year features:", Xtr.shape)
