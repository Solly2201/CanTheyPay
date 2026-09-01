"""Re-verify the partial sweep artifacts that were double-written by two
concurrent identically-seeded processes: retrain V0 and V3 for two seeds each
IN MEMORY (no file writes) and compare val PR-AUC to val_results.json.
Deterministic seeding means an exact match validates the stored artifacts.
Run: python experiments/tuning/verify_artifacts.py"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from src.models.variants import CANDIDATES, VariantDataset
from src.tune_architecture import predict, train_one

rec = json.loads((root / "experiments/tuning/val_results.json").read_text())
ok = True
for name in ["V0_base", "V3_item_embedding"]:
    cfg = next(c for c in CANDIDATES if c.name == name)
    train_ds = VariantDataset("train", cfg)
    val_ds = VariantDataset("validation", cfg)
    for i, seed in enumerate([42, 43]):
        model = train_one(cfg, train_ds, val_ds, seed)
        pr = float(average_precision_score(val_ds.labels, predict(model, val_ds)))
        stored = rec[name]["per_seed"][i]
        match = abs(pr - stored) < 1e-6
        ok &= match
        print(f"{name} seed {seed}: fresh {pr:.6f} stored {stored:.6f} "
              f"{'MATCH' if match else 'MISMATCH'}")
print("ARTIFACTS VALID" if ok else "ARTIFACTS INVALID - full re-run required")
