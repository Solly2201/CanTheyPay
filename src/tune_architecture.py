"""Architecture-refinement study with a pre-registered protocol.

PROTOCOL (fixed before any run):
- All decisions use VALIDATION PR-AUC only. Test data is never loaded in
  `sweep` mode; per-seed checkpoints are saved so the frozen winner can be
  evaluated on test exactly once afterwards (`finalize` mode).
- Each candidate is trained with 5 seeds (42..46), same hyperparameters and
  early stopping as the main suite. Reported per candidate: per-seed val
  PR-AUC mean/std and the 5-seed probability-ensemble val PR-AUC.
- DECISION RULE: a variant replaces V0 (the in-harness control identical to
  the frozen crossmodal_y1 design) only if BOTH
    (a) ensemble val PR-AUC >= V0's + 0.02, and
    (b) per-seed mean val PR-AUC > V0's.
  If the best text-side variant (V1-V4) and best attention-side variant
  (V5-V9) both pass, their combination is trained and adopted only if it
  passes the same rule against each of them. If nothing passes, V0 is kept
  (parsimony) and the frozen model stands.

Usage:
  python -m src.tune_architecture sweep       # trains all candidates (val only)
  python -m src.tune_architecture decide      # applies the rule, prints verdict
  python -m src.tune_architecture finalize NAME   # ONE test evaluation of NAME
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (BATCH_SIZE, EXPERIMENTS_DIR, LR, MAX_EPOCHS, PATIENCE,
                        SEED, set_seed)
from src.models.variants import CANDIDATES, VariantDataset, VariantModel

TUNE_DIR = EXPERIMENTS_DIR / "tuning"
TUNE_DIR.mkdir(exist_ok=True)
SEEDS = [SEED + i for i in range(5)]
MARGIN = 0.02


def predict(model, ds, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for fin, text, mask, iid, _ in DataLoader(ds, batch_size=batch_size):
            out.append(torch.sigmoid(model(fin, text, mask, iid)).numpy())
    return np.concatenate(out)


def train_one(cfg, train_ds, val_ds, seed):
    set_seed(seed)
    model = VariantModel(train_ds.n_features, cfg)
    y = train_ds.labels
    pw = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1))
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, best_state, wait = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for fin, text, mask, iid, label in loader:
            opt.zero_grad()
            loss = crit(model(fin, text, mask, iid), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pr = average_precision_score(val_ds.labels, predict(model, val_ds))
        if pr > best + 1e-4:
            best, wait = pr, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def sweep(configs=None):
    results = {}
    path = TUNE_DIR / "val_results.json"
    if path.exists():
        results = json.loads(path.read_text())
    for cfg in (configs or CANDIDATES):
        if cfg.name in results:
            print(f"{cfg.name}: cached, skipping")
            continue
        print(f"=== {cfg.name} ({cfg.notes}) ===")
        train_ds = VariantDataset("train", cfg)
        val_ds = VariantDataset("validation", cfg)
        seed_scores, val_preds = [], []
        for seed in SEEDS:
            model = train_one(cfg, train_ds, val_ds, seed)
            s = predict(model, val_ds)
            val_preds.append(s)
            seed_scores.append(float(average_precision_score(val_ds.labels, s)))
            torch.save(model.state_dict(),
                       TUNE_DIR / f"{cfg.name}_seed{seed}.pt")
            print(f"  seed {seed}: val PR-AUC {seed_scores[-1]:.4f}")
        ens = float(average_precision_score(val_ds.labels,
                                            np.mean(val_preds, axis=0)))
        results[cfg.name] = {
            "notes": cfg.notes, "per_seed": seed_scores,
            "seed_mean": float(np.mean(seed_scores)),
            "seed_std": float(np.std(seed_scores)), "ensemble_val_prauc": ens,
        }
        np.savez(TUNE_DIR / f"{cfg.name}_valpreds.npz",
                 preds=np.stack(val_preds), labels=val_ds.labels)
        path.write_text(json.dumps(results, indent=2))
        print(f"{cfg.name}: ensemble val PR-AUC {ens:.4f} "
              f"(seeds {np.mean(seed_scores):.4f}+/-{np.std(seed_scores):.4f})")
    print("SWEEP COMPLETE")


def decide():
    res = json.loads((TUNE_DIR / "val_results.json").read_text())
    base = res["V0_base"]
    print(f"{'variant':24s} {'ens val PR-AUC':>14s} {'seed mean+/-std':>18s} verdict")
    passing = {}
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["ensemble_val_prauc"]):
        ok = (r["ensemble_val_prauc"] >= base["ensemble_val_prauc"] + MARGIN
              and r["seed_mean"] > base["seed_mean"] and name != "V0_base")
        if ok:
            passing[name] = r
        print(f"{name:24s} {r['ensemble_val_prauc']:14.4f} "
              f"{r['seed_mean']:9.4f}+/-{r['seed_std']:.4f} "
              f"{'PASS' if ok else ('base' if name == 'V0_base' else 'fail')}")
    print(f"\nRule: ensemble >= {base['ensemble_val_prauc']:.4f}+{MARGIN} AND "
          f"seed mean > {base['seed_mean']:.4f}")
    if not passing:
        print("VERDICT: no variant passes -> keep V0 (frozen model stands)")
    else:
        print("VERDICT: passing variants:", ", ".join(passing))
    return passing


def finalize(name):
    """ONE test evaluation of the frozen config. Run only after `decide`."""
    from src.models.variants import VariantConfig
    cfg = next(c for c in CANDIDATES if c.name == name)
    test_ds = VariantDataset("test", cfg)
    val = np.load(TUNE_DIR / f"{name}_valpreds.npz")
    preds = []
    for seed in SEEDS:
        model = VariantModel(test_ds.n_features, cfg)
        model.load_state_dict(torch.load(TUNE_DIR / f"{name}_seed{seed}.pt",
                                         map_location="cpu"))
        preds.append(predict(model, test_ds))
    ens = np.mean(preds, axis=0)
    from src.evaluate import evaluate_scores
    out = evaluate_scores(val["labels"], val["preds"].mean(axis=0),
                          test_ds.labels, ens)
    out["per_seed_test_pr_auc"] = [
        float(average_precision_score(test_ds.labels, p)) for p in preds]
    (TUNE_DIR / f"final_{name}_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        sweep()
    elif mode == "decide":
        decide()
    elif mode == "finalize":
        finalize(sys.argv[2])
