"""Model diagnostics and interpretability outputs.

Important distinction (stated in the report as well): attention maps and
permutation importances are MODEL DIAGNOSTICS - they describe what the model
uses, not a causal explanation of bankruptcy. We never claim attention ==
explanation.

Outputs:
  experiments/explain_financial_importance.csv  permutation importance (cross-modal)
  experiments/explain_xgb_importance.csv        XGBoost gain importance
  experiments/explain_attention_examples.md     top-attended text chunks for the
                                                highest-risk test companies
Usage: python -m src.explain
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EXPERIMENTS_DIR, MODELS_DIR, PROCESSED_DIR, SEED, set_seed
from src.dataset import BankruptcyDataset, numpy_views
from src.features import RATIO_NAMES
from src.models.baselines import make_xgb
from src.models.multimodal import CrossModalModel
from src.train import predict


def load_crossmodal(n_features, n_years=3):
    ckpt = torch.load(MODELS_DIR / f"crossmodal_y{n_years}.pt",
                      map_location="cpu", weights_only=False)
    model = CrossModalModel(n_features=n_features, d_model=ckpt["d_model"],
                            n_years=n_years)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def permutation_importance(model, ds, n_repeats=5, seed=SEED):
    """PR-AUC drop when each per-year financial feature is shuffled across
    companies (all 3 year copies shuffled together)."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    base = average_precision_score(ds.labels, predict(model, ds))
    rows = []
    orig = ds.features.copy()
    for j, name in enumerate(RATIO_NAMES):
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(ds))
            ds.features[:, :, j] = orig[perm][:, :, j]
            drops.append(base - average_precision_score(ds.labels, predict(model, ds)))
            ds.features[:] = orig
        rows.append({"feature": name, "pr_auc_drop": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False)


def xgb_importance(train_ds):
    Xf, _, y, _ = numpy_views(train_ds)
    cols = [f"{p}_{n}" for p in ("t-2", "t-1", "t") for n in RATIO_NAMES]
    xgb = make_xgb((y == 0).sum() / max((y == 1).sum(), 1)).fit(Xf, y)
    imp = pd.DataFrame({"feature": cols, "gain": xgb.feature_importances_})
    return imp.sort_values("gain", ascending=False)


def attention_examples(model, ds, split="test", top_companies=10, top_chunks=3):
    """For the highest-scored companies with text: which chunks did the
    financial year-tokens attend to most?"""
    text_df = pd.read_parquet(PROCESSED_DIR / f"text_{split}.parquet")
    text_df = text_df.sort_values(["cik", "item", "chunk_idx"])
    scores = predict(model, ds)
    has_text = ds.has_text
    idx = np.argsort(-scores)
    idx = [i for i in idx if has_text[i]][:top_companies]

    lines = ["# Attention diagnostics: top-attended text chunks",
             "",
             "For the highest-risk test companies, the text chunks receiving the "
             "most cross-attention from the financial year-tokens. These are "
             "diagnostics of what the model reads, **not** causal explanations.",
             ""]
    for i in idx:
        cik = str(ds.cik[i])
        fin, text, mask, label = ds[i]
        with torch.no_grad():
            model(fin.unsqueeze(0), text.unsqueeze(0), mask.unsqueeze(0))
        attn = model.last_attn[0]          # [T, K+1]; column 0 is no-text token
        chunk_attn = attn.mean(dim=0)[1:]  # avg over year queries -> [K]
        chunks = text_df[text_df["cik"] == cik].reset_index(drop=True)
        k = int(mask.sum().item())
        order = np.argsort(-chunk_attn[:k].numpy())[:top_chunks]
        lines.append(f"## {cik} (label={int(label)}, score={scores[i]:.3f})")
        for r in order:
            if r < len(chunks):
                snippet = " ".join(str(chunks.iloc[r]["text"]).split()[:60])
                lines.append(f"- attn={chunk_attn[r]:.3f} [{chunks.iloc[r]['item']}] "
                             f"{snippet}...")
        lines.append("")
    return "\n".join(lines)


def main():
    set_seed(SEED)
    train_ds = BankruptcyDataset("train")
    test_ds = BankruptcyDataset("test")
    model = load_crossmodal(train_ds.n_features)

    print("Permutation importance (cross-modal, test split) ...")
    pi = permutation_importance(model, test_ds)
    pi.to_csv(EXPERIMENTS_DIR / "explain_financial_importance.csv", index=False)
    print(pi.head(10).to_string(index=False))

    print("XGBoost gain importance ...")
    xi = xgb_importance(train_ds)
    xi.to_csv(EXPERIMENTS_DIR / "explain_xgb_importance.csv", index=False)
    print(xi.head(10).to_string(index=False))

    print("Attention examples ...")
    md = attention_examples(model, test_ds)
    (EXPERIMENTS_DIR / "explain_attention_examples.md").write_text(
        md, encoding="utf-8")
    print("Saved experiments/explain_attention_examples.md")


if __name__ == "__main__":
    main()
