"""Training loop for the neural models: weighted BCE, early stopping on
validation PR-AUC, deterministic seeding."""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (BATCH_SIZE, DROPOUT, D_MODEL, LR, MAX_EPOCHS, MODELS_DIR,
                        N_HEADS, PATIENCE, SEED, set_seed)
from src.models.multimodal import MODEL_REGISTRY


def predict(model, ds, device="cpu", batch_size=256):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size)
    out = []
    with torch.no_grad():
        for fin, text, mask, _ in loader:
            logits = model(fin.to(device), text.to(device), mask.to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_model(name, train_ds, val_ds, n_years=3, seed=SEED, device="cpu",
                d_model=D_MODEL, verbose=True):
    set_seed(seed)
    cls = MODEL_REGISTRY[name]
    kwargs = dict(n_features=train_ds.n_features, d_model=d_model,
                  n_heads=N_HEADS, dropout=DROPOUT)
    if name != "text_only":
        kwargs["n_years"] = n_years
    model = cls(**kwargs).to(device)

    y = train_ds.labels
    pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1),
                              dtype=torch.float32, device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best_prauc, best_state, patience = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        total = 0.0
        for fin, text, mask, label in loader:
            fin, text, mask, label = (x.to(device) for x in (fin, text, mask, label))
            opt.zero_grad()
            loss = crit(model(fin, text, mask), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(label)
        val_scores = predict(model, val_ds, device)
        prauc = average_precision_score(val_ds.labels, val_scores)
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d} loss {total/len(train_ds):.4f} "
                  f"val PR-AUC {prauc:.4f}")
        if prauc > best_prauc + 1e-4:
            best_prauc, patience = prauc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "model": name, "n_years": n_years,
                "n_features": train_ds.n_features, "d_model": d_model},
               MODELS_DIR / f"{name}_y{n_years}.pt")
    if verbose:
        print(f"  best val PR-AUC {best_prauc:.4f}")
    return model
