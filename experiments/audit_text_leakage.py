"""Audit F: probe the text modality for label leakage.

Distinguish two kinds of bankruptcy-related language in year-t 10-K text:
  LEGITIMATE early-warning: going-concern doubt, covenant breach risk,
    "may be forced to seek protection" - written BEFORE any filing.
  LEAKAGE (post-petition): the 10-K was filed after the Chapter 7/11 petition
    and describes it as a done/ongoing fact ("we filed", "debtor in
    possession", "emerged from chapter").

We count companies whose chunks contain each phrase family, by label.
Run: python experiments/audit_text_leakage.py"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

# stopwords/numbers were stripped from the corpus, so phrases must be written
# in the same collapsed form (no "the"/"of"/"for"/"a"; no digits)
POST_PETITION = [
    "debtor possession", "debtorinpossession", "filed voluntary petition",
    "filed petition", "bankruptcy court approved", "plan reorganization confirmed",
    "emerged chapter", "emerged bankruptcy", "bankruptcy proceedings commenced",
    "chapter cases", "petition date",
]
EARLY_WARNING = [
    "going concern", "substantial doubt", "covenant", "default",
    "seek protection", "restructuring", "insolvency",
]


def scan(split):
    fin = np.load(root / f"data/processed/financial_{split}.npz")
    label = dict(zip(fin["cik"].astype(str), fin["label"]))
    text = pd.read_parquet(root / f"data/processed/text_{split}.parquet")
    joined = text.groupby("cik")["text"].apply(" ".join)

    rows = []
    for family, phrases in [("post_petition", POST_PETITION),
                            ("early_warning", EARLY_WARNING)]:
        pat = re.compile("|".join(re.escape(p) for p in phrases))
        hit = joined.apply(lambda s: bool(pat.search(s)))
        for lab in (0, 1):
            ciks = [c for c in joined.index if label.get(str(c)) == lab]
            n = len(ciks)
            h = int(hit.loc[ciks].sum())
            rows.append({"split": split, "family": family, "label": lab,
                         "companies_with_text": n, "hits": h,
                         "rate": round(h / max(n, 1), 3)})
    return rows


all_rows = []
for split in ["train", "validation", "test"]:
    all_rows += scan(split)
df = pd.DataFrame(all_rows)
print(df.to_string(index=False))

# list the actual post-petition offenders in test for manual review
fin = np.load(root / "data/processed/financial_test.npz")
label = dict(zip(fin["cik"].astype(str), fin["label"]))
text = pd.read_parquet(root / "data/processed/text_test.parquet")
joined = text.groupby("cik")["text"].apply(" ".join)
pat = re.compile("|".join(re.escape(p) for p in POST_PETITION))
print("\nTest-split label=1 companies with post-petition phrasing:")
for c, s in joined.items():
    if label.get(str(c)) == 1:
        m = pat.search(s)
        if m:
            i = m.start()
            print(f"  {c}: ...{s[max(0, i-80):i+120]}...")
