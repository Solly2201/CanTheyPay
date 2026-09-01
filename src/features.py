"""Financial feature engineering shared by the training pipeline and the demo.

Input: the 18 raw accounting items ($ millions) for one fiscal year, plus
optionally the previous year's items for growth features.
Output: a fixed-order per-year feature vector.
"""
import numpy as np

RATIO_NAMES = [
    "current_ratio",        # current assets / current liabilities
    "working_capital_ta",   # (CA - CL) / total assets
    "retained_earnings_ta",
    "ebit_ta",
    "mv_tl",                # market value / total liabilities (Altman X4)
    "roa",                  # net income / total assets
    "leverage",             # total liabilities / total assets
    "gross_margin",
    "net_margin",
    "ebitda_margin",
    "debt_ebitda",          # long-term debt / EBITDA (clipped)
    "receivables_rev",
    "inventory_rev",
    "opex_rev",
    "log_total_assets",
    "log_revenue",
    "log_market_value",
    "rev_growth",           # vs previous year (0 if unavailable)
    "assets_growth",
    "ni_delta_ta",          # (NI_t - NI_{t-1}) / total assets
]


def _sd(a, b):
    """Safe divide."""
    b = float(b)
    if abs(b) < 1e-6:
        return 0.0
    return float(a) / b


def year_features(items: dict, prev_items: dict | None = None) -> np.ndarray:
    """items: {raw_item_name: value in $M} for one fiscal year."""
    ca = items["current_assets"]; ta = items["total_assets"]
    cl = items["total_current_liabilities"]; tl = items["total_liabilities"]
    ni = items["net_income"]; re_ = items["retained_earnings"]
    ebit = items["ebit"]; ebitda = items["ebitda"]; gp = items["gross_profit"]
    rev = items["total_revenue"]; mv = items["market_value"]
    ltd = items["total_long_term_debt"]; rec = items["total_receivables"]
    inv = items["inventory"]; opex = items["total_operating_expenses"]

    f = [
        _sd(ca, cl),
        _sd(ca - cl, ta),
        _sd(re_, ta),
        _sd(ebit, ta),
        _sd(mv, tl),
        _sd(ni, ta),
        _sd(tl, ta),
        _sd(gp, rev),
        _sd(ni, rev),
        _sd(ebitda, rev),
        _sd(ltd, ebitda),
        _sd(rec, rev),
        _sd(inv, rev),
        _sd(opex, rev),
        np.log1p(max(ta, 0.0)),
        np.log1p(max(rev, 0.0)),
        np.log1p(max(mv, 0.0)),
    ]
    if prev_items is not None:
        f += [
            _sd(rev - prev_items["total_revenue"], abs(prev_items["total_revenue"])),
            _sd(ta - prev_items["total_assets"], abs(prev_items["total_assets"])),
            _sd(ni - prev_items["net_income"], ta),
        ]
    else:
        f += [0.0, 0.0, 0.0]
    return np.array(f, dtype=np.float32)


def sequence_features(years: list[dict]) -> np.ndarray:
    """years: list of raw-item dicts in chronological order (oldest first).
    Returns [T, F] array; growth features use the preceding year when present."""
    out = []
    for i, y in enumerate(years):
        prev = years[i - 1] if i > 0 else None
        out.append(year_features(y, prev))
    return np.stack(out)


def winsorize_fit(x: np.ndarray, lo=1.0, hi=99.0):
    """Fit clipping bounds per feature on TRAIN data only. x: [N, F]."""
    return np.percentile(x, lo, axis=0), np.percentile(x, hi, axis=0)


def winsorize_apply(x, bounds):
    lo, hi = bounds
    return np.clip(x, lo, hi)
