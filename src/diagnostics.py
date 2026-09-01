"""Cheap, deterministic diagnostics shown next to a demo prediction.

These are *diagnostics*, not explanations. Two kinds are produced, both from
information that already exists at inference time - nothing is retrained and no
attribution method (SHAP/LIME/etc.) is involved:

  1. Financial signals: descriptive readings of the already-engineered ratio
     features (``src.features.RATIO_NAMES``) for the three supplied years -
     which ratios moved the most, and which standard distress level-screens
     the year-t values trip. These describe the *input*, not the model.

  2. Text signals: the chunks that received the most cross-attention from the
     financial year-tokens (``CrossModalModel.last_attn``). Attention shows
     where the model routed weight; it is not evidence that a chunk caused the
     prediction.

The same caveat the report makes applies here: attention maps and ratio
readings are model/input diagnostics, never causal explanations.
"""
import numpy as np

from src.features import RATIO_NAMES

# Curated subset of the engineered ratios that has an unambiguous solvency
# direction, so a move can be labelled "improving"/"deteriorating" honestly.
# (name, higher_is_safer, display label)
SIGNAL_RATIOS = [
    ("current_ratio", True, "Current ratio (CA / CL)"),
    ("working_capital_ta", True, "Working capital / assets"),
    ("retained_earnings_ta", True, "Retained earnings / assets"),
    ("ebit_ta", True, "EBIT / assets"),
    ("roa", True, "Return on assets"),
    ("net_margin", True, "Net margin"),
    ("ebitda_margin", True, "EBITDA margin"),
    ("mv_tl", True, "Market value / total liabilities"),
    ("leverage", False, "Leverage (TL / TA)"),
    ("debt_ebitda", False, "Long-term debt / EBITDA"),
]

# Textbook level screens applied to the most recent year (t). These are
# standard solvency/profitability checks on the supplied numbers - they are not
# derived from the model and carry no claim about the predicted probability.
LEVEL_SCREENS = [
    ("current_ratio", "lt", 1.0,
     "Current ratio below 1.0 - current liabilities exceed current assets"),
    ("leverage", "gt", 1.0,
     "Total liabilities exceed total assets - negative book equity"),
    ("retained_earnings_ta", "lt", 0.0, "Negative retained earnings"),
    ("ebit_ta", "lt", 0.0, "Operating loss (EBIT negative)"),
    ("roa", "lt", 0.0, "Negative return on assets"),
    ("net_margin", "lt", 0.0, "Negative net margin"),
    ("working_capital_ta", "lt", 0.0, "Negative working capital"),
]

_IDX = {name: i for i, name in enumerate(RATIO_NAMES)}
YEAR_LABELS = ["t-2", "t-1", "t"]


def _as_matrix(ratios) -> np.ndarray:
    """ratios: [T, F] engineered (un-standardized) ratio features."""
    m = np.asarray(ratios, dtype=float)
    if m.ndim != 2 or m.shape[1] != len(RATIO_NAMES):
        raise ValueError(f"expected [T, {len(RATIO_NAMES)}] ratios, got {m.shape}")
    return m


def ratio_signals(ratios, top_n: int = 4) -> list[dict]:
    """The curated ratios that moved most between the first and last supplied
    year, ranked by scale-normalised movement.

    Movement is normalised by the largest absolute value the ratio takes over
    the supplied years, so ratios on different scales stay comparable. The
    ranking is purely descriptive: it says which inputs changed most, not which
    ones drove the model's output.
    """
    m = _as_matrix(ratios)
    out = []
    for name, higher_is_safer, label in SIGNAL_RATIOS:
        vals = m[:, _IDX[name]]
        delta = float(vals[-1] - vals[0])
        scale = float(max(np.abs(vals).max(), 1e-6))
        rel = delta / scale
        # positive => moved in the direction associated with more distress
        adverse = -rel if higher_is_safer else rel
        out.append({
            "name": name,
            "label": label,
            "values": [float(v) for v in vals],
            "delta": delta,
            "magnitude": abs(rel),
            "adverse": adverse > 0,
            "direction": ("deteriorating" if adverse > 0 else
                          "improving" if adverse < 0 else "flat"),
        })
    # deterministic ordering: strongest move first, ties broken by ratio order
    out.sort(key=lambda d: (-d["magnitude"], _IDX[d["name"]]))
    return out[:top_n]


def level_flags(ratios) -> list[str]:
    """Standard distress level-screens tripped by the most recent year."""
    m = _as_matrix(ratios)
    last = m[-1]
    flags = []
    for name, op, bound, message in LEVEL_SCREENS:
        v = float(last[_IDX[name]])
        if (op == "lt" and v < bound) or (op == "gt" and v > bound):
            flags.append(f"{message} ({name} = {v:.2f})")
    return flags


def ratio_table(ratios) -> dict:
    """The curated ratios across the supplied years, ready for a small table.

    Returns {"index": [labels], "columns": ["t-2", "t-1", "t"], "data": [[...]]}
    so the caller can build a DataFrame without this module importing pandas.
    """
    m = _as_matrix(ratios)
    years = YEAR_LABELS[-m.shape[0]:]
    return {
        "index": [label for _, _, label in SIGNAL_RATIOS],
        "columns": years,
        "data": [[float(v) for v in m[:, _IDX[name]]]
                 for name, _, _ in SIGNAL_RATIOS],
    }


def text_signals(chunks: list[str], attention, top_n: int = 3,
                 words: int = 45) -> list[dict]:
    """Highest cross-attention text chunks, with a short readable snippet.

    ``attention`` is the per-chunk weight already returned by inference
    (``result["chunk_attention"]``), averaged over the financial year-queries.
    """
    if not chunks or attention is None or len(attention) == 0:
        return []
    w = np.asarray(attention, dtype=float)
    k = min(len(w), len(chunks))
    if k == 0:
        return []
    w = w[:k]
    order = np.argsort(-w, kind="stable")[:top_n]
    out = []
    for r in order:
        snippet = " ".join(str(chunks[int(r)]).split()[:words])
        out.append({"rank": len(out) + 1, "index": int(r),
                    "weight": float(w[r]), "snippet": snippet})
    return out


NOT_CAUSAL_NOTE = (
    "These are model/input diagnostics, **not** causal explanations. They "
    "describe how the supplied numbers moved and where the model's "
    "cross-attention went - not why a company would fail, and not an "
    "attribution of the predicted probability to any single item."
)

NO_TEXT_NOTE = (
    "Text diagnostics unavailable - no usable report text was supplied, so "
    "this prediction is financial-only (the model substitutes its learned "
    "no-text token)."
)
