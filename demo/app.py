"""CanTheyPay - bankruptcy risk demo (Streamlit).

Three modes:
  1. Real US public company by ticker  -> live SEC EDGAR financials + 10-K text
  2. Dataset company                   -> browse the (anonymized) test split
  3. Manual input                      -> enter the 18 items + paste report text

Run: streamlit run demo/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DIR, RAW_ITEMS

st.set_page_config(page_title="CanTheyPay - Bankruptcy Risk", page_icon="📉")
st.title("CanTheyPay — Corporate Bankruptcy Risk")
st.caption(
    "Multi-modal cross-attention model (financial history + 10-K text via "
    "FinBERT), trained on 3,142 US public companies (2001–2014). Predictions "
    "are a college research project, **not** financial advice.")

mode = st.sidebar.radio("Input mode", [
    "Real company (SEC EDGAR)", "Dataset company", "Manual input"])


def format_probability(prob: float) -> str:
    """Percentage with four decimals, e.g. 0.4432% (the demo routinely sits in
    the sub-1% range, where one decimal collapses everything to 0.0%)."""
    return f"{prob * 100:.4f}%"


def show_diagnostics(ratios, chunk_attention=None, text_chunks=None):
    """Compact, deterministic diagnostics for the prediction just shown.

    Uses only what inference already produced: the engineered ratio features
    for the supplied years and the model's cross-attention weights. No
    attribution method, no extra dependency, no retraining.
    """
    from src.diagnostics import (NO_TEXT_NOTE, NOT_CAUSAL_NOTE, level_flags,
                                 ratio_signals, ratio_table, text_signals)

    with st.expander("Model diagnostics (not causal explanations)", expanded=True):
        st.caption(NOT_CAUSAL_NOTE)

        st.markdown("**Largest ratio moves, t-2 → t**")
        for s in ratio_signals(ratios):
            arrow = "🔻" if s["adverse"] else "🔺"
            path = " → ".join(f"{v:,.2f}" for v in s["values"])
            st.write(f"{arrow} **{s['label']}**: {path}  ·  {s['direction']}")

        flags = level_flags(ratios)
        st.markdown("**Level screens at year t**")
        if flags:
            for f in flags:
                st.write(f"- {f}")
        else:
            st.write("- None of the standard distress screens are tripped.")

        tab = ratio_table(ratios)
        st.caption("Curated ratios across the supplied years:")
        st.dataframe(pd.DataFrame(tab["data"], index=tab["index"],
                                  columns=tab["columns"]).style.format("{:,.3f}"))

        st.markdown("**Text the model attended to most**")
        signals = text_signals(text_chunks or [], chunk_attention)
        if signals:
            for t in signals:
                st.write(f"- `attn={t['weight']:.3f}` {t['snippet']}…")
            st.caption(
                "Cross-attention weight from the financial year-tokens to each "
                "text chunk. High weight means the model routed capacity there; "
                "it is **not** evidence that the sentence caused the prediction.")
        else:
            st.info(NO_TEXT_NOTE)


def show_prediction(result, context_notes=None, text_chunks=None):
    prob, cat = result["probability"], result["category"]
    color = {"Low risk": "green", "Elevated risk": "orange",
             "High risk": "red"}[cat]
    st.metric("1-year bankruptcy/distress probability", format_probability(prob))
    st.markdown(f"**Risk category:** :{color}[{cat}] "
                f"(decision threshold from validation: {result['threshold']:.2f})")
    if result["used_text_chunks"]:
        st.write(f"Text used: {result['used_text_chunks']} chunks "
                 f"(cross-attention weights below are model diagnostics, "
                 f"not explanations).")
    else:
        st.info("No usable report text — prediction is financial-only "
                "(the model handles this with a learned no-text token).")
    for note in (context_notes or []):
        st.warning(note)
    show_diagnostics(result["ratios"], result.get("chunk_attention"),
                     text_chunks)


@st.cache_data(show_spinner="Fetching from SEC EDGAR ...")
def edgar_fetch(ticker):
    from src.edgar_live import fetch_10k_text, fetch_financials, ticker_to_cik
    hit = ticker_to_cik(ticker)
    if hit is None:
        return {"error": f"Ticker '{ticker}' not found in SEC EDGAR. Private "
                         "companies (e.g. OpenAI, Flipkart), subsidiaries "
                         "(YouTube, LinkedIn) and non-US listings (Zomato/"
                         "Eternal, Swiggy) have no SEC filings - no valid "
                         "prediction is possible for them."}
    cik, name = hit
    fin = fetch_financials(cik)
    if "error" in fin:
        return {"error": f"{name}: {fin['error']} - not enough public XBRL "
                         "data for a valid prediction."}
    text = fetch_10k_text(cik)
    return {"cik": cik, "name": name, "fin": fin, "text": text}


@st.cache_data(show_spinner=False)
def dataset_chunk_texts(cik, split="test"):
    """Chunk texts for one dataset company, in the same order the embeddings
    were stacked in (src.dataset.load_split sorts by cik, item, chunk_idx)."""
    df = pd.read_parquet(PROCESSED_DIR / f"text_{split}.parquet")
    df = df[df["cik"].astype(str) == str(cik)].sort_values(["item", "chunk_idx"])
    return [f"[{r.item}] {r.text}" for r in df.itertuples()]


if mode == "Real company (SEC EDGAR)":
    st.subheader("Live prediction for a US public filer")
    ticker = st.text_input("Ticker (e.g. NVDA, AMZN, MSFT, META, GOOGL)", "NVDA")
    if st.button("Fetch & predict"):
        data = edgar_fetch(ticker.strip())
        if "error" in data:
            st.error(data["error"])
        else:
            fin = data["fin"]
            st.write(f"**{data['name']}** (CIK {data['cik']}), fiscal years "
                     f"{fin['years']}")
            df = pd.DataFrame(fin["items"], index=[f"FY{y}" for y in fin["years"]])
            st.dataframe(df.T.style.format("{:,.0f}"))
            notes = [f"Approximation: {a}" for a in fin["approximations"]]
            if fin["missing"]:
                st.error("Missing required items: " + ", ".join(fin["missing"]) +
                         ". Prediction skipped - insufficient public data.")
            else:
                chunks = (data["text"].get("item_1_chunks", []) +
                          data["text"].get("item_7_chunks", []))
                if "warning" in data["text"]:
                    notes.append(data["text"]["warning"])
                notes.append(
                    "Out-of-distribution note: mega-caps like this are far "
                    "larger than the median training company; treat the "
                    "probability as illustrative.")
                from src.inference import predict_company
                with st.spinner("Running FinBERT + cross-modal model ..."):
                    result = predict_company(fin["items"], chunks)
                show_prediction(result, notes, text_chunks=chunks)

elif mode == "Dataset company":
    st.subheader("Browse the held-out test split (anonymized companies)")
    fin = np.load(PROCESSED_DIR / "financial_test.npz")
    ciks = fin["cik"]
    labels = fin["label"]
    pick = st.selectbox(
        "Company", [f"{c}  ({'bankrupt' if l else 'healthy'}, fy{y})"
                    for c, l, y in zip(ciks, labels, fin["fyear"])])
    i = int(np.where(ciks == pick.split()[0])[0][0])
    raw = fin["raw_features"][i]
    st.caption("Standardized model inputs are hidden; showing engineered "
               "ratio features per year:")
    from src.features import RATIO_NAMES
    st.dataframe(pd.DataFrame(raw, columns=RATIO_NAMES,
                              index=["t-2", "t-1", "t"]).T)
    if st.button("Predict"):
        from src.dataset import BankruptcyDataset
        from src.inference import load_model, tuned_threshold
        import torch
        ds = BankruptcyDataset("test")
        f, t, m, y = ds[i]
        model = load_model()
        with torch.no_grad():
            prob = float(torch.sigmoid(
                model(f.unsqueeze(0), t.unsqueeze(0), m.unsqueeze(0))).item())
        thr = tuned_threshold()
        k = int(m.sum())
        attn = model.last_attn
        chunk_attention = (attn[0].mean(dim=0)[1:1 + k].tolist()
                           if attn is not None and k else [])
        st.metric("Predicted probability", format_probability(prob))
        st.write(f"Actual outcome: **{'bankrupt' if y else 'healthy'}** | "
                 f"threshold {thr:.2f} | text chunks: {k}")
        show_diagnostics(raw, chunk_attention,
                         dataset_chunk_texts(str(ciks[i])) if k else [])

else:
    st.subheader("Manual input (18 accounting items, $ millions)")
    st.caption("Enter three fiscal years, oldest first. For non-US-GAAP "
               "companies (Indian filers etc.) results are out-of-distribution "
               "and not valid - shown for demonstration only.")

    MANUAL_STORE = "manual_items"
    MANUAL_STEP = 1.0          # $1M per +/- click; 0.01 was invisible at %.1f

    def manual_key(item, j):
        """Stable, unique widget key per (accounting item, year): 18 x 3 = 54."""
        return f"manual__{item}__y{j}"

    ALL_MANUAL_KEYS = [manual_key(it, j) for j in range(3) for it in RAW_ITEMS]
    if MANUAL_STORE not in st.session_state:
        st.session_state[MANUAL_STORE] = {k: 0.0 for k in ALL_MANUAL_KEYS}
    store = st.session_state[MANUAL_STORE]

    def sync_manual(k):
        """Mirror a widget's value into a plain (non-widget) session-state dict.

        Streamlit drops widget state for widgets that were not rendered in a
        run, so the 54 inputs would otherwise reset to 0 whenever the sidebar
        mode is switched away and back. The mirror is not a widget key, so it
        survives, and it is what seeds `value=` on the next render.
        """
        store[k] = float(st.session_state[k])

    years = []
    cols = st.columns(3)
    for j, col in enumerate(cols):
        with col:
            st.markdown(f"**Year t-{2-j}**")
            entries = {}
            for item in RAW_ITEMS:
                k = manual_key(item, j)
                entries[item] = st.number_input(
                    item.replace("_", " "), key=k,
                    value=float(store.get(k, 0.0)),
                    step=MANUAL_STEP, format="%.1f",
                    on_change=sync_manual, args=(k,))
                store[k] = float(entries[item])
            years.append(entries)

    text = st.text_area("Optional: paste annual-report text (MD&A / business "
                        "description)", height=150)
    if st.button("Predict", key="manual"):
        if years[-1]["total_assets"] <= 0:
            st.error("Total assets (year t) must be positive.")
        else:
            from src.edgar_live import preprocess_text
            from src.inference import predict_company
            chunks = preprocess_text(text) if text.strip() else []
            with st.spinner("Running model ..."):
                result = predict_company(years, chunks)
            show_prediction(result, [
                "Manual input is unvalidated; garbage in, garbage out."],
                text_chunks=chunks)
