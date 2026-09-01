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


def show_prediction(result, context_notes=None):
    prob, cat = result["probability"], result["category"]
    color = {"Low risk": "green", "Elevated risk": "orange",
             "High risk": "red"}[cat]
    st.metric("1-year bankruptcy/distress probability", f"{prob:.1%}")
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
                show_prediction(result, notes)

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
        st.metric("Predicted probability", f"{prob:.1%}")
        st.write(f"Actual outcome: **{'bankrupt' if y else 'healthy'}** | "
                 f"threshold {thr:.2f} | text chunks: {int(m.sum())}")

else:
    st.subheader("Manual input (18 accounting items, $ millions)")
    st.caption("Enter three fiscal years, oldest first. For non-US-GAAP "
               "companies (Indian filers etc.) results are out-of-distribution "
               "and not valid - shown for demonstration only.")
    years = []
    cols = st.columns(3)
    for j, col in enumerate(cols):
        with col:
            st.markdown(f"**Year t-{2-j}**")
            years.append({item: st.number_input(
                item.replace("_", " "), value=0.0, key=f"{item}_{j}",
                format="%.1f") for item in RAW_ITEMS})
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
                "Manual input is unvalidated; garbage in, garbage out."])
