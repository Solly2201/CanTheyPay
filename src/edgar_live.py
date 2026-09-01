"""Live SEC EDGAR fetch for the DEMO ONLY (never used for training).

Given a ticker, pulls the company's XBRL company facts and maps them onto the
same 18 raw accounting items ($ millions) the model was trained on, for the
three most recent fiscal years with data; optionally fetches the latest 10-K
and extracts Item 1 / Item 7 text, preprocessed the same way as the training
corpus (lowercase, numbers and stopwords stripped, 256-word chunks).

Honesty notes surfaced to the user in the demo:
- "market_value" is approximated by the SEC's EntityPublicFloat (aggregate
  market value of common equity held by non-affiliates) - the closest
  EDGAR-native equivalent of market capitalization.
- If a required item cannot be found under any known XBRL tag, we return the
  list of missing items and the demo refuses to fabricate a prediction.

SEC fair-access rules: max 10 req/s and a descriptive User-Agent.
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_ITEMS

UA = {"User-Agent": "CanTheyPay college DL project jimil@imaginarium.io"}

# XBRL tag candidates per raw item, in priority order.
TAG_MAP = {
    "current_assets": ["AssetsCurrent"],
    "total_assets": ["Assets"],
    "cost_of_goods_sold": ["CostOfGoodsAndServicesSold", "CostOfRevenue",
                           "CostOfGoodsSold", "CostOfServices"],
    "total_long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt",
                             "LongTermDebtAndCapitalLeaseObligations"],
    "depreciation_and_amortization": ["DepreciationDepletionAndAmortization",
                                      "DepreciationAndAmortization",
                                      "DepreciationAmortizationAndAccretionNet"],
    "ebit": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "inventory": ["InventoryNet"],
    "total_current_liabilities": ["LiabilitiesCurrent"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "total_receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
                          "AccountsNotesAndLoansReceivableNetCurrent"],
    "total_revenue": ["Revenues",
                      "RevenueFromContractWithCustomerExcludingAssessedTax",
                      "SalesRevenueNet",
                      "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "total_liabilities": ["Liabilities"],
    "total_operating_expenses": ["OperatingExpenses", "CostsAndExpenses"],
}
OPTIONAL_ITEMS = {"inventory", "total_receivables", "total_long_term_debt",
                  "depreciation_and_amortization", "gross_profit",
                  "total_operating_expenses", "cost_of_goods_sold"}

_STOPWORDS = None


def _stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _STOPWORDS = set(ENGLISH_STOP_WORDS)
    return _STOPWORDS


def _get(url):
    time.sleep(0.15)  # stay well under 10 req/s
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r


def ticker_to_cik(ticker: str) -> tuple[int, str] | None:
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"]), row["title"]
    return None


def _annual_values(units: dict) -> dict[int, float]:
    """{fiscal year end -> value} from 10-K FY records; latest filing wins."""
    vals = {}
    for rec in units.get("USD", []):
        if rec.get("form") not in ("10-K", "10-K/A") or rec.get("fp") != "FY":
            continue
        end = rec.get("end", "")
        if len(end) < 4:
            continue
        start = rec.get("start")
        if start:  # flow item: require a ~12 month duration
            months = (int(end[:4]) - int(start[:4])) * 12 + int(end[5:7]) - int(start[5:7])
            if not 10 <= months <= 14:
                continue
        year = int(end[:4])
        key = (year, rec.get("filed", ""))
        if year not in vals or key >= vals[year][0]:
            vals[year] = (key, float(rec["val"]))
    return {y: v for y, (_, v) in vals.items()}


def fetch_financials(cik: int) -> dict:
    """Returns {'years': [y-2, y-1, y], 'items': [dict per year],
    'missing': [...], 'approximations': [...]}."""
    facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})

    per_item = {}
    for item, tags in TAG_MAP.items():
        for tag in tags:
            if tag in gaap:
                vals = _annual_values(gaap[tag]["units"])
                if vals:
                    per_item[item] = vals
                    break

    approximations = []
    # market value ~ public float (dei tag, reported on the 10-K cover page)
    if "EntityPublicFloat" in dei:
        mv = {}
        for rec in dei["EntityPublicFloat"]["units"].get("USD", []):
            if rec.get("form", "").startswith("10-K"):
                mv[int(rec["end"][:4])] = float(rec["val"])
        if mv:
            per_item["market_value"] = mv
            approximations.append("market_value = SEC EntityPublicFloat "
                                  "(public float, not full market cap)")

    years = sorted(set(per_item.get("total_assets", {}))
                   & set(per_item.get("net_income", {})))
    if len(years) < 3:
        return {"error": f"only {len(years)} usable fiscal years in EDGAR XBRL",
                "years": years}
    years = years[-3:]

    rows, missing = [], set()
    for y in years:
        row = {}
        for item in RAW_ITEMS:
            src = per_item.get(item, {})
            v = src.get(y)
            if v is None and item == "market_value" and src:
                # public float is dated mid-fiscal-year; take the nearest year
                nearest = min(src, key=lambda yy: abs(yy - y))
                if abs(nearest - y) <= 1:
                    v = src[nearest]
            if v is None:
                if item == "net_sales":
                    v = per_item.get("total_revenue", {}).get(y)
                elif item == "ebitda":
                    e = per_item.get("ebit", {}).get(y)
                    d = per_item.get("depreciation_and_amortization", {}).get(y, 0.0)
                    v = None if e is None else e + (d or 0.0)
                elif item == "gross_profit":
                    r = per_item.get("total_revenue", {}).get(y)
                    c = per_item.get("cost_of_goods_sold", {}).get(y)
                    v = None if r is None or c is None else r - c
            if v is None:
                if item in OPTIONAL_ITEMS:
                    v = 0.0
                    approximations.append(f"{item} missing for FY{y}, set to 0")
                else:
                    missing.add(f"{item} (FY{y})")
                    v = 0.0
            row[item] = v / 1e6  # dollars -> $ millions (training scale)
        rows.append(row)
    return {"years": years, "items": rows, "missing": sorted(missing),
            "approximations": sorted(set(approximations)),
            "company": facts.get("entityName", "")}


def preprocess_text(raw: str, chunk_words: int = 256, max_chunks: int = 8):
    """Replicate the training corpus preprocessing (lowercase, strip numbers
    and punctuation, drop stopwords) and cut into 256-word chunks."""
    sw = _stopwords()
    text = re.sub(r"[^a-z\s]", " ", raw.lower())
    words = [w for w in text.split() if len(w) > 1 and w not in sw]
    chunks = [" ".join(words[i:i + chunk_words])
              for i in range(0, len(words), chunk_words)]
    return [c for c in chunks if len(c.split()) >= 20][:max_chunks]


def _markup_parser(markup: str) -> str:
    """Pick the parser matching an EDGAR primary document's actual format.

    Filings from ~2019 on are Inline XBRL: genuine XHTML carrying an
    `<?xml ...?>` declaration (though EDGAR still serves them as text/html).
    Older filings are plain HTML. Handing XHTML to an HTML parser is what
    raises bs4's XMLParsedAsHTMLWarning, so branch on the declaration instead
    of assuming one format for every filing.
    """
    return "lxml-xml" if markup.lstrip()[:5].lower() == "<?xml" else "lxml"


def fetch_10k_text(cik: int) -> dict:
    """Best-effort: latest 10-K primary document -> Item 1 and Item 7 text."""
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    rec = sub["filings"]["recent"]
    idx = next((i for i, f in enumerate(rec["form"]) if f == "10-K"), None)
    if idx is None:
        return {"error": "no 10-K found"}
    acc = rec["accessionNumber"][idx].replace("-", "")
    doc = rec["primaryDocument"][idx]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
    html = _get(url).text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, _markup_parser(html))
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)

    def section(start_pat, end_pat):
        # last occurrence of the heading skips the table of contents
        starts = [m.start() for m in re.finditer(start_pat, text, re.I)]
        if not starts:
            return ""
        s = starts[-1] if len(starts) > 1 else starts[0]
        m_end = re.search(end_pat, text[s + 100:], re.I)
        return text[s:s + 100 + m_end.start()] if m_end else text[s:s + 60000]

    item1 = section(r"item\s*1\s*[\.\:\-]?\s*business",
                    r"item\s*1a\s*[\.\:\-]?\s*risk")
    item7 = section(r"item\s*7\s*[\.\:\-]?\s*management",
                    r"item\s*7a\s*[\.\:\-]?\s*quant")
    out = {"filing_url": url, "filing_date": rec["filingDate"][idx]}
    out["item_1_chunks"] = preprocess_text(item1) if len(item1) > 500 else []
    out["item_7_chunks"] = preprocess_text(item7) if len(item7) > 500 else []
    if not out["item_1_chunks"] and not out["item_7_chunks"]:
        out["warning"] = ("could not reliably extract Item 1/7 from this filing's "
                          "HTML; prediction will use financial data only")
    return out
