"""
SEC EDGAR fundamentals ingestion — free, official, point-in-time.

Why EDGAR: it unlocks value/quality factor families (the classic
uncorrelated-with-momentum diversifiers) that price-only data cannot, and unlike
Yahoo fundamentals it is POINT-IN-TIME — every reported number carries the SEC
`filed` date, so we stamp data when it was KNOWABLE, not when it describes.
Delisted companies' filings persist, so it is survivorship-clean too.

Pipeline:
  ticker -> CIK (company_tickers.json)
  CIK -> companyfacts JSON (data.sec.gov/api/xbrl/companyfacts/CIK##########.json)
  tag -> PIT series: for each period-end take the EARLIEST filing (original
         report, not a later restatement), indexed by the `filed` (knowable) date.

SEC fair-access: a descriptive User-Agent with a real contact is REQUIRED, and
requests are kept under ~10/sec. Set EDGAR_UA in .env to "Your Name you@email".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# SEC requires a descriptive UA with contact info. Override via EDGAR_UA in .env.
_DEFAULT_UA = "MFT-research (set EDGAR_UA in .env) research@example.com"


def _headers() -> dict:
    return {"User-Agent": os.getenv("EDGAR_UA", _DEFAULT_UA),
            "Accept-Encoding": "gzip, deflate"}


def _get(url: str, retries: int = 3) -> requests.Response | None:
    """Polite GET with backoff. Returns None on 404 (no filings for that CIK)."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_headers(), timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            time.sleep(0.12)  # stay well under SEC's ~10 req/s limit
            return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


# ── Ticker -> CIK map ─────────────────────────────────────────────────────────

def load_cik_map(cache_path: Path | None = None) -> dict[str, int]:
    """
    Return {TICKER: cik_int}. Cached to disk on first download.
    """
    if cache_path and cache_path.exists():
        return {k: int(v) for k, v in json.loads(cache_path.read_text()).items()}

    resp = _get(CIK_MAP_URL)
    if resp is None:
        raise RuntimeError("Could not download SEC ticker->CIK map")
    raw = resp.json()
    mapping = {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(mapping))
    return mapping


# ── Fundamentals ──────────────────────────────────────────────────────────────

def fetch_companyfacts(cik: int) -> dict | None:
    """Raw companyfacts JSON for a CIK. None if the CIK has no XBRL facts."""
    resp = _get(COMPANYFACTS_URL.format(cik=cik))
    return resp.json() if resp is not None else None


def _collect_rows(facts: dict, tag: str, namespace: str, unit: str | None, flow: bool) -> pd.DataFrame:
    """Raw [end, filed, val] rows for one tag (annual-only if flow). Empty if absent."""
    node = facts.get("facts", {}).get(namespace, {}).get(tag)
    if not node:
        return pd.DataFrame()
    units = node.get("units", {})
    if not units:
        return pd.DataFrame()
    u = unit or next(iter(units))
    rows = units.get(u, [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "filed" not in df or "end" not in df or "val" not in df:
        return pd.DataFrame()
    df["filed"] = pd.to_datetime(df["filed"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    if flow:
        if "start" not in df:
            return pd.DataFrame()
        df["start"] = pd.to_datetime(df["start"], utc=True)
        days = (df["end"] - df["start"]).dt.days
        df = df[(days >= 300) & (days <= 400)]  # annual periods only
    return df[["end", "filed", "val"]] if not df.empty else pd.DataFrame()


def extract_pit_series(
    facts: dict,
    tag: str | list[str],
    namespace: str = "us-gaap",
    unit: str | None = None,
    flow: bool = False,
) -> pd.Series:
    """
    PIT series indexed by the `filed` (knowable) date.

    `tag` may be a single XBRL tag or a LIST of candidate tags. Candidates are
    COMBINED at the row level — companies switch tags across eras (e.g. revenue:
    old `Revenues`/`SalesRevenueNet` -> post-2018
    `RevenueFromContractWithCustomerExcludingAssessedTax`), so combining recovers
    the full history. For each reporting period (`end`) we keep the EARLIEST
    filing (original report, not later restatements). The result is a step series:
    value(t) = most recently reported figure knowable by date t. Forward-fill onto
    a price index to align to trading days.

    flow=True for income-statement items (revenue, COGS, net income): keep only
    ANNUAL periods (~300-400 days) so a 3-month and a 12-month figure sharing an
    `end` are never conflated.
    """
    tags = [tag] if isinstance(tag, str) else list(tag)
    parts = [r for r in (_collect_rows(facts, t, namespace, unit, flow) for t in tags) if not r.empty]
    if not parts:
        return pd.Series(dtype=float, name=tags[0])
    df = pd.concat(parts, ignore_index=True)

    # Earliest filing per period-end across all candidate tags
    df = df.sort_values("filed").groupby("end", as_index=False).first()

    df = df.sort_values(["filed", "end"])
    s = pd.Series(df["val"].to_numpy(dtype=float), index=df["filed"].to_numpy(), name=tags[0])
    s.index = pd.DatetimeIndex(s.index, name="filed")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# Items -> (candidate tags tried in order, namespace, unit, is_flow).
# Companies use different XBRL tags for the same concept, so we fall back.
FUNDAMENTAL_ITEMS = {
    "book_equity": (["StockholdersEquity"], "us-gaap", "USD", False),
    "assets":      (["Assets"], "us-gaap", "USD", False),
    "net_income":  (["NetIncomeLoss"], "us-gaap", "USD", True),
    "shares":      (["EntityCommonStockSharesOutstanding"], "dei", "shares", False),
    "revenue":     (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                     "SalesRevenueNet"], "us-gaap", "USD", True),
    "cogs":        (["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
                    "us-gaap", "USD", True),
}


def extract_fundamentals(facts: dict) -> pd.DataFrame:
    """
    Build a tidy long DataFrame of the FUNDAMENTAL_ITEMS for one company:
    columns [filed, item, value]. Each row is knowable as of `filed`. For each
    item the candidate tags are tried in order until one yields data.
    """
    frames = []
    for item, (tags, ns, unit, flow) in FUNDAMENTAL_ITEMS.items():
        s = extract_pit_series(facts, tags, namespace=ns, unit=unit, flow=flow)
        if not s.empty:
            frames.append(pd.DataFrame({"filed": s.index, "item": item, "value": s.to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=["filed", "item", "value"])
    return pd.concat(frames, ignore_index=True).sort_values(["item", "filed"])


def save_fundamentals(df: pd.DataFrame, ticker: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{ticker}.parquet", engine="pyarrow", compression="snappy")


def load_fundamentals(ticker: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No EDGAR fundamentals for {ticker} at {path}")
    return pd.read_parquet(path, engine="pyarrow")


def pit_value(df: pd.DataFrame, item: str, as_of: pd.Timestamp) -> float:
    """
    Most recent value of `item` knowable strictly on/before `as_of`.
    df is the long frame from extract_fundamentals(). NaN if nothing known yet.
    """
    sub = df[(df["item"] == item) & (df["filed"] <= as_of)]
    if sub.empty:
        return float("nan")
    return float(sub.sort_values("filed")["value"].iloc[-1])
