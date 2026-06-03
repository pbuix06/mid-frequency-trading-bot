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


def extract_pit_series(
    facts: dict,
    tag: str,
    namespace: str = "us-gaap",
    unit: str | None = None,
) -> pd.Series:
    """
    PIT series for one XBRL tag, indexed by the `filed` (knowable) date.

    For each reporting period (`end` date) we keep the EARLIEST filing — the value
    as first reported, which is what you would have known then (later restatements
    arrive with later `filed` dates and are intentionally ignored here). The
    result is a step series: value(t) = most recently reported figure knowable by
    date t. Forward-fill it onto a price index to align to trading days.

    Returns an empty Series if the tag/unit is absent.
    """
    node = facts.get("facts", {}).get(namespace, {}).get(tag)
    if not node:
        return pd.Series(dtype=float, name=tag)
    units = node.get("units", {})
    if not units:
        return pd.Series(dtype=float, name=tag)
    unit = unit or next(iter(units))
    rows = units.get(unit, [])
    if not rows:
        return pd.Series(dtype=float, name=tag)

    df = pd.DataFrame(rows)
    if "filed" not in df or "end" not in df or "val" not in df:
        return pd.Series(dtype=float, name=tag)
    df["filed"] = pd.to_datetime(df["filed"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)

    # Earliest filing per period-end (original report, not restatements)
    df = df.sort_values("filed").groupby("end", as_index=False).first()

    # Step series keyed by knowable date; on a tie keep the latest period-end
    df = df.sort_values(["filed", "end"])
    s = pd.Series(df["val"].to_numpy(dtype=float), index=df["filed"].to_numpy(), name=tag)
    s.index = pd.DatetimeIndex(s.index, name="filed")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# Tags we use for value/quality factors (us-gaap unless noted).
FUNDAMENTAL_TAGS = {
    "book_equity": ("StockholdersEquity", "us-gaap", "USD"),
    "assets": ("Assets", "us-gaap", "USD"),
    "net_income": ("NetIncomeLoss", "us-gaap", "USD"),
    "shares": ("EntityCommonStockSharesOutstanding", "dei", "shares"),
}


def extract_fundamentals(facts: dict) -> pd.DataFrame:
    """
    Build a tidy long DataFrame of the FUNDAMENTAL_TAGS for one company:
    columns [filed, item, value]. Each row is knowable as of `filed`.
    """
    frames = []
    for item, (tag, ns, unit) in FUNDAMENTAL_TAGS.items():
        s = extract_pit_series(facts, tag, namespace=ns, unit=unit)
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
