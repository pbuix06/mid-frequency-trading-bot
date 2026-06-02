"""
EODHD data ingestion — fetch, clean, and store PIT OHLCV Parquet.

Storage layout: data/pit/{TICKER}.parquet
One file per ticker. Includes delisted tickers (survivorship-bias-free).

Prices are backward-adjusted for splits and dividends using EODHD's
adjusted_close. Ratio applied to all OHLC so bars are internally consistent.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://eodhistoricaldata.com/api"
DEFAULT_FROM    = "2010-01-01"   # broad universe ingest (full ~41k ticker download)
RESEARCH_FROM   = "2000-01-01"   # Phase 3+ research instruments — full regime coverage
MIN_BARS        = 30

# Lock-box: the final 15% of the dataset.
# Hardcoded 2022-07-01. NEVER computed dynamically — once set, immutable.
# Research runs may include bars through this cutoff; anything after it is the final exam.
LOCKBOX_CUTOFF = pd.Timestamp("2022-07-01", tz="UTC")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MFT-research/1.0)",
    "Accept": "application/json",
}


def _get(url: str, params: dict, retries: int = 3) -> requests.Response | None:
    """GET with retry/backoff. Returns None on 404. Raises on other failures."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_us_tickers(api_key: str, delisted: bool = False) -> list[str]:
    """Return sorted list of US common stock ticker codes from EODHD."""
    url = f"{BASE_URL}/exchange-symbol-list/US"
    params: dict = {"api_token": api_key, "fmt": "json", "type": "common_stock"}
    if delisted:
        params["delisted"] = 1
    resp = _get(url, params)
    if resp is None:
        return []
    return sorted({row["Code"] for row in resp.json() if row.get("Code")})


def fetch_ohlcv(
    ticker: str,
    api_key: str,
    from_date: str = DEFAULT_FROM,
    to_date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch split/dividend-adjusted daily OHLCV for one US ticker.

    Returns empty DataFrame if the ticker has no data.
    OHLC are backward-adjusted using the ratio of adjusted_close / raw_close.
    """
    url = f"{BASE_URL}/eod/{ticker}.US"
    params: dict = {"api_token": api_key, "from": from_date, "fmt": "json", "period": "d"}
    if to_date:
        params["to"] = to_date

    resp = _get(url, params)
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Backward-adjust OHLC using adjusted_close ratio
    if "adjusted_close" in df.columns:
        ratio = df["adjusted_close"] / df["close"].replace(0, float("nan"))
        for col in ("open", "high", "low"):
            df[col] = (df[col] * ratio).round(6)
        df["close"] = df["adjusted_close"].round(6)
        df = df.drop(columns=["adjusted_close"])

    df["volume"] = df["volume"].fillna(0).astype(float)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df["close"] > 0]
    return df


def fetch_ohlcv_symbol(
    eodhd_symbol: str,
    api_key: str,
    from_date: str = DEFAULT_FROM,
    to_date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch adjusted daily OHLCV using the full EODHD symbol (no .US appended).

    Use for:
      - US ETFs:      fetch_ohlcv_symbol("TLT.US",  key)
      - FX pairs:     fetch_ohlcv_symbol("EURUSD.FOREX", key)
      - Crypto:       fetch_ohlcv_symbol("BTC-USD.CC",   key)

    Applies the same adjusted_close ratio as fetch_ohlcv().
    """
    url = f"{BASE_URL}/eod/{eodhd_symbol}"
    params: dict = {"api_token": api_key, "from": from_date, "fmt": "json", "period": "d"}
    if to_date:
        params["to"] = to_date

    resp = _get(url, params)
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("timestamp").sort_index()

    if "adjusted_close" in df.columns:
        ratio = df["adjusted_close"] / df["close"].replace(0, float("nan"))
        for col in ("open", "high", "low"):
            df[col] = (df[col] * ratio).round(6)
        df["close"] = df["adjusted_close"].round(6)
        df = df.drop(columns=["adjusted_close"])

    df["volume"] = df["volume"].fillna(0).astype(float)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df["close"] > 0]
    return df


def save_ticker(df: pd.DataFrame, ticker: str, pit_dir: Path) -> None:
    """Write OHLCV DataFrame to data/pit/{ticker}.parquet (snappy compressed)."""
    pit_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pit_dir / f"{ticker}.parquet", engine="pyarrow", compression="snappy")


def load_ticker(ticker: str, pit_dir: Path) -> pd.DataFrame:
    """Load a single ticker from data/pit/. Raises FileNotFoundError if missing."""
    path = pit_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No PIT data for {ticker} at {path}")
    return pd.read_parquet(path, engine="pyarrow")


def lockbox_cutoff(*args, **kwargs) -> pd.Timestamp:
    """Return the hardcoded lock-box cutoff. Argument accepted for backward compat."""
    return LOCKBOX_CUTOFF
