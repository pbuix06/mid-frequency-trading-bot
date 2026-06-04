"""
Alpaca intraday (minute-bar) ingestion — the gate to the minute-frequency pivot.

Free Alpaca "Basic" market data gives historical minute bars for US equities via
a plain REST API (no Windows terminal, works on macOS). This is enough to
PROTOTYPE an intraday reversal sleeve before committing to paid data.

Honest limitations of the free tier (carry these into every conclusion):
  - IEX feed only (~2-3% of consolidated volume) -> minute bars can be thin and
    noisy for less-liquid names. Use LIQUID large-caps only.
  - history ~2016+ ; not survivorship-clean (broker data). Fine for a scoped
    proof-of-concept on names liquid throughout, NOT for a broad clean study.
  - For a real minute system you upgrade to the SIP feed / Polygon later — same
    code path, just feed='sip' or a different adapter.

Setup (one-time):
  1. Create a free account at alpaca.markets.
  2. Generate API keys (paper keys are fine for DATA).
  3. Add to .env (never commit):
        ALPACA_API_KEY_ID=...
        ALPACA_API_SECRET_KEY=...

Bars carry o/h/l/c/v plus n (trade count) and vw (VWAP) — useful microstructure
features for intraday alphas. All timestamps UTC.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
_DEFAULT_FEED = "iex"          # free tier
_PAGE_LIMIT = 10_000


def _headers() -> dict:
    kid = os.getenv("ALPACA_API_KEY_ID", "")
    sec = os.getenv("ALPACA_API_SECRET_KEY", "")
    if not kid or not sec:
        raise RuntimeError(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not in .env — "
            "create a free alpaca.markets account and add your keys."
        )
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


def _parse_bars(bars: list[dict]) -> pd.DataFrame:
    """
    Parse Alpaca bar JSON -> OHLCV DataFrame with UTC DatetimeIndex.
    Columns: open, high, low, close, volume, trades, vwap.
    Network-free (testable): takes the already-decoded `bars` list.
    """
    if not bars:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "trades", "vwap"]
        )
    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["t"], utc=True)
    df = df.set_index("timestamp").sort_index()
    trades = df["n"] if "n" in df.columns else pd.Series(0.0, index=df.index)
    vwap = df["vw"] if "vw" in df.columns else df["c"]
    out = pd.DataFrame({
        "open":   df["o"].astype(float),
        "high":   df["h"].astype(float),
        "low":    df["l"].astype(float),
        "close":  df["c"].astype(float),
        "volume": df["v"].astype(float),
        "trades": trades.astype(float),
        "vwap":   vwap.astype(float),
    }, index=df.index)
    return out[out["close"] > 0]


def fetch_minute_bars(
    symbol: str,
    start: str,
    end: str,
    feed: str = _DEFAULT_FEED,
    adjustment: str = "all",
    retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch 1-minute bars for `symbol` over [start, end] (ISO dates), paginated.

    Args:
        start, end: ISO date/datetime strings (UTC assumed).
        feed:       'iex' (free) or 'sip' (paid).
        adjustment: 'all' (split + dividend), 'raw', 'split', 'dividend'.
    """
    url = DATA_URL.format(symbol=symbol)
    params = {
        "timeframe": "1Min", "start": start, "end": end,
        "limit": _PAGE_LIMIT, "feed": feed, "adjustment": adjustment,
    }
    all_bars: list[dict] = []
    page_token = None
    while True:
        if page_token:
            params["page_token"] = page_token
        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=_headers(), params=params, timeout=30)
                resp.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        payload = resp.json()
        all_bars.extend(payload.get("bars") or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
        time.sleep(0.05)
    return _parse_bars(all_bars)


def save_intraday(df: pd.DataFrame, symbol: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{symbol}.parquet", engine="pyarrow", compression="snappy")


def load_intraday(symbol: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No intraday data for {symbol} at {path}")
    return pd.read_parquet(path, engine="pyarrow")
