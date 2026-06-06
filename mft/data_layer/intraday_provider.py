"""
Vendor-agnostic intraday data interface — the seam between data vendors and the
strategy, so the alpha never depends on Alpaca / Polygon / Databento directly.

Every provider returns the SAME normalized minute-bar schema (UTC index):

    required:  open, high, low, close, volume
    optional:  vwap, trades, bid, ask, spread   (NaN when the vendor lacks them)
    stamped:   df.attrs["symbol"], df.attrs["source"]

Switching vendors (free IEX -> paid SIP/Polygon) is then a one-line provider swap,
not a strategy rewrite — the "one code path" rule extended to data. Add a concrete
vendor by subclassing IntradayProvider and returning `normalize_bars(...)`; do NOT
scatter vendor-specific parsing through the research/live code.

Note: a Polygon/Databento ingester is intentionally NOT written here — per the
continuation instructions it is added only once a paid vendor is actually chosen
(see docs/INTRADAY_ORB_EXPERIMENT_SPEC.md §4). The ABC below is what it will
implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from mft.data_layer.alpaca_ingest import (
    INTRADAY_LOCKBOX,
    fetch_minute_bars,
    load_intraday,
)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["vwap", "trades", "bid", "ask", "spread"]
NORMALIZED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def normalize_bars(df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    """
    Coerce a vendor frame into the normalized schema. Network-free, deterministic.

    Enforces: UTC tz-aware DatetimeIndex, sorted, de-duplicated, required columns
    present, optional columns created as NaN when absent, spread derived from
    bid/ask when not given, close > 0. Stamps symbol/source into df.attrs.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}:{symbol} missing required columns {missing}")

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    out = df.copy()
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()

    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    # Derive spread from quotes when the vendor gave bid/ask but no spread.
    if out["spread"].isna().all() and not out["bid"].isna().all() and not out["ask"].isna().all():
        out["spread"] = out["ask"].astype(float) - out["bid"].astype(float)

    out = out[NORMALIZED_COLUMNS].astype(
        {c: float for c in REQUIRED_COLUMNS}
    )
    out = out[out["close"] > 0]
    out.attrs["symbol"] = symbol
    out.attrs["source"] = source
    return out


def enforce_intraday_lockbox(df: pd.DataFrame, lockbox: pd.Timestamp = INTRADAY_LOCKBOX) -> pd.DataFrame:
    """Drop bars on/after the intraday lock-box. Research must call this."""
    return df[df.index < lockbox]


class IntradayProvider(ABC):
    """A source of normalized minute bars. Subclass per vendor."""

    name: str = "abstract"

    @abstractmethod
    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return normalized bars for [start, end] (ISO, UTC). See NORMALIZED_COLUMNS."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.name!r})"


class ParquetIntradayProvider(IntradayProvider):
    """
    Offline provider over locally-saved Parquet (e.g. the ingested IEX data in
    data/intraday/). This is what backtests use — fully decoupled from any vendor
    API, so research is reproducible and network-free.
    """

    name = "parquet"

    def __init__(self, data_dir: Path, source_tag: str = "parquet"):
        self.data_dir = Path(data_dir)
        self.name = source_tag

    def get_bars(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        raw = load_intraday(symbol, self.data_dir)
        df = normalize_bars(raw, symbol, self.name)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df


class AlpacaIntradayProvider(IntradayProvider):
    """
    Live Alpaca provider. feed='iex' (free) or 'sip' (paid) — same code path, the
    only difference is the feed string, exactly as the data-layer docstring
    promised. Requires ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in .env.
    """

    name = "alpaca"

    def __init__(self, feed: str = "iex", adjustment: str = "all"):
        self.feed = feed
        self.adjustment = adjustment
        self.name = f"alpaca-{feed}"

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        raw = fetch_minute_bars(symbol, start, end, feed=self.feed, adjustment=self.adjustment)
        return normalize_bars(raw, symbol, self.name)
