"""
Crypto research panel — 1m -> 5m resample + aligned 24/7 panel for mft/research/.

KEY DIFFERENCE from equities: crypto trades 24/7. There is NO regular-session mask, NO
overnight flat, NO time-of-day grouping. The resampler keeps every bar. The market proxy
(the "SPY of crypto") is BTCUSDT — alts are measured relative to BTC for lead-lag/residual.

NOTE (flagged for the alpha phase, not built yet): the equity xs_backtest resets its
non-overlapping rebalance grid per ET calendar day. For 24/7 crypto that grid will want a
continuous (or UTC-midnight) reset instead — a small change to make when crypto alphas run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mft.research.panel import Panel  # reuse the same container the harness consumes

ROOT = Path(__file__).parents[2]
CRYPTO_DIR = ROOT / "data" / "crypto"
SPOT_1M = CRYPTO_DIR / "spot_1m"
PERP_1M = CRYPTO_DIR / "perp_1m"
FUNDING_DIR = CRYPTO_DIR / "funding"
OI_DIR = CRYPTO_DIR / "open_interest"

_SUM_COLS = ["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]


def resample_crypto_bars(df_1m: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """1m -> freq OHLCV(+quote/trades/taker) with NO session masking (24/7). vwap derived."""
    if df_1m.empty:
        return df_1m
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for col in _SUM_COLS:
        if col in df_1m.columns:
            agg[col] = "sum"
    out = df_1m.resample(freq, label="left", closed="left").agg(agg)
    out = out[out["volume"].notna() & (out["volume"] >= 0)]
    out = out[out["close"].notna()]
    if "quote_volume" in out.columns:
        out["vwap"] = out["quote_volume"] / out["volume"].replace(0, np.nan)  # avg traded price
    return out


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def load_spot(symbol: str, freq: str = "5min", spot_dir: Path = SPOT_1M) -> pd.DataFrame:
    return resample_crypto_bars(_load_parquet(spot_dir / f"{symbol}.parquet"), freq)


def load_crypto_panel(symbols: list[str], start: str | None = None, end: str | None = None,
                      freq: str = "5min", market: str = "BTCUSDT",
                      spot_dir: Path = SPOT_1M) -> Panel:
    """Aligned 24/7 spot panel. Market proxy defaults to BTCUSDT (the crypto 'market')."""
    syms = list(dict.fromkeys([*symbols, market]))
    bars: dict[str, pd.DataFrame] = {}
    for s in syms:
        p = spot_dir / f"{s}.parquet"
        if not p.exists():
            continue
        df = resample_crypto_bars(_load_parquet(p), freq)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
        if not df.empty:
            bars[s] = df
    present = [s for s in symbols if s in bars]
    return Panel(bars=bars, freq=freq, market=market, symbols=present)


def load_funding(symbol: str, funding_dir: Path = FUNDING_DIR) -> pd.Series:
    p = funding_dir / f"{symbol}.parquet"
    if not p.exists():
        return pd.Series(dtype=float, name="funding_rate")
    return _load_parquet(p)["funding_rate"]


def load_open_interest(symbol: str, oi_dir: Path = OI_DIR) -> pd.DataFrame:
    p = oi_dir / f"{symbol}.parquet"
    return _load_parquet(p) if p.exists() else pd.DataFrame()


def compute_basis(spot_close: pd.Series, perp_close: pd.Series) -> pd.Series:
    """Perp premium/discount = perp/spot - 1 (aligned on the common index)."""
    s, p = spot_close.align(perp_close, join="inner")
    return (p / s - 1.0).rename("basis")


# ── funding-rate helpers (8h funding -> 5m bar grid, causal) ──────────────────

def map_to_bars(series: pd.Series, bar_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill an 8-hourly series onto the bar grid, KNOWN-ONLY: each bar gets the most
    recent value whose timestamp is <= that bar (reindex method='ffill'). Bars before the
    first observation are NaN. This is the no-look-ahead funding/OI alignment.
    """
    s = series[~series.index.duplicated(keep="last")].sort_index()
    return s.reindex(bar_index, method="ffill")


def funding_zscore_series(funding: pd.Series, window: int = 10) -> pd.Series:
    """
    Per-symbol funding z-score on the FUNDING (8h) index — trailing, causal. De-means the
    coin's structural funding level so cross-sectional ranking compares 'unusually crowded'.
    """
    mu = funding.rolling(window).mean()
    sd = funding.rolling(window).std()
    return ((funding - mu) / sd.replace(0, np.nan)).rename("funding_z")


def cumulative_funding_series(funding: pd.Series) -> pd.Series:
    """Cumulative funding (for carry: funding over a hold = cumfund[exit] - cumfund[entry])."""
    return funding.sort_index().cumsum().rename("cum_funding")


def oi_change(oi: pd.Series, bars: int) -> pd.Series:
    """Trailing open-interest change over `bars` (past-only)."""
    return oi.pct_change(bars)
