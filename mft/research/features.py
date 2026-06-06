"""
Past-only intraday features (per symbol).

THE RULE (tested in tests/test_research_no_lookahead.py): every feature value at the
close of bar t uses ONLY bars with timestamp <= t. The current bar's own OHLCV is
known at its close, so rolling windows are current-bar-inclusive — but nothing from
t+1 onward may ever enter. A signal built from these is then tradeable at t+1 (the
entry lag is enforced in targets.py / xs_backtest.py, NOT here).

All functions take a single symbol's resampled frame (and, where market-relative, an
aligned market close series) and return a Series on the same index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FREQ_MIN = 5  # research bar size; minutes_to_bars below assumes this


def minutes_to_bars(minutes: int, freq_min: int = FREQ_MIN) -> int:
    if minutes % freq_min != 0:
        raise ValueError(f"{minutes}m is not a multiple of the {freq_min}m bar")
    return minutes // freq_min


# ── price / return features ───────────────────────────────────────────────────

def trailing_return(close: pd.Series, bars: int) -> pd.Series:
    """Return over the last `bars` bars, known at bar t's close."""
    return close.pct_change(bars).rename(f"ret_{bars}b")


def bar_return(close: pd.Series) -> pd.Series:
    return close.pct_change(1)


def rolling_vol(close: pd.Series, window: int) -> pd.Series:
    """Realized vol = std of 1-bar returns over a trailing window (incl. bar t)."""
    return bar_return(close).rolling(window).std().rename(f"vol_{window}b")


def return_zscore(close: pd.Series, lookback_bars: int, window: int, skip: int = 1) -> pd.Series:
    """
    Z-score of the recent `lookback_bars` return vs its own trailing distribution.

    `skip` ignores the most recent `skip` bars when measuring the *level* (the existing
    IntradayReversal trick) to avoid trading the 1-bar bid/ask bounce we cannot capture
    live. Everything uses data <= bar t.
    """
    r = close.pct_change(lookback_bars).shift(skip)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std()
    return ((r - mu) / sd.replace(0, np.nan)).rename("ret_z")


# ── VWAP ──────────────────────────────────────────────────────────────────────

def session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Running session VWAP from the open up to (and including) bar t — past-only.
    Uses the per-bar vwap*volume when available, else close*volume.
    """
    et_day = pd.Index(df.index.tz_convert("America/New_York").normalize().tz_localize(None))
    price = df["vwap"] if "vwap" in df.columns and df["vwap"].notna().any() else df["close"]
    pv = (price.astype(float) * df["volume"].astype(float)).groupby(et_day).cumsum()
    vv = df["volume"].astype(float).groupby(et_day).cumsum()
    out = pv / vv.replace(0, np.nan)
    out.index = df.index
    return out.rename("session_vwap")


def vwap_distance(df: pd.DataFrame) -> pd.Series:
    """close / session_vwap - 1. >0 = price above the day's average traded price."""
    return (df["close"] / session_vwap(df) - 1.0).rename("vwap_dist")


# ── volume ────────────────────────────────────────────────────────────────────

def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    """Z-score of bar volume vs its trailing distribution (incl. bar t)."""
    mu = volume.rolling(window).mean()
    sd = volume.rolling(window).std()
    return ((volume - mu) / sd.replace(0, np.nan)).rename("vol_z")


def relative_volume_tod(df: pd.DataFrame, lookback_days: int = 20) -> pd.Series:
    """
    Bar volume relative to the average volume at the SAME time-of-day over the
    PRIOR `lookback_days` days (today excluded). This removes the strong intraday
    U-shape so a volume spike means "unusual for this minute", not "it's the open".

    Strictly past-only: the trailing mean is shift(1) over prior days only.
    """
    et = df.index.tz_convert("America/New_York")
    day = pd.Index(et.normalize().tz_localize(None), name="day")
    tod = pd.Index(et.strftime("%H:%M"), name="tod")

    vol = df["volume"].astype(float)
    wide = vol.groupby([day, tod]).mean().unstack("tod")              # [day x tod]
    min_days = max(2, lookback_days // 2)
    trail = wide.shift(1).rolling(lookback_days, min_periods=min_days).mean()  # prior days only
    rel_wide = wide / trail.replace(0, np.nan)

    key = pd.MultiIndex.from_arrays([day, tod])
    rel = rel_wide.stack(future_stack=True).reindex(key)
    rel.index = df.index
    return rel.rename("rel_vol_tod")


def volume_zscore_tod(df: pd.DataFrame, lookback_days: int = 20) -> pd.Series:
    """
    Same-time-of-day volume Z-SCORE: (today's bar volume - mean) / std, where mean/std
    are over the SAME minute-of-day on the PRIOR `lookback_days` days (today excluded).

    Like relative_volume_tod but standardized — "how many sigmas of unusual is this
    minute's volume", controlling for the intraday U-shape. Strictly past-only.
    """
    et = df.index.tz_convert("America/New_York")
    day = pd.Index(et.normalize().tz_localize(None), name="day")
    tod = pd.Index(et.strftime("%H:%M"), name="tod")

    vol = df["volume"].astype(float)
    wide = vol.groupby([day, tod]).mean().unstack("tod")
    min_days = max(2, lookback_days // 2)
    roll = wide.shift(1).rolling(lookback_days, min_periods=min_days)  # prior days only
    z_wide = (wide - roll.mean()) / roll.std().replace(0, np.nan)

    key = pd.MultiIndex.from_arrays([day, tod])
    z = z_wide.stack(future_stack=True).reindex(key)
    z.index = df.index
    return z.rename("vol_z_tod")


# ── market-relative ───────────────────────────────────────────────────────────

def market_adjusted_return(close: pd.Series, market_close: pd.Series, bars: int) -> pd.Series:
    """Stock trailing return minus the market's trailing return (same horizon)."""
    mkt = market_close.reindex(close.index).ffill()
    return (close.pct_change(bars) - mkt.pct_change(bars)).rename(f"madj_ret_{bars}b")


def rolling_beta(close: pd.Series, market_close: pd.Series, window: int, skip: int = 1) -> pd.Series:
    """
    Rolling beta of the stock's 1-bar returns to the market's, estimated over a
    trailing window ENDING at bar t-skip (so the contemporaneous bar t is not used to
    explain itself). cov/var over the window.
    """
    rs = bar_return(close)
    rm = bar_return(market_close.reindex(close.index).ffill())
    cov = rs.rolling(window).cov(rm)
    var = rm.rolling(window).var()
    beta = (cov / var.replace(0, np.nan)).shift(skip)
    return beta.rename("beta")


def residual_return(close: pd.Series, market_close: pd.Series, bars: int,
                    beta_window: int = 78, skip: int = 1) -> pd.Series:
    """
    Trailing residual return = stock_ret - beta * market_ret, with beta estimated
    PAST-ONLY (rolling_beta, shifted). The clean way to separate alpha from beta.
    """
    beta = rolling_beta(close, market_close, beta_window, skip=skip)
    mkt = market_close.reindex(close.index).ffill()
    resid = close.pct_change(bars) - beta * mkt.pct_change(bars)
    return resid.rename(f"resid_ret_{bars}b")
