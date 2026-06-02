"""
Data cleaning: corporate actions, outlier handling, timezone discipline.

Build and unit-test each function independently.
Wrong split adjustment = fake price gaps = fake signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index is UTC. All bars must be UTC internally."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")
    return df


def adjust_splits(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-adjust OHLCV for stock splits.

    Args:
        df: OHLCV DataFrame, sorted ascending.
        splits: DataFrame with columns [ex_date (DatetimeIndex), ratio (new/old)].

    Returns:
        Adjusted DataFrame. Prices before ex_date divided by ratio;
        volume multiplied by ratio (shares are cheaper, more of them).
    """
    df = df.copy()
    for _, row in splits.iterrows():
        mask = df.index < row["ex_date"]
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df.loc[mask, col] /= row["ratio"]
        if "volume" in df.columns:
            df.loc[mask, "volume"] *= row["ratio"]
    return df


def remove_outliers(
    df: pd.DataFrame,
    z_threshold: float = 5.0,
    fill: str = "ffill",
    window: int = 63,
) -> pd.DataFrame:
    """
    Detect and neutralize price outliers using a CAUSAL z-score on log returns.

    PIT-safe: each bar's z-score is computed from a TRAILING window of prior
    returns only (shifted by 1 so the current bar never enters its own
    statistics). Full-sample mean/std would leak future information into early
    bars — never use that in PIT research.

    Args:
        df: OHLCV DataFrame.
        z_threshold: Z-score above which a bar is considered an outlier.
        fill: How to handle outlier rows — "ffill" (forward-fill) or "drop".
        window: Trailing window (bars) for the rolling mean/std. Default 63 (~3mo).

    Returns:
        Cleaned DataFrame. Outlier OHLCV values replaced by NaN then filled.
    """
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # Trailing stats excluding the current bar (shift(1)). Min one quarter of
    # the window before the test activates, so the warmup period is never flagged.
    roll = log_ret.shift(1).rolling(window, min_periods=max(20, window // 4))
    mu = roll.mean()
    sigma = roll.std()

    z = (log_ret - mu).abs() / (sigma + 1e-10)
    outlier_mask = (z > z_threshold).fillna(False)

    for col in ("open", "high", "low", "close"):
        df.loc[outlier_mask, col] = np.nan

    if fill == "ffill":
        df = df.ffill()
    elif fill == "drop":
        df = df.dropna(subset=["close"])

    return df


def fill_missing_bars(
    df: pd.DataFrame,
    freq: str = "B",
) -> pd.DataFrame:
    """
    Forward-fill missing bars (weekends, holidays, halts).
    Volume is set to 0 on filled bars to distinguish them from traded bars.
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    df = df.reindex(full_index)
    df["volume"] = df["volume"].fillna(0.0)
    df = df.ffill()
    return df
