"""
Point-in-time (PIT) alignment.

PIT is the #1 source of look-ahead bias. Fundamentals and earnings must be
stamped with the date they were KNOWABLE, not the period they describe.
Apply reporting lags explicitly and verify with the look-ahead unit test.
"""

from __future__ import annotations

import pandas as pd


def make_pit_window(
    data: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    """
    Extract a PIT data window ending at `as_of` (inclusive).

    The window contains ONLY data knowable at `as_of`. This is the
    fundamental invariant the look-ahead unit test verifies.

    Args:
        data: Full OHLCV DataFrame, DatetimeIndex in UTC.
        as_of: The timestamp at which we are computing a signal.
        lookback: Number of bars of history required (alpha.lookback).

    Returns:
        DataFrame of shape (≤ lookback+1, n_cols), index <= as_of.
    """
    available = data[data.index <= as_of]
    return available.iloc[-(lookback + 1) :]


def align_pit(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    reporting_lag_days: int = 0,
) -> pd.DataFrame:
    """
    Align fundamental / alternative features to a price index using PIT semantics.

    Features become available only AFTER they are knowable. We apply a reporting
    lag (e.g. 1 day for overnight releases, 45 days for quarterly earnings) then
    forward-fill onto the price index.

    Args:
        feature_df: Features with DatetimeIndex = knowledge/release dates.
        price_df: Price data providing the target index.
        reporting_lag_days: Additional lag to apply before forward-filling.

    Returns:
        Features aligned to price_df.index via as-of join.
    """
    if reporting_lag_days > 0:
        feature_df = feature_df.copy()
        feature_df.index = feature_df.index + pd.Timedelta(days=reporting_lag_days)

    return feature_df.reindex(price_df.index, method="ffill")
