"""
The sacred test for the intraday research engine: features are PAST-ONLY and targets
use the future with the correct entry lag. A future bar must never change a past feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.research import features as F
from mft.research import targets as T


def make_5min(n_days: int = 8, seed: int = 0) -> pd.DataFrame:
    """Synthetic 5-min RTH bars (09:30–15:55 ET) over n_days business days."""
    rng = np.random.default_rng(seed)
    idx = []
    day = pd.Timestamp("2021-03-01", tz="America/New_York")
    made = 0
    while made < n_days:
        if day.weekday() < 5:
            bars = pd.date_range(day + pd.Timedelta("9h30m"), day + pd.Timedelta("15h55m"),
                                 freq="5min", tz="America/New_York")
            idx.append(bars)
            made += 1
        day += pd.Timedelta(days=1)
    index = pd.DatetimeIndex(np.concatenate([b.values for b in idx])).tz_localize("UTC")
    n = len(index)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, n)), index=index)
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": pd.Series(rng.integers(1_000, 50_000, n), index=index, dtype=float),
        "vwap": close,
    }, index=index)
    return df


PERTURB_AT = 200  # a bar in the middle


def _assert_past_unchanged(fn, df, perturb_col="close"):
    """Perturb one middle bar; every feature value strictly BEFORE it must be identical."""
    base = fn(df)
    dirty = df.copy()
    dirty.iloc[PERTURB_AT, dirty.columns.get_loc(perturb_col)] *= 5.0  # shock the future
    after = fn(dirty)
    pd.testing.assert_series_equal(
        base.iloc[:PERTURB_AT], after.iloc[:PERTURB_AT], check_names=False
    )


def test_trailing_return_is_past_only():
    df = make_5min()
    _assert_past_unchanged(lambda d: F.trailing_return(d["close"], 6), df)


def test_return_zscore_is_past_only():
    df = make_5min()
    _assert_past_unchanged(lambda d: F.return_zscore(d["close"], 3, 30), df)


def test_vwap_distance_is_past_only():
    df = make_5min()
    _assert_past_unchanged(F.vwap_distance, df)


def test_volume_zscore_is_past_only():
    df = make_5min()
    _assert_past_unchanged(lambda d: F.volume_zscore(d["volume"], 30), df, perturb_col="volume")


def test_relative_volume_tod_is_past_only():
    df = make_5min(n_days=12)
    _assert_past_unchanged(lambda d: F.relative_volume_tod(d, lookback_days=3),
                           df, perturb_col="volume")


def test_residual_return_is_past_only():
    df = make_5min(seed=1)
    mkt = make_5min(seed=2)["close"]
    _assert_past_unchanged(lambda d: F.residual_return(d["close"], mkt, 6), df)


def test_volume_zscore_tod_is_past_only():
    df = make_5min(n_days=12)
    _assert_past_unchanged(lambda d: F.volume_zscore_tod(d, lookback_days=3),
                           df, perturb_col="volume")


def test_orb_signal_is_past_only():
    from mft.research.breakout import opening_range_signal
    df = make_5min(seed=3)
    _assert_past_unchanged(lambda d: opening_range_signal(d, None, or_bars=3), df)


def test_beta_adjusted_forward_is_nan_across_session_boundary():
    df = make_5min(n_days=3, seed=4)
    mkt = make_5min(n_days=3, seed=5)["close"]
    fwd = T.beta_adjusted_forward_return(df["close"], mkt, 6, entry_lag=1, intraday_only=True)
    et_date = df.index.tz_convert("America/New_York").normalize()
    last = pd.Series(et_date, index=df.index).groupby(et_date.values).tail(1).index
    assert fwd.reindex(last).isna().all()


def test_forward_return_uses_future_with_entry_lag():
    """fwd at t must equal close[t+1+h]/close[t+1]-1 and NOT depend on close[t]."""
    df = make_5min()
    close = df["close"]
    h, lag = 6, 1
    fwd = T.forward_return(close, h, entry_lag=lag, intraday_only=False)
    # pick a t well inside one day
    t = 10
    expected = close.iloc[t + lag + h] / close.iloc[t + lag] - 1.0
    assert fwd.iloc[t] == pytest.approx(expected)

    # changing close[t] (the decision bar) must NOT change fwd[t] (it enters at t+1)
    dirty = df.copy()
    dirty.iloc[t, dirty.columns.get_loc("close")] *= 2.0
    fwd2 = T.forward_return(dirty["close"], h, entry_lag=lag, intraday_only=False)
    assert fwd2.iloc[t] == pytest.approx(expected)


def test_forward_return_is_nan_across_session_boundary():
    """A hold that would cross overnight must be NaN under intraday_only."""
    df = make_5min(n_days=3)
    close = df["close"]
    h = 6
    fwd = T.forward_return(close, h, entry_lag=1, intraday_only=True)
    et_date = close.index.tz_convert("America/New_York").normalize()
    # last few bars of each day cannot complete an h-bar hold same-session -> NaN
    last_bar_of_day = pd.Series(et_date, index=close.index).groupby(et_date.values).tail(1).index
    assert fwd.reindex(last_bar_of_day).isna().all()
