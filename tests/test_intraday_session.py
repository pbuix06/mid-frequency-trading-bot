"""Session-feature + longer-hold intraday strategy tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.backtest.intraday_session import gap_fade_returns, opening_range_break_returns
from mft.features.intraday import daily_session_features


def _minute_panel(days: list[list[float]]) -> pd.DataFrame:
    """days: list of per-day close paths (each a list of minute closes, 09:30 ET start)."""
    frames = []
    for d, closes in enumerate(days):
        start = pd.Timestamp("2023-01-03 09:30", tz="America/New_York") + pd.Timedelta(days=d)
        idx = pd.date_range(start, periods=len(closes), freq="1min").tz_convert("UTC")
        c = pd.Series(closes, index=idx, dtype=float)
        frames.append(pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e4}, index=idx))
    return pd.concat(frames)


def test_gap_and_intraday_ret_computed():
    # Day 1 closes at 100; Day 2 opens 102 (gap +2%) and closes 101.
    df = _minute_panel([[100.0] * 10, [102.0, 101.5, 101.0, 100.8, 101.0] + [101.0] * 5])
    feat = daily_session_features(df, or_minutes=2)
    day2 = feat.iloc[1]
    assert abs(day2["overnight_gap"] - (102.0 / 100.0 - 1)) < 1e-9
    assert abs(day2["intraday_ret"] - (101.0 / 102.0 - 1)) < 1e-9


def test_gap_fade_profits_when_gap_reverts():
    # Gap up then drift down -> fade (short) is profitable.
    df = _minute_panel([[100.0] * 10, [102.0] + [101.0] * 9])
    feat = daily_session_features(df, or_minutes=2)
    r = gap_fade_returns(feat, threshold=0.0, cost_per_trade=0.0)
    # day2: pos=-1, intraday_ret = 101/102-1 < 0 -> product > 0
    assert r.iloc[-1] > 0


def test_gap_fade_threshold_skips_small_gaps():
    df = _minute_panel([[100.0] * 10, [100.05] + [100.0] * 9])  # 5bp gap
    feat = daily_session_features(df, or_minutes=2)
    r = gap_fade_returns(feat, threshold=0.01, cost_per_trade=0.0)  # need >1% gap
    assert r.iloc[-1] == 0.0   # tiny gap -> flat


def test_opening_range_breakout_direction():
    # Flat opening range then breaks UP and closes higher -> long profits.
    path = [100.0, 100.0] + [100.5, 101.0, 101.5, 102.0] + [102.0] * 4
    df = _minute_panel([[100.0] * 12, path])
    feat = daily_session_features(df, or_minutes=2)
    r = opening_range_break_returns(feat, cost_per_trade=0.0)
    assert feat.iloc[1]["brk_dir"] == 1
    assert r.iloc[-1] > 0   # broke up, closed up -> long wins


def test_no_lookahead_features_only_use_own_and_prior_days():
    df = _minute_panel([[100.0] * 10, [102.0] + [101.0] * 9, [99.0] + [98.0] * 9])
    feat = daily_session_features(df, or_minutes=2)
    # Poison day 3 entirely; day-2 features must be unchanged.
    df2 = df.copy()
    last_day = df2.index.tz_convert("America/New_York").date == pd.Timestamp("2023-01-05").date()
    df2.loc[last_day, ["open", "high", "low", "close"]] = 1e6
    feat2 = daily_session_features(df2, or_minutes=2)
    pd.testing.assert_series_equal(feat.iloc[1], feat2.iloc[1])
