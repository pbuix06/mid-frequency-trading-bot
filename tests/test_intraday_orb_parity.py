"""
Parity: the production IntradayORB sleeve must agree with the research session
feature builder on each session's breakout DIRECTION.

This is the "one code path" guarantee for the ORB candidate — the research script
(via daily_session_features.brk_dir) and the live alpha must not diverge on the
signal. They legitimately differ on execution TIMING (the alpha enters next-bar,
the research approximates a fill at the breakout level), so we assert direction
parity, not PnL parity.
"""

from __future__ import annotations

import pandas as pd

from mft.alphas.intraday_orb import IntradayORB
from mft.features.intraday import daily_session_features

OR = 2


def _minute_panel(days: list[list[float]], *, highs=None, lows=None) -> pd.DataFrame:
    frames = []
    for d, closes in enumerate(days):
        start = pd.Timestamp("2023-01-03 09:30", tz="America/New_York") + pd.Timedelta(days=d)
        idx = pd.date_range(start, periods=len(closes), freq="1min").tz_convert("UTC")
        c = pd.Series(closes, index=idx, dtype=float)
        h = pd.Series(highs[d], index=idx, dtype=float) if highs else c
        low = pd.Series(lows[d], index=idx, dtype=float) if lows else c
        frames.append(pd.DataFrame({"open": c, "high": h, "low": low, "close": c, "volume": 1e4}, index=idx))
    return pd.concat(frames)


def _alpha_session_dirs(alpha: IntradayORB, panel: pd.DataFrame) -> dict:
    """Held breakout direction per ET date from the bar-by-bar alpha."""
    dirs: dict = {}
    for i in range(len(panel)):
        sig = alpha.compute_signal(panel.iloc[: i + 1])[alpha.symbol]
        if sig != 0.0:
            d = panel.index[i].tz_convert("America/New_York").date()
            dirs[d] = sig   # last non-flat wins; held direction is constant intra-session
    return dirs


def test_orb_direction_matches_feature_builder():
    # Day 0: OR=[99,101] then breaks UP via a 102 high.    -> +1
    # Day 1: OR=[99,101] then breaks DOWN via a 98 low.    -> -1
    # Day 2: stays inside [99,101] the whole session.      ->  0 (no trade)
    closes = [
        [100.0, 101.0, 100.5, 101.8, 101.5, 101.0, 101.0],
        [100.0, 101.0, 100.5, 98.2, 98.5, 99.0, 99.0],
        [100.0, 101.0, 100.5, 100.2, 100.5, 100.0, 100.0],
    ]
    highs = [
        [100.0, 101.0, 100.5, 102.0, 101.5, 101.0, 101.0],
        [100.0, 101.0, 100.5, 98.2, 98.5, 99.0, 99.0],
        [100.0, 101.0, 100.5, 100.2, 100.5, 100.0, 100.0],
    ]
    lows = [
        [100.0, 99.0, 100.5, 101.8, 101.5, 101.0, 101.0],
        [100.0, 99.0, 100.5, 98.0, 98.5, 99.0, 99.0],
        [100.0, 99.0, 100.5, 100.2, 100.5, 100.0, 100.0],
    ]
    panel = _minute_panel(closes, highs=highs, lows=lows)

    feat = daily_session_features(panel, or_minutes=OR)
    # alpha with a 0-minute close buffer so the last bar still carries the held dir
    alpha = IntradayORB("X", or_minutes=OR, flatten_before_close_min=0)
    alpha_dirs = _alpha_session_dirs(alpha, panel)

    for day, row in feat.iterrows():
        d = day.tz_convert("America/New_York").date()
        expected = float(row["brk_dir"])
        got = alpha_dirs.get(d, 0.0)
        assert got == expected, f"{d}: alpha {got} != feature builder {expected}"


def test_target_series_equals_bar_by_bar_compute_signal():
    """The vectorized research twin must be byte-identical to the canonical live
    per-bar path on every bar (preserves 'one code path')."""
    closes = [
        [100.0, 101.0, 100.5, 101.8, 101.5, 101.0, 101.0],
        [100.0, 101.0, 100.5, 98.2, 98.5, 99.0, 99.0],
        [100.0, 101.0, 100.5, 100.2, 100.5, 100.0, 100.0],
    ]
    highs = [
        [100.0, 101.0, 100.5, 102.0, 101.5, 101.0, 101.0],
        [100.0, 101.0, 100.5, 98.2, 98.5, 99.0, 99.0],
        [100.0, 101.0, 100.5, 100.2, 100.5, 100.0, 100.0],
    ]
    lows = [
        [100.0, 99.0, 100.5, 101.8, 101.5, 101.0, 101.0],
        [100.0, 99.0, 100.5, 98.0, 98.5, 99.0, 99.0],
        [100.0, 99.0, 100.5, 100.2, 100.5, 100.0, 100.0],
    ]
    panel = _minute_panel(closes, highs=highs, lows=lows)
    alpha = IntradayORB("X", or_minutes=OR, flatten_before_close_min=1)

    vec = alpha.target_series(panel)
    bybar = pd.Series(
        [alpha.compute_signal(panel.iloc[: i + 1])["X"] for i in range(len(panel))],
        index=panel.index,
    )
    pd.testing.assert_series_equal(vec, bybar, check_names=False)
