"""
Tests for IntradayAfternoonContinuation: PIT safety, flat-morning / hold-afternoon /
flat-into-close, direction = sign(morning return), and target_series == compute_signal.
"""

from __future__ import annotations

import pandas as pd

from mft.alphas.intraday_continuation import IntradayAfternoonContinuation


def _session(values: list[float], day: str = "2023-01-03") -> pd.DataFrame:
    """One trading day of flat 1-min bars (o=h=l=c=value) from 09:30 ET."""
    start = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    idx = pd.date_range(start, periods=len(values), freq="1min").tz_convert("UTC")
    c = pd.Series(values, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e4}, index=idx)


def _signals(alpha, panel) -> list[float]:
    return [alpha.compute_signal(panel.iloc[: i + 1])[alpha.symbol] for i in range(len(panel))]


# 09:30 -> 12:00 is 150 min; use a short decision time so synthetic days stay small.
DEC = "09:33"  # decision after 3 morning bars
ALPHA_KW = dict(decision_time=DEC, min_morning_bars=2, flatten_before_close_min=1)


def test_flat_in_the_morning_before_decision():
    alpha = IntradayAfternoonContinuation("X", **ALPHA_KW)
    # up morning: 100 -> 102 by the decision bar, then drift
    panel = _session([100.0, 101.0, 102.0, 102.5, 103.0, 103.0, 103.0, 103.0])
    sig = _signals(alpha, panel)
    # bars before the decision time (09:30, 09:31, 09:32) carry no position
    assert sig[0] == 0.0 and sig[1] == 0.0 and sig[2] == 0.0


def test_long_when_morning_up():
    alpha = IntradayAfternoonContinuation("X", **ALPHA_KW)
    panel = _session([100.0, 101.0, 102.0, 102.5, 103.0, 103.0, 103.0, 103.0])
    sig = _signals(alpha, panel)
    # from the decision bar (index 3) onward (pre-close-flat) -> long
    assert sig[3] == 1.0 and sig[4] == 1.0


def test_short_when_morning_down():
    alpha = IntradayAfternoonContinuation("X", **ALPHA_KW)
    panel = _session([100.0, 99.0, 98.0, 97.5, 97.0, 97.0, 97.0, 97.0])
    sig = _signals(alpha, panel)
    assert sig[3] == -1.0 and sig[4] == -1.0


def test_threshold_skips_flat_mornings():
    alpha = IntradayAfternoonContinuation(
        "X", decision_time=DEC, min_morning_bars=2, threshold=0.05
    )
    panel = _session([100.0, 100.05, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # ~0% morning
    sig = _signals(alpha, panel)
    assert all(s == 0.0 for s in sig)


def test_flat_into_close_no_overnight():
    alpha = IntradayAfternoonContinuation("X", **ALPHA_KW)
    # Full session 09:30->15:59 so the close-flatten actually triggers.
    panel = _session([100.0, 101.0, 102.0] + [103.0] * 387)  # 390 bars
    sig = _signals(alpha, panel)
    assert sig[-1] == 0.0          # flat in the final minute (no overnight)
    assert sig[100] == 1.0         # but holding the continuation mid-session


def test_future_bars_cannot_change_earlier_signal():
    alpha = IntradayAfternoonContinuation("X", **ALPHA_KW)
    panel = _session([100.0, 101.0, 102.0, 102.5, 103.0, 103.0, 103.0, 103.0])
    clean = _signals(alpha, panel)
    poisoned = panel.copy()
    poisoned.iloc[4:, :] = 1e6  # explode everything after the decision bar
    after = _signals(alpha, poisoned)
    assert clean[:4] == after[:4]


def test_target_series_equals_bar_by_bar():
    alpha = IntradayAfternoonContinuation("X", **ALPHA_KW)
    d1 = _session([100.0, 101.0, 102.0, 102.5, 103.0, 103.0, 103.0, 103.0], day="2023-01-03")
    d2 = _session([100.0, 99.0, 98.0, 97.5, 97.0, 97.0, 97.0, 97.0], day="2023-01-04")
    panel = pd.concat([d1, d2])
    vec = alpha.target_series(panel)
    bybar = pd.Series(_signals(alpha, panel), index=panel.index)
    pd.testing.assert_series_equal(vec, bybar, check_names=False)
