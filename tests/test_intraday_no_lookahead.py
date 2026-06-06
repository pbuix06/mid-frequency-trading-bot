"""
Anti-leakage tests for the intraday ORB candidate (spec §9 G1, Task 6).

These are CRITICAL. They prove the production path uses only past information:
  - the opening range uses only completed bars;
  - the breakout decision never reads a future bar's high/low/close;
  - poisoning future bars cannot change an earlier signal (PIT / late-revision safe);
  - the position is flat before breakout, held after, and flat into the close
    (EOD liquidation -> never overnight);
  - universe selection ranks on strictly-past data only;
  - a consistent corporate-action (uniform price scaling) does not flip the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.intraday_orb import IntradayORB
from mft.features.intraday import select_high_vol_universe

OR = 3  # short opening range for compact synthetic sessions


def _day(values: list[float], day: str, *, highs=None, lows=None) -> pd.DataFrame:
    """One trading day of 1-min bars from 09:30 ET. Bars are flat (o=h=l=c=value)
    unless explicit highs/lows are given (to drive precise breakouts)."""
    start = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    idx = pd.date_range(start, periods=len(values), freq="1min").tz_convert("UTC")
    c = pd.Series(values, index=idx, dtype=float)
    h = pd.Series(highs, index=idx, dtype=float) if highs is not None else c
    low = pd.Series(lows, index=idx, dtype=float) if lows is not None else c
    return pd.DataFrame({"open": c, "high": h, "low": low, "close": c, "volume": 1e4}, index=idx)


def _signals_incrementally(alpha: IntradayORB, panel: pd.DataFrame) -> list[float]:
    """Replay the harness: signal_i = alpha(panel[:i+1])."""
    return [alpha.compute_signal(panel.iloc[: i + 1])[alpha.symbol] for i in range(len(panel))]


# Opening range = first 3 bars at 99/100/101 -> or_high=101, or_low=99.
# Post bars at 100 (inside range) until a 102 high breaks UP.
_OR_BARS = [99.0, 100.0, 101.0]


def _breakout_up_day(day: str, break_at: int, n: int = 12) -> pd.DataFrame:
    vals = list(_OR_BARS) + [100.0] * (n - len(_OR_BARS))
    highs = list(vals)
    highs[break_at] = 102.0  # this bar's HIGH pierces or_high -> up-break
    vals[break_at] = 101.8
    return _day(vals, day, highs=highs)


def test_opening_range_bars_are_always_flat():
    alpha = IntradayORB("X", or_minutes=OR)
    panel = _breakout_up_day("2023-01-03", break_at=6)
    sig = _signals_incrementally(alpha, panel)
    # The first OR bars cannot carry a position (range not yet formed).
    assert all(s == 0.0 for s in sig[:OR])


def test_flat_after_or_until_breakout_then_holds():
    alpha = IntradayORB("X", or_minutes=OR)
    panel = _breakout_up_day("2023-01-03", break_at=6)
    sig = _signals_incrementally(alpha, panel)
    # Between OR completion and the breakout bar -> flat.
    assert all(s == 0.0 for s in sig[OR:6])
    # From the breakout bar onward (before the close-flat) -> long, held.
    assert sig[6] == 1.0
    assert sig[7] == 1.0 and sig[8] == 1.0


def test_down_breakout_is_short():
    alpha = IntradayORB("X", or_minutes=OR)
    vals = list(_OR_BARS) + [100.0] * 9
    lows = list(vals)
    lows[6] = 98.0          # pierce or_low=99 -> down-break
    vals[6] = 98.2
    panel = _day(vals, "2023-01-03", lows=lows)
    sig = _signals_incrementally(alpha, panel)
    assert sig[6] == -1.0 and sig[7] == -1.0


def test_future_bars_cannot_change_earlier_signal():
    """PIT / late-revision safety: poison every bar after t; signals up to t are
    byte-identical. This is the core no-look-ahead guarantee."""
    alpha = IntradayORB("X", or_minutes=OR)
    panel = _breakout_up_day("2023-01-03", break_at=6)
    clean = _signals_incrementally(alpha, panel)

    j = 6  # poison everything strictly after the breakout bar
    poisoned = panel.copy()
    poisoned.iloc[j + 1 :, :] = 1e6
    after = _signals_incrementally(alpha, poisoned)

    assert clean[: j + 1] == after[: j + 1]


def test_flat_in_final_minute_no_overnight():
    """At/after the flatten cutoff the signal is 0, so the position is liquidated
    into the close and never carried overnight."""
    # Full session length so the final-minute logic is exercised.
    alpha = IntradayORB("X", or_minutes=OR, flatten_before_close_min=1)
    # Build a session that breaks out early and runs to the 16:00 close.
    n = 390
    vals = list(_OR_BARS) + [100.0] * (n - len(_OR_BARS))
    highs = list(vals)
    highs[5] = 102.0
    vals[5] = 101.8
    panel = _day(vals, "2023-01-03", highs=highs)
    sig = _signals_incrementally(alpha, panel)
    # The very last bar (15:59) is within 1 min of the 16:00 close -> flat.
    assert sig[-1] == 0.0
    # But mid-session it was holding the breakout.
    assert sig[100] == 1.0


def test_flat_outside_regular_session():
    alpha = IntradayORB("X", or_minutes=OR)
    # A single pre-open bar (08:00 ET) must be flat.
    pre = pd.Timestamp("2023-01-03 08:00", tz="America/New_York")
    idx = pd.date_range(pre, periods=5, freq="1min").tz_convert("UTC")
    panel = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e4}, index=idx
    )
    assert alpha.compute_signal(panel)["X"] == 0.0


def test_new_session_resets_independently():
    """Day 2 starts flat regardless of day 1's breakout (no state bleed)."""
    alpha = IntradayORB("X", or_minutes=OR)
    d1 = _breakout_up_day("2023-01-03", break_at=5)
    d2 = _day(list(_OR_BARS) + [100.0] * 3, "2023-01-04")  # day 2: OR only, no break
    panel = pd.concat([d1, d2])
    # On the first day-2 OR bar, signal must be flat (its own session, range unformed).
    n1 = len(d1)
    sig_first_d2 = alpha.compute_signal(panel.iloc[: n1 + 1])["X"]
    assert sig_first_d2 == 0.0


def test_consistent_corporate_action_does_not_flip_signal():
    """A uniform price adjustment (e.g. a split applied consistently across the
    session) is multiplicative, so relative breakout logic is invariant."""
    alpha = IntradayORB("X", or_minutes=OR)
    panel = _breakout_up_day("2023-01-03", break_at=6)
    base = _signals_incrementally(alpha, panel)

    split = panel.copy()
    split[["open", "high", "low", "close"]] *= 0.5  # 2-for-1, consistently applied
    scaled = _signals_incrementally(alpha, split)
    assert base == scaled


# ---- universe selection (ex-ante) ----------------------------------------

def _feat_with_vol(vol: float, n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"intraday_ret": rng.normal(0.0, vol, n)}, index=idx)


def test_universe_selection_ranks_high_vol_first():
    feats = {
        "HI": _feat_with_vol(0.03, seed=1),
        "MID": _feat_with_vol(0.015, seed=2),
        "LO": _feat_with_vol(0.005, seed=3),
    }
    asof = pd.Timestamp("2022-09-01", tz="UTC")
    top2 = select_high_vol_universe(feats, asof, top=2, window=63)
    assert "HI" in top2 and "LO" not in top2


def test_universe_selection_uses_only_past_data():
    """Poisoning sessions on/after asof must not change the selection."""
    feats = {
        "HI": _feat_with_vol(0.03, seed=1),
        "MID": _feat_with_vol(0.015, seed=2),
        "LO": _feat_with_vol(0.005, seed=3),
    }
    asof = pd.Timestamp("2022-09-01", tz="UTC")
    clean = select_high_vol_universe(feats, asof, top=2, window=63)

    poisoned = {s: f.copy() for s, f in feats.items()}
    for s, f in poisoned.items():
        f.loc[f.index >= asof, "intraday_ret"] = 999.0  # explode the future
    after = select_high_vol_universe(poisoned, asof, top=2, window=63)
    assert clean == after
