"""
Session backtest for longer-hold intraday strategies (one decision/day).

Each day: take a position decided from pre-holding-period info, hold open->close
(flat overnight), realize the holding return, pay a round-trip cost (enter + exit).
Daily return = position * holding_return - roundtrip_cost * (position != 0).

This is the taker-viable band (Path B): ~one round-trip/day, so the per-trade
edge can exceed the spread — unlike per-minute reversal.

Strategies (position functions on the session-features frame):
  gap_fade            : fade the overnight gap (short gap-ups, long gap-downs);
                        hold open->close. Decided AT the open.
  opening_range_break : trade the first opening-range breakout in its direction;
                        enter at the breakout level, hold to close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# liquid-name microstructure: ~1.5 bp per trade -> ~3 bp round-trip
BASE_COST_PER_TRADE = 0.00015


def gap_fade_returns(
    feat: pd.DataFrame,
    threshold: float = 0.0,
    cost_per_trade: float = BASE_COST_PER_TRADE,
) -> pd.Series:
    """
    Fade the overnight gap, held open->close.
    Only act when |gap| > threshold (skip tiny gaps that don't clear costs).
    """
    gap = feat["overnight_gap"]
    pos = pd.Series(0.0, index=feat.index)
    pos[gap > threshold] = -1.0     # gapped up -> short the fade
    pos[gap < -threshold] = 1.0     # gapped down -> long the fade
    ret = pos * feat["intraday_ret"] - 2.0 * cost_per_trade * (pos != 0)
    return ret.dropna().rename("gap_fade")


def opening_range_break_returns(
    feat: pd.DataFrame,
    cost_per_trade: float = BASE_COST_PER_TRADE,
    entry_slippage_bps: float = 0.0,
) -> pd.Series:
    """
    Trade the first opening-range breakout in its direction; enter at the
    breakout level (or_high / or_low), hold to close. Intraday momentum.

    `entry_slippage_bps` worsens the entry AGAINST the trade (you don't fill
    exactly at the level — a stop runs as price moves). 0 = the optimistic
    fill-at-level the research used. This knob exposes how execution-sensitive the
    edge is: the gap to a realistic taker fill is the whole question (spec §8).
    """
    pos = feat["brk_dir"].astype(float)
    slip = entry_slippage_bps * 1e-4
    entry = feat["brk_entry"] * (1.0 + pos * slip)   # adverse: pay up to go long, etc.
    hold = np.where(pos != 0, feat["close"] / entry - 1.0, 0.0)
    ret = pos * pd.Series(hold, index=feat.index) - 2.0 * cost_per_trade * (pos != 0)
    return ret.dropna().rename("or_break")
