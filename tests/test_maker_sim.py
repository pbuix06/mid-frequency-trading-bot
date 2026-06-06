"""
Maker-fill logic: never fill without price touching/trading through the limit; conservative
trade-through is stricter than optimistic touch; timeouts, sides, exit modes, probabilistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.research.maker_sim import (
    MakerConfig,
    _exit_price,
    _passive_fill,
    aggregate,
    limit_price,
    simulate,
)


def test_limit_price_passive_side():
    assert limit_price(100.0, "buy", 10) == 99.95     # bid, below mid
    assert limit_price(100.0, "sell", 10) == 100.05   # ask, above mid


def test_model_A_touch_fills_on_reach_not_beyond():
    cfg = MakerConfig(model="A", timeout_bars=1, spread_bps=10)
    high = np.array([100.0, 100.0])
    low = np.array([100.0, 99.94])
    assert _passive_fill(high, low, 0, 2, 99.95, "buy", cfg, None) == 1   # low<=limit -> fill
    low2 = np.array([100.0, 99.96])
    assert _passive_fill(high, low2, 0, 2, 99.95, "buy", cfg, None) == -1  # never reached -> no fill


def test_model_B_requires_trade_through_buffer():
    cfg = MakerConfig(model="B", buffer_bps=2, timeout_bars=1, spread_bps=10)
    limit = 99.95
    low_touch = np.array([100.0, 99.94])    # touches but < buffer? 99.95*(1-2e-4)=99.93 -> 99.94 not below
    low_through = np.array([100.0, 99.92])  # below 99.93 -> trade-through
    h = np.array([100.0, 100.0])
    assert _passive_fill(h, low_touch, 0, 2, limit, "buy", cfg, None) == -1   # mere touch NOT enough in B
    assert _passive_fill(h, low_through, 0, 2, limit, "buy", cfg, None) == 1


def test_timeout_respected():
    cfg = MakerConfig(model="A", timeout_bars=2, spread_bps=10)
    h = np.zeros(5)
    low = np.array([100, 100, 100, 99.9, 99.9])   # reaches only at bar 3
    assert _passive_fill(h, low, 0, 5, 99.95, "buy", cfg, None) == -1   # outside 2-bar window
    cfg3 = MakerConfig(model="A", timeout_bars=3, spread_bps=10)
    assert _passive_fill(h, low, 0, 5, 99.95, "buy", cfg3, None) == 3


def test_sell_side_symmetric():
    cfg = MakerConfig(model="A", timeout_bars=1, spread_bps=10)
    high = np.array([100.0, 100.06])
    low = np.array([100.0, 100.0])
    assert _passive_fill(high, low, 0, 2, 100.05, "sell", cfg, None) == 1


def test_exit_modes_price_and_fee():
    close = np.array([100.0] * 10)
    high = np.array([100.1] * 10)
    low = np.array([99.9] * 10)
    # long, mode 1 taker -> sell at bid, taker fee
    px, fee, mk = _exit_price(close, high, low, 10, 0, "buy", MakerConfig(exit_mode=1, spread_bps=10, hold_bars=3), None)
    assert px == 100.0 * (1 - 5e-4) and fee == 5.0 and mk is False
    # long, mode 2 maker -> sell at ask, maker fee
    px2, fee2, mk2 = _exit_price(close, high, low, 10, 0, "buy", MakerConfig(exit_mode=2, spread_bps=10, maker_fee_bps=1, hold_bars=3), None)
    assert px2 == 100.0 * (1 + 5e-4) and fee2 == 1.0 and mk2 is True


def test_model_C_probability_extremes():
    rng = np.random.default_rng(0)
    cfg = MakerConfig(model="C", timeout_bars=1, spread_bps=10)
    deep = np.array([100.0, 90.0])        # huge penetration below limit -> prob ~1
    none = np.array([100.0, 100.0])        # never reaches -> prob 0
    h = np.array([100.0, 100.0])
    assert _passive_fill(h, deep, 0, 2, 99.95, "buy", cfg, rng) == 1
    assert _passive_fill(h, none, 0, 2, 99.95, "buy", cfg, rng) == -1


def test_simulate_filled_long_bounce_is_positive_gross():
    # a recent loser that bounces: dips to fill the bid, then rises by horizon
    ts = pd.date_range("2023-01-01", periods=12, freq="5min", tz="UTC")
    close = np.array([100, 99.9, 100.2, 100.5, 100.6, 100.7, 100.8, 100.9, 101, 101, 101, 101.0])
    low = close - 0.1
    high = close + 0.1
    bars = {"X": {"close": close, "high": high, "low": low, "ts": ts}}
    cfg = MakerConfig(model="A", spread_bps=10, maker_fee_bps=0, taker_fee_bps=0,
                      adverse_selection_bps=0, timeout_bars=2, exit_mode=1, hold_bars=4)
    tr = simulate([("X", 0, "buy")], bars, cfg)
    assert bool(tr["filled"].iloc[0]) and tr["gross"].iloc[0] > 0
    agg = aggregate(tr, years=0.1)
    assert agg["n_filled"] == 1 and agg["fill_rate"] == 1.0
