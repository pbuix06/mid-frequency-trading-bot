"""
Funding logic: 8h funding maps to bars KNOWN-ONLY (no future funding), z-score is causal,
carry over a hold = cum_funding[exit] - cum_funding[entry], and the carry sign is right
(a short on positive funding RECEIVES it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.research.crypto_panel import (
    cumulative_funding_series,
    funding_zscore_series,
    map_to_bars,
    oi_change,
)
from mft.research.funding_backtest import funding_ls_backtest


def _funding():
    idx = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 08:00", "2023-01-01 16:00"], utc=True)
    return pd.Series([0.01, 0.02, 0.03], index=idx)


def test_map_to_bars_is_known_only():
    f = _funding()
    bars = pd.date_range("2023-01-01 00:00", "2023-01-01 20:00", freq="4h", tz="UTC")
    m = map_to_bars(f, bars)
    # bar at/after a funding time carries the last KNOWN funding; nothing from the future
    assert list(m.values) == [0.01, 0.01, 0.02, 0.02, 0.03, 0.03]
    # a bar before the first funding -> NaN (not knowable)
    pre = pd.DatetimeIndex(["2022-12-31 12:00"]).tz_localize("UTC")
    assert pd.isna(map_to_bars(f, pre).iloc[0])


def test_map_to_bars_future_change_does_not_touch_past():
    f = _funding()
    bars = pd.date_range("2023-01-01 00:00", "2023-01-01 20:00", freq="4h", tz="UTC")
    base = map_to_bars(f, bars)
    f2 = f.copy()
    f2.iloc[-1] = 0.99            # change the LAST (future) funding
    after = map_to_bars(f2, bars)
    assert (base.iloc[:4].values == after.iloc[:4].values).all()   # earlier bars unchanged


def test_cumulative_funding_carry_window():
    f = _funding()
    cum = cumulative_funding_series(f)               # [0.01, 0.03, 0.06]
    bars = pd.date_range("2023-01-01 00:00", "2023-01-01 20:00", freq="4h", tz="UTC")
    cb = map_to_bars(cum, bars)
    # carry over (04:00, 16:00] = funding at 08:00 + 16:00 = 0.05
    carry = cb.loc["2023-01-01 16:00"] - cb.loc["2023-01-01 04:00"]
    assert carry == pytest.approx(0.05)


def test_funding_zscore_is_causal():
    f = pd.Series(np.arange(20.0), index=pd.date_range("2023-01-01", periods=20, freq="8h", tz="UTC"))
    z = funding_zscore_series(f, window=5)
    base = z.copy()
    f2 = f.copy()
    f2.iloc[15] *= 10                  # perturb a later value
    z2 = funding_zscore_series(f2, window=5)
    pd.testing.assert_series_equal(base.iloc[:15], z2.iloc[:15])


def test_oi_change_causal():
    oi = pd.Series(np.arange(10.0) + 1, index=pd.date_range("2023-01-01", periods=10, freq="5min", tz="UTC"))
    ch = oi_change(oi, 2)
    assert pd.isna(ch.iloc[0]) and abs(ch.iloc[2] - (3 / 1 - 1)) < 1e-12


def test_funding_carry_sign_short_receives_positive_funding():
    idx = pd.date_range("2023-01-01", periods=6, freq="5min", tz="UTC")
    syms = ["A", "B", "C", "D"]
    close = pd.DataFrame(100.0, index=idx, columns=syms)            # flat price -> price PnL 0
    sig = pd.DataFrame(np.tile([0, 1, 2, 3], (6, 1)), index=idx, columns=syms)  # D long, A short
    cum = pd.DataFrame(0.0, index=idx, columns=syms)
    cum["A"] = np.linspace(0, 0.005, 6)                            # A (the short) accrues +funding
    res = funding_ls_backtest(sig, close, cum, top_n=1, hold_bars=2, cost_bps_per_side=0.0)
    assert res.metrics["funding_bps_per_trade"] > 0                # short receives -> positive carry
    assert abs(res.metrics["price_bps_per_trade"]) < 1e-6          # flat price
