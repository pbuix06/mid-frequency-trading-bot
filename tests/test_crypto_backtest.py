"""
24/7 crypto backtest mode: the rebalance grid must be CONTINUOUS (no ET-day reset, no
RTH, weekend bars kept, holds span midnight). Contrasted against the equity mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.research.xs_backtest import _rebalance_points, cross_sectional_ls


def crypto_panel(n_days: int = 3, n_sym: int = 6, seed: int = 0) -> pd.DataFrame:
    """Synthetic 24/7 5-min panel starting on a Friday (so it includes a weekend)."""
    idx = pd.date_range("2023-03-03 00:00", periods=n_days * 288, freq="5min", tz="UTC")  # Fri 00:00
    rng = np.random.default_rng(seed)
    cols = {f"C{i}": 100 + np.cumsum(rng.normal(0, 0.2, len(idx))) for i in range(n_sym)}
    return pd.DataFrame(cols, index=idx)


def test_continuous_grid_is_uniform_no_et_reset():
    idx = crypto_panel().index
    reb = _rebalance_points(idx, hold_bars=6, continuous=True)
    pos = idx.get_indexer(reb)
    assert np.all(np.diff(pos) == 6)          # perfectly uniform — never resets at midnight


def test_equity_grid_does_reset_at_day_boundaries():
    idx = crypto_panel().index
    # hold_bars=7 does not evenly divide the ET-day segment lengths, so the per-day reset
    # shows up as non-uniform spacing (the continuous grid would still be perfectly uniform).
    reb = _rebalance_points(idx, hold_bars=7, continuous=False)
    pos = idx.get_indexer(reb)
    assert not np.all(np.diff(pos) == 7)      # ET-day reset creates short end-of-day buckets
    assert np.all(np.diff(idx.get_indexer(_rebalance_points(idx, 7, continuous=True))) == 7)


def test_continuous_keeps_weekend_bars():
    idx = crypto_panel().index
    reb = _rebalance_points(idx, hold_bars=6, continuous=True)
    assert (reb.weekday >= 5).any()           # Saturday/Sunday decision bars exist (24/7)


def test_tod_windows_rejected_in_continuous():
    idx = crypto_panel().index
    with pytest.raises(ValueError):
        _rebalance_points(idx, hold_bars=6, tod_windows=[("09:30", "11:30")], continuous=True)


def test_continuous_backtest_has_more_trades_and_spans_midnight():
    close = crypto_panel(n_days=4, seed=1)
    sig = close.rank(axis=1)
    cont = cross_sectional_ls(sig, close, top_n=2, hold_bars=6, continuous=True)
    eq = cross_sectional_ls(sig, close, top_n=2, hold_bars=6, continuous=False)
    # 24/7 mode trades through midnight & weekends; equity mode drops day-end + midnight-spanning holds
    assert cont.n_trades > eq.n_trades
    # weekend trades present in continuous net
    assert (cont.net.index.weekday >= 5).any()
    # a decision bar just before ET-midnight (05:00 UTC in EST) produced a (non-nulled) trade
    et = cont.net.index.tz_convert("America/New_York")
    assert (et.hour == 23).any()
