"""
Engine correctness: resampling aggregation, IC sanity (perfect/inverse/none), bucket
monotonicity, split discipline, and a backtest smoke test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.research import signal_lab as L
from mft.research import splits as S
from mft.research.panel import resample_bars
from mft.research.xs_backtest import cross_sectional_ls
from tests.test_research_no_lookahead import make_5min


def test_resample_aggregates_ohlcv_correctly():
    # one RTH day of 1-min bars with a known ramp
    idx = pd.date_range("2021-03-01 09:30", "2021-03-01 09:39", freq="1min",
                        tz="America/New_York").tz_convert("UTC")
    df = pd.DataFrame({
        "open": np.arange(10, dtype=float),
        "high": np.arange(10, dtype=float) + 0.5,
        "low": np.arange(10, dtype=float) - 0.5,
        "close": np.arange(10, dtype=float) + 0.1,
        "volume": np.ones(10) * 100,
        "vwap": np.arange(10, dtype=float),
    }, index=idx)
    out = resample_bars(df, "5min")
    assert len(out) == 2                         # 10 one-min bars -> 2 five-min bars
    first = out.iloc[0]
    assert first["open"] == 0.0                  # first
    assert first["high"] == 4.5                  # max
    assert first["low"] == -0.5                  # min
    assert first["close"] == 4.1                 # last
    assert first["volume"] == 500.0              # sum


def _wide(seed, n_days=6):
    syms = list("ABCDEFGHIJ")
    rng = np.random.default_rng(seed)
    base = make_5min(n_days=n_days)
    cols = {}
    for s in syms:
        c = 100 + np.cumsum(rng.normal(0, 0.2, len(base)))
        cols[s] = pd.Series(c, index=base.index)
    return pd.DataFrame(cols)


def test_ic_perfect_and_inverse_and_none():
    close = _wide(0)
    # forward target
    from mft.research.targets import forward_return_panel
    tgt = forward_return_panel(close, 6, entry_lag=1, intraday_only=True)
    # a signal equal to the target should have IC = +1 where defined
    ic_pos = L.ic_series(tgt, tgt, method="spearman")
    assert ic_pos.mean() == pytest.approx(1.0, abs=1e-6)
    # inverse signal -> -1
    ic_neg = L.ic_series(-tgt, tgt, method="spearman")
    assert ic_neg.mean() == pytest.approx(-1.0, abs=1e-6)
    # independent random signal -> ~0
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(rng.normal(size=tgt.shape), index=tgt.index, columns=tgt.columns)
    ic_zero = L.ic_summary(L.ic_series(noise, tgt, method="spearman"))
    assert abs(ic_zero["ic_mean"]) < 0.05


def test_bucket_monotonicity_for_predictive_signal():
    close = _wide(1)
    from mft.research.targets import forward_return_panel
    tgt = forward_return_panel(close, 6)
    br = L.bucket_returns(tgt, tgt, n_buckets=5)   # signal == target -> perfectly monotone
    assert L.bucket_monotonicity(br) == pytest.approx(1.0, abs=1e-9)


def test_backtest_runs_and_is_finite():
    close = _wide(2)
    res = cross_sectional_ls(close.rank(axis=1), close, top_n=3, hold_bars=6, cost_bps_per_side=2.0)
    assert res.n_trades > 0
    assert np.isfinite(res.metrics["sharpe"])
    # net must be below gross (costs are actually charged)
    assert res.net.mean() < res.gross.mean()
    # leg + win/loss reporting present
    for k in ("long_leg_bps_per_trade", "short_leg_bps_per_trade", "avg_win_bps", "avg_loss_bps"):
        assert k in res.metrics


def test_orb_signal_sign_breaks_up():
    from mft.research.breakout import opening_range_signal
    idx = pd.date_range("2021-03-01 09:30", "2021-03-01 12:00", freq="5min",
                        tz="America/New_York").tz_convert("UTC")
    n = len(idx)
    close = np.concatenate([[100, 100.5, 99.5], np.linspace(101, 110, n - 3)])  # OR=3 bars, then up
    df = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2,
                       "close": close, "volume": np.ones(n) * 1000, "vwap": close}, index=idx)
    sig = opening_range_signal(df, None, or_bars=3).dropna()
    assert len(sig) > 0 and (sig > 0).all()   # broke above OR and stayed up -> positive signal


def test_tod_windows_restricts_trading_to_allowed_times():
    close = _wide(3)
    sig = close.rank(axis=1)
    # only allow the first 30 min of the session
    res = cross_sectional_ls(sig, close, top_n=3, hold_bars=1,
                             tod_windows=[("09:30", "10:00")])
    et_times = res.net.index.tz_convert("America/New_York").time
    assert len(res.net) > 0
    assert all(pd.Timestamp("09:30").time() <= t < pd.Timestamp("10:00").time()
               for t in et_times)


def test_lockbox_is_sealed():
    with pytest.raises(PermissionError):
        S.load_lockbox()
    assert S.load_lockbox(i_am_done_tuning=True).name == "lockbox"


def test_assert_no_lockbox_catches_leak():
    bad = pd.DatetimeIndex([pd.Timestamp("2023-08-01", tz="UTC")])
    with pytest.raises(AssertionError):
        S.assert_no_lockbox(bad)
    ok = pd.DatetimeIndex([pd.Timestamp("2022-01-01", tz="UTC")])
    S.assert_no_lockbox(ok)  # no raise
