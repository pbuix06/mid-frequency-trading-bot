"""Tests for the pre-Phase-4 research diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.validation.diagnostics import (
    cost_stress_curve,
    rolling_sharpe,
    rolling_sharpe_summary,
    turnover_from_weights,
)


def _returns(n=600, mu=0.0004, sd=0.01, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(rng.normal(mu, sd, n), index=idx)


def test_rolling_sharpe_length_and_positivity():
    r = _returns(mu=0.001, sd=0.01)  # strong positive drift
    rs = rolling_sharpe(r, window=252)
    assert len(rs) == len(r) - 252 + 1 - 0  # dropna of exactly window-1 leading NaNs
    summary = rolling_sharpe_summary(r, window=252)
    assert summary["frac_positive"] > 0.9  # positive-drift series mostly positive


def test_rolling_sharpe_short_series_returns_empty():
    r = _returns(n=100)
    assert rolling_sharpe(r, window=252).empty


def test_turnover_full_swap_is_two():
    """Swapping a 100% long book to 100% short = two-way turnover of 2.0."""
    idx = pd.date_range("2015-01-01", periods=2, freq="MS", tz="UTC")
    w = pd.DataFrame({"A": [1.0, -1.0]}, index=idx)
    t = turnover_from_weights(w)
    assert abs(t["per_rebalance_mean"] - 2.0) < 1e-9
    assert t["n_transitions"] == 1  # 2 snapshots = 1 transition


def test_turnover_no_change_is_zero():
    idx = pd.date_range("2015-01-01", periods=3, freq="MS", tz="UTC")
    w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]}, index=idx)
    t = turnover_from_weights(w)
    assert abs(t["per_rebalance_mean"]) < 1e-9


def test_cost_stress_curve_monotonic_decay():
    """Higher costs must not increase Sharpe; a positive edge degrades."""
    base = _returns(mu=0.0008, sd=0.01, seed=3)

    def run_fn(commission_pct, slippage_pct):
        # crude cost model: subtract round-trip cost amortized per bar
        drag = (commission_pct + slippage_pct) * 0.5
        return base - drag

    rows = cost_stress_curve(run_fn, multipliers=(1.0, 2.0, 3.0))
    sharpes = [r["sharpe"] for r in rows]
    assert sharpes[0] >= sharpes[1] >= sharpes[2]
    assert rows[0]["cost_pct"] == 0.001
    assert rows[1]["cost_pct"] == 0.002
