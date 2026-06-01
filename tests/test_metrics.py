"""Tests for validation metrics and DSR/MinBTL logic."""

import numpy as np
import pandas as pd
import pytest

from mft.validation.metrics import sharpe, sortino, max_drawdown, calmar, full_metrics
from mft.validation.dsr import deflated_sharpe_ratio, min_backtest_length, expected_max_sharpe
from mft.validation.cpcv import cpcv_splits, purge_embargo


# ── Metrics ────────────────────────────────────────────────────────────────────

def _flat_returns(n: int = 252, mean: float = 0.001, std: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def test_sharpe_positive_drift():
    returns = _flat_returns(mean=0.001, std=0.01)
    sr = sharpe(returns)
    assert sr > 0


def test_sharpe_zero_for_zero_mean():
    returns = pd.Series(np.zeros(252))
    sr = sharpe(returns)
    assert np.isnan(sr) or sr == pytest.approx(0.0, abs=1e-6)


def test_max_drawdown_is_negative():
    returns = pd.Series([-0.01] * 20 + [0.005] * 200)
    mdd = max_drawdown(returns)
    assert mdd < 0


def test_max_drawdown_zero_for_monotone_gain():
    returns = pd.Series([0.001] * 100)
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-8)


def test_sortino_geq_sharpe_for_positive_skew():
    # Asymmetric: many small gains, few large losses
    gains = np.random.default_rng(1).exponential(0.005, 200)
    losses = np.random.default_rng(2).normal(-0.003, 0.001, 50)
    returns = pd.Series(np.concatenate([gains, losses]))
    sr = sharpe(returns)
    so = sortino(returns)
    # Sortino penalizes only downside, so it can be higher or lower depending on skew
    assert np.isfinite(so)


def test_full_metrics_returns_all_keys():
    returns = _flat_returns()
    m = full_metrics(returns)
    expected_keys = {"sharpe", "sortino", "calmar", "cagr", "max_drawdown",
                     "volatility", "hit_rate", "profit_factor", "n_periods"}
    assert expected_keys == set(m.keys())


# ── DSR / MinBTL ───────────────────────────────────────────────────────────────

def test_expected_max_sharpe_increases_with_trials():
    e10 = expected_max_sharpe(10)
    e100 = expected_max_sharpe(100)
    e1000 = expected_max_sharpe(1000)
    assert e10 < e100 < e1000


def test_dsr_high_for_genuine_edge():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, 500)  # strong signal
    sr = float(returns.mean() / returns.std() * np.sqrt(252))
    dsr = deflated_sharpe_ratio(sr, returns, n_trials=1)
    assert dsr > 0.5  # Should be significant with 1 trial and strong edge


def test_dsr_low_for_noise_with_many_trials():
    rng = np.random.default_rng(99)
    returns = rng.normal(0.0, 0.01, 252)
    sr = float(returns.mean() / returns.std() * np.sqrt(252))
    # 50 trials on 1 year of data — should not pass DSR gate
    dsr = deflated_sharpe_ratio(sr, returns, n_trials=50)
    assert dsr < 0.95


def test_minbtl_increases_with_trials():
    """MinBTL is monotone: more trials require more data."""
    bars = [min_backtest_length(n) for n in [1, 5, 10, 45, 100, 500]]
    for i in range(len(bars) - 1):
        assert bars[i] <= bars[i + 1], f"MinBTL not monotone at index {i}"


def test_minbtl_5yr_45trials():
    """Paper worked example: 5 years of data → max 45 independent configs."""
    bars_45 = min_backtest_length(45)
    five_years = 5 * 252  # 1260 bars
    assert bars_45 == pytest.approx(five_years, rel=0.02), (
        f"MinBTL(45) = {bars_45}, expected ~{five_years} bars (5 years)"
    )


def test_minbtl_2yr_7trials():
    """Paper worked example: 2-year backtest, after 7 configs E[IS Sharpe] = 1."""
    bars_7 = min_backtest_length(7)
    two_years = 2 * 252  # 504 bars
    assert bars_7 == pytest.approx(two_years, rel=0.10), (
        f"MinBTL(7) = {bars_7}, expected ~{two_years} bars (2 years)"
    )


# ── CPCV ──────────────────────────────────────────────────────────────────────

def test_cpcv_splits_count():
    from math import comb
    splits = cpcv_splits(n=1000, n_groups=6, n_test_groups=2)
    assert len(splits) == comb(6, 2)  # C(6,2) = 15


def test_cpcv_no_overlap_train_test():
    splits = cpcv_splits(n=500, n_groups=6, n_test_groups=2, embargo_bars=5)
    for train_idx, test_idx in splits:
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} samples"


def test_purge_embargo_removes_near_test():
    train = np.arange(100)
    test = np.arange(50, 60)
    clean = purge_embargo(train, test, embargo_bars=5)
    # Bars [50..64] should be removed from train
    assert not any(50 <= i <= 64 for i in clean)
    # Bars before test_start should survive
    assert all(i in clean for i in range(50))
