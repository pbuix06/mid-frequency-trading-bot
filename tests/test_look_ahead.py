"""
Look-ahead bias unit test.

THIS IS THE MOST IMPORTANT TEST IN THE REPO.
Run it in CI. Never delete it. Never skip it.

A strategy must NEVER use data from bar t+1 when computing a signal at bar t.
We catch this by injecting a sentinel value at t+1 and asserting it doesn't
contaminate the signal at t.

A passing test here saves months of debugging why paper results don't match live.
"""

import numpy as np
import pandas as pd
import pytest

from mft.alphas.sma_crossover import SMACrossover
from mft.data_layer.pit import make_pit_window
from mft.features import momentum

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.997,
            "close": prices,
            "volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )


def _inject_sentinel(data: pd.DataFrame, bar_idx: int, sentinel: float = 1e6) -> pd.DataFrame:
    """Replace close at bar_idx with a sentinel value that is unmissably large."""
    df = data.copy()
    df.iloc[bar_idx, df.columns.get_loc("close")] = sentinel
    df.iloc[bar_idx, df.columns.get_loc("open")] = sentinel
    df.iloc[bar_idx, df.columns.get_loc("high")] = sentinel
    df.iloc[bar_idx, df.columns.get_loc("low")] = sentinel
    return df


# ── PIT window tests ───────────────────────────────────────────────────────────

def test_pit_window_excludes_future_bars():
    """make_pit_window must never include any bar after as_of."""
    data = _make_data(n=100)
    as_of = data.index[50]
    window = make_pit_window(data, as_of, lookback=30)

    assert window.index.max() <= as_of, (
        f"PIT window contains bar {window.index.max()} > as_of {as_of} — look-ahead bias!"
    )


def test_pit_window_length():
    """Window should contain at most lookback+1 bars."""
    data = _make_data(n=100)
    as_of = data.index[60]
    lookback = 30
    window = make_pit_window(data, as_of, lookback=lookback)

    assert len(window) <= lookback + 1


def test_pit_window_last_bar_is_as_of():
    """The last bar in the window must be as_of itself."""
    data = _make_data(n=100)
    as_of = data.index[60]
    window = make_pit_window(data, as_of, lookback=20)

    assert window.index[-1] == as_of


# ── Alpha look-ahead tests ─────────────────────────────────────────────────────

def test_sma_signal_unaffected_by_future_sentinel():
    """
    Core look-ahead test.

    We place a sentinel (price = 1,000,000) at bar t+1.
    The SMA signal computed at bar t must not be affected by it.
    If it is, compute_signal() has look-ahead bias.
    """
    alpha = SMACrossover(symbol="TEST", fast=10, slow=30)
    data_clean = _make_data(n=200)
    t_idx = 100  # bar t

    as_of = data_clean.index[t_idx]
    window_clean = make_pit_window(data_clean, as_of, lookback=alpha.lookback)
    signal_clean = alpha.compute_signal(window_clean)

    # Inject sentinel at t+1
    data_sentinel = _inject_sentinel(data_clean, bar_idx=t_idx + 1, sentinel=1e6)
    window_sentinel = make_pit_window(data_sentinel, as_of, lookback=alpha.lookback)
    signal_sentinel = alpha.compute_signal(window_sentinel)

    assert signal_clean == signal_sentinel, (
        f"Signal changed when sentinel was placed at t+1: "
        f"clean={signal_clean}, sentinel={signal_sentinel}. "
        "Look-ahead bias detected!"
    )


def test_sma_signal_affected_by_current_bar_sentinel():
    """
    Sanity check: signal SHOULD change if the sentinel is at bar t itself.
    This confirms the look-ahead test above would catch real contamination.
    """
    alpha = SMACrossover(symbol="TEST", fast=10, slow=30)
    data_clean = _make_data(n=200)
    t_idx = 100

    as_of = data_clean.index[t_idx]
    window_clean = make_pit_window(data_clean, as_of, lookback=alpha.lookback)
    signal_clean = alpha.compute_signal(window_clean)

    # Inject sentinel AT bar t (should change the signal)
    data_sentinel = _inject_sentinel(data_clean, bar_idx=t_idx, sentinel=1e6)
    window_sentinel = make_pit_window(data_sentinel, as_of, lookback=alpha.lookback)
    signal_sentinel = alpha.compute_signal(window_sentinel)

    # Signal will be 1.0 (long) because fast_ma >> slow_ma with sentinel close
    assert signal_clean != signal_sentinel, (
        "Sentinel at bar t did not affect the signal — test harness may be broken."
    )


def test_sma_signal_bounds():
    """SMA signal must be in {0.0, 1.0} (long-only alpha)."""
    alpha = SMACrossover(symbol="TEST", fast=10, slow=30)
    data = _make_data(n=200)

    for i in range(alpha.lookback, len(data)):
        window = make_pit_window(data, data.index[i], lookback=alpha.lookback)
        sig = alpha.compute_signal(window)
        for sym, w in sig.items():
            assert w in (0.0, 1.0), f"Signal {w} not in {{0.0, 1.0}} at bar {i}"


# ── Feature pure-function tests ────────────────────────────────────────────────

def test_log_returns_is_deterministic():
    """Same input must always produce same output."""
    data = _make_data(n=100)
    r1 = momentum.log_returns(data["close"])
    r2 = momentum.log_returns(data["close"])
    pd.testing.assert_series_equal(r1, r2)


def test_features_do_not_use_future_data():
    """
    Features computed on a prefix window must match the same-length prefix
    of features computed on a longer window. If not, the feature is using
    future data (e.g., via centering or normalization on the full series).
    """
    data = _make_data(n=100)
    cutoff = 60

    # Compute on truncated window (as seen in production at bar 60)
    window_60 = data.iloc[:cutoff]
    ret_60 = momentum.log_returns(window_60["close"])
    z_60 = momentum.z_score(window_60["close"], window=20)

    # Compute on full window (future data included)
    ret_full = momentum.log_returns(data["close"])
    z_full = momentum.z_score(data["close"], window=20)

    # The first 60 values must be identical
    pd.testing.assert_series_equal(ret_60, ret_full.iloc[:cutoff], check_names=False)

    # z_score uses rolling stats — the first cutoff values must match
    pd.testing.assert_series_equal(
        z_60.dropna(), z_full.iloc[:cutoff].dropna(), check_names=False
    )


# ── Parity test skeleton (Gate 0) ─────────────────────────────────────────────

def test_event_harness_produces_reasonable_equity(synthetic_ohlcv):
    """
    Event harness must produce a non-trivial equity curve.
    Full vectorbt parity test requires vectorbt installed — see scripts/parity_check.py.
    """
    from mft.backtest.event_harness import equity_to_returns, run_event_driven

    alpha = SMACrossover(symbol="SPY", fast=10, slow=30)
    state = run_event_driven(alpha, synthetic_ohlcv, symbol="SPY", init_cash=100_000)

    assert len(state.equity_curve) > 0, "No equity curve generated"
    assert state.equity_curve[0][1] == pytest.approx(100_000, rel=0.01)

    returns = equity_to_returns(state)
    assert len(returns) > 0
    # Equity should be within 50% of initial (sanity, not a performance test)
    final_equity = state.equity_curve[-1][1]
    assert 50_000 < final_equity < 500_000, f"Equity {final_equity} out of sanity range"
