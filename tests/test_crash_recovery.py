"""
Gate 2: Crash recovery tests.

Verifies that trading state can be saved to disk atomically and restored,
and that resuming from a checkpoint produces the same final equity as
an uninterrupted run.

Assumption: crash occurs at a clean bar boundary (after all fills for that
bar have been processed). Mid-fill crash recovery requires a pending-orders
journal, which is a Phase 6 concern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.sma_crossover import SMACrossover
from mft.backtest.event_harness import run_event_driven
from mft.execution.state import StateStore, TradingState


def _make_data(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open":   prices * rng.uniform(0.998, 1.000, n),
            "high":   prices * rng.uniform(1.001, 1.005, n),
            "low":    prices * rng.uniform(0.995, 0.999, n),
            "close":  prices,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


def test_state_roundtrip(tmp_path):
    """TradingState saves and loads without any data loss."""
    store = StateStore()
    path = tmp_path / "state.json"
    orig = TradingState(
        cash=87_500.25,
        positions={"SPY": 120.0, "AAPL": -50.0},
        last_signals={"SPY": 1.0, "AAPL": -1.0},
        last_bar_ts=pd.Timestamp("2023-06-01", tz="UTC"),
    )
    store.save(orig, path)
    loaded = store.load(path)

    assert loaded.cash == orig.cash
    assert loaded.positions == orig.positions
    assert loaded.last_signals == orig.last_signals
    assert loaded.last_bar_ts == orig.last_bar_ts


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    """After a successful save, the .tmp staging file is cleaned up."""
    store = StateStore()
    path = tmp_path / "state.json"

    assert not store.exists(path)

    store.save(TradingState(cash=100_000.0), path)

    assert store.exists(path)
    assert not path.with_suffix(".tmp").exists()


def test_crash_recovery_same_final_equity(tmp_path):
    """
    Resuming from a mid-run checkpoint produces the same final equity as
    running uninterrupted. Uses zero costs so cash arithmetic is exact.
    """
    data = _make_data()
    alpha = SMACrossover(symbol="SPY", fast=20, slow=50)
    kwargs = {"slippage_pct": 0.0, "commission_pct": 0.0}

    # Reference: uninterrupted full run
    full = run_event_driven(alpha, data, symbol="SPY", **kwargs)
    full_final = full.equity_curve[-1][1]

    # Simulated crash at bar 200
    crash_idx = 200
    first_half = run_event_driven(
        alpha, data.iloc[: crash_idx + 1], symbol="SPY", **kwargs
    )
    assert first_half.last_bar_ts is not None

    # Save state at crash point
    store = StateStore()
    state_path = tmp_path / "state.json"
    saved = first_half.to_trading_state()
    store.save(saved, state_path)

    # Recovery: load state, run second half on full dataset
    restored = store.load(state_path)
    resumed = run_event_driven(
        alpha, data, symbol="SPY", initial_state=restored, **kwargs
    )
    resumed_final = resumed.equity_curve[-1][1]

    assert abs(full_final - resumed_final) < 1e-6, (
        f"Equity diverged after crash recovery: "
        f"full={full_final:.6f}, resumed={resumed_final:.6f}"
    )


def test_simstate_exposes_last_bar_ts(tmp_path):
    """SimState.last_bar_ts is set after each processed bar."""
    data = _make_data(n=100)
    alpha = SMACrossover(symbol="SPY", fast=20, slow=50)
    state = run_event_driven(alpha, data, symbol="SPY")

    assert state.last_bar_ts is not None
    assert state.last_bar_ts == data.index[-1]
