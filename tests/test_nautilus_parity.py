"""
Phase 2 parity test — NautilusTrader vs event harness.

Runs the same SMA alpha through both engines on synthetic data and asserts
the results reconcile within tolerance. Uses synthetic data (no disk I/O)
so this can run in CI without requiring the data/pit/ directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.alphas.sma_crossover import SMACrossover
from mft.backtest.event_harness import equity_to_returns, run_event_driven
from mft.backtest.nautilus_harness import run_nautilus
from mft.validation.metrics import full_metrics

# NautilusTrader enforces integer share quantities; event_harness allows
# fractional shares (qty = delta_value / price). On short synthetic series
# this creates meaningful equity divergence even when signals are identical.
# On real multi-year data the gap is < 0.05. Tolerance set to 0.20.
SHARPE_TOL = 0.20
ROUNDTRIP_TOL = 2


def _make_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def parity_results():
    data = _make_data()
    alpha = SMACrossover(symbol="TEST", fast=20, slow=50)

    nt_result = run_nautilus(alpha, data, symbol="TEST")
    # Zero cost on both sides — isolates execution semantics from cost model
    ev_state = run_event_driven(alpha, data, symbol="TEST", slippage_pct=0.0, commission_pct=0.0)

    nt_metrics = full_metrics(nt_result.returns)
    ev_metrics = full_metrics(equity_to_returns(ev_state))

    return {
        "nt": nt_metrics,
        "ev": ev_metrics,
        "nt_roundtrips": nt_result.n_fills / 2,
        "ev_roundtrips": len(ev_state.fills) / 2,
    }


def test_sharpe_within_tolerance(parity_results):
    diff = abs(parity_results["nt"]["sharpe"] - parity_results["ev"]["sharpe"])
    assert diff < SHARPE_TOL, f"Sharpe diff {diff:.4f} exceeds tolerance {SHARPE_TOL}"


def test_roundtrips_within_tolerance(parity_results):
    diff = abs(parity_results["nt_roundtrips"] - parity_results["ev_roundtrips"])
    assert diff <= ROUNDTRIP_TOL, f"Round-trip diff {diff} exceeds tolerance {ROUNDTRIP_TOL}"


def test_nautilus_produces_nonzero_equity(parity_results):
    assert parity_results["nt"]["sharpe"] != 0.0


def test_nautilus_max_drawdown_is_negative(parity_results):
    assert parity_results["nt"]["max_drawdown"] < 0
