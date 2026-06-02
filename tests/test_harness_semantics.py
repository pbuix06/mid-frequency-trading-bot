"""Backtest harness execution semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.base import AlphaBase
from mft.backtest.event_harness import run_event_driven


class ScheduledWeightAlpha(AlphaBase):
    """Deterministic target-weight schedule for harness tests."""

    def __init__(self, symbol: str, weights: dict[pd.Timestamp, float]):
        self.symbol = symbol
        self.weights = weights

    @property
    def lookback(self) -> int:
        return 1

    def compute_signal(self, window: pd.DataFrame) -> dict[str, float]:
        ts = window.index[-1]
        return {self.symbol: self.weights.get(ts, 0.0)}


def test_event_harness_rebalances_fractional_target_delta():
    dates = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    prices = np.full(len(dates), 100.0)
    data = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(len(dates)) * 1_000_000,
        },
        index=dates,
    )
    alpha = ScheduledWeightAlpha("SPY", {dates[1]: 0.50, dates[2]: 0.25})

    state = run_event_driven(
        alpha,
        data,
        symbol="SPY",
        init_cash=100_000,
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(state.fills) == 3
    assert state.fills[0].qty == 500.0
    assert state.fills[1].qty == -250.0
    assert state.fills[2].qty == -250.0
    assert state.positions["SPY"] == 0.0
