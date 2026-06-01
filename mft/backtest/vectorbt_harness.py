"""
vectorbt research harness — fast signal iteration for parameter sweeps.

This harness calls the SAME alpha.compute_signal() as the event harness.
That is the one-code-path guarantee. Equity curves must reconcile (Gate 0).

Usage:
    alpha = SMACrossover("SPY", fast=20, slow=50)
    pf = run_research(alpha, data, symbol="SPY")
    metrics = get_metrics(pf)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.base import AlphaBase


def run_research(
    alpha: AlphaBase,
    data: pd.DataFrame,
    symbol: str,
    *,
    init_cash: float = 100_000,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.001,
) -> "vbt.Portfolio":
    """
    Roll alpha.compute_signal() bar-by-bar, simulate with vectorbt.

    Signals are computed on closed-bar data; orders execute on next-bar open
    (via vectorbt's slippage parameter approximation).
    """
    try:
        import vectorbt as vbt
    except ImportError as e:
        raise ImportError("pip install vectorbt") from e

    close = data["close"]
    n = len(data)
    lookback = alpha.lookback

    raw_weights = np.full(n, np.nan)
    for i in range(lookback, n):
        window = data.iloc[i - lookback : i + 1]
        sig = alpha.compute_signal(window)
        raw_weights[i] = sig.get(symbol, 0.0)

    weights = pd.Series(raw_weights, index=data.index, name="signal")
    entries = weights > 0
    exits = weights == 0

    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=commission_pct,
        slippage=slippage_pct,
        freq="1D",
    )
    return pf


def get_metrics(portfolio: "vbt.Portfolio") -> dict:
    """Extract the standard metric vector from a vectorbt Portfolio."""
    stats = portfolio.stats()
    return {
        "total_return": float(stats.get("Total Return [%]", np.nan)) / 100,
        "sharpe": float(stats.get("Sharpe Ratio", np.nan)),
        "max_dd": float(stats.get("Max Drawdown [%]", np.nan)) / 100,
        "calmar": float(stats.get("Calmar Ratio", np.nan)),
        "n_trades": int(stats.get("Total Trades", 0)),
    }
