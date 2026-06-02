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


def run_research_xs(
    alpha: AlphaBase,
    multi_data: dict[str, pd.DataFrame],
    *,
    rebalance_freq: int = 21,
    init_cash: float = 100_000,
    commission_pct: float = 0.001,
    end_date: "pd.Timestamp | None" = None,
) -> dict:
    """
    Cross-sectional research harness for multi-asset alphas (e.g. XSMomentum).

    Args:
        alpha:          AlphaBase with compute_signal(window) where window has
                        columns = symbol names, rows = close prices.
        multi_data:     dict mapping symbol → OHLCV DataFrame (same date range).
        rebalance_freq: Recompute signals every N bars (default 21 ≈ monthly).
        init_cash:      Starting portfolio value.
        commission_pct: One-way commission rate.

    Returns:
        dict with equity_curve (pd.Series), returns (pd.Series), and
        per-symbol weight history (pd.DataFrame).
    """
    # Enforce lock-box: truncate all series before building the common index
    if end_date is not None:
        multi_data = {sym: df[df.index <= end_date] for sym, df in multi_data.items()}

    # Align all symbols to a common index; forward-fill to handle late-start tickers
    close_dict = {sym: df["close"] for sym, df in multi_data.items()}
    close_df = pd.DataFrame(close_dict).ffill().dropna(how="all")
    # Drop symbols that have no valid data at all
    close_df = close_df.dropna(axis=1, how="all")
    symbols = list(close_df.columns)
    n = len(close_df)
    lookback = alpha.lookback

    # Track portfolio state
    cash = float(init_cash)
    positions: dict[str, float] = {s: 0.0 for s in symbols}
    equity_history: list[tuple[pd.Timestamp, float]] = []
    weight_history: list[dict] = []

    prev_weights: dict[str, float] = {s: 0.0 for s in symbols}

    for i in range(lookback, n):
        ts = close_df.index[i]
        prices = close_df.iloc[i]

        # Current equity — skip NaN prices (asset not yet listed / data gap)
        pos_value = sum(
            positions[s] * float(prices[s])
            for s in symbols
            if pd.notna(prices[s])
        )
        portfolio_value = cash + pos_value
        equity_history.append((ts, portfolio_value))

        # Rebalance on schedule
        if (i - lookback) % rebalance_freq != 0:
            continue

        window = close_df.iloc[i - lookback: i + 1]
        new_weights = alpha.compute_signal(window)
        weight_history.append({"ts": ts, **new_weights})

        # Execute rebalance: adjust each symbol toward target weight
        for sym in symbols:
            p = float(prices[sym]) if pd.notna(prices[sym]) else float("nan")
            if not (p > 0):          # NaN, zero, or negative — skip
                continue
            target_w = new_weights.get(sym, 0.0)
            target_val = portfolio_value * target_w
            current_val = positions[sym] * p
            delta = target_val - current_val

            if abs(delta) < 1.0:
                continue

            commission = abs(delta) * commission_pct
            cash -= delta + commission
            positions[sym] = target_val / p

        prev_weights = new_weights

    ts_vals, eq_vals = zip(*equity_history) if equity_history else ([], [])
    equity = pd.Series(list(eq_vals), index=list(ts_vals), name="equity")
    returns = equity.pct_change(fill_method=None).dropna()

    weight_df = pd.DataFrame(weight_history).set_index("ts") if weight_history else pd.DataFrame()

    return {
        "equity": equity,
        "returns": returns,
        "weights": weight_df,
        "final_equity": equity.iloc[-1] if len(equity) else init_cash,
    }
