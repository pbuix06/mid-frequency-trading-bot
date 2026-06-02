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
) -> object:
    """
    Roll alpha.compute_signal() bar-by-bar, simulate with vectorbt.

    Signals are computed on closed-bar data. Target weights are applied on the
    next bar's open, so the same bar is never used for both signal and fill.
    """
    try:
        import vectorbt as vbt
        from vectorbt.portfolio.enums import SizeType
    except ImportError as e:
        raise ImportError("pip install vectorbt") from e

    close = data["close"]
    open_ = data["open"]
    n = len(data)
    lookback = alpha.lookback

    raw_weights = np.zeros(n)
    for i in range(lookback, n):
        window = data.iloc[i - lookback : i + 1]
        sig = alpha.compute_signal(window)
        raw_weights[i] = raw_weights[i - 1] if not sig else sig.get(symbol, 0.0)

    weights = pd.Series(raw_weights, index=data.index, name="target_weight")
    changed = weights.ne(weights.shift()).fillna(True)
    target_orders = weights.where(changed).shift(1)

    pf = vbt.Portfolio.from_orders(
        close,
        size=target_orders,
        size_type=SizeType.TargetPercent,
        direction="both",
        price=open_,
        init_cash=init_cash,
        fees=commission_pct,
        slippage=slippage_pct,
        freq="1D",
    )
    return pf


def get_metrics(portfolio: object) -> dict:
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
    slippage_pct: float = 0.001,
    end_date: pd.Timestamp | None = None,
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
        slippage_pct:   One-way slippage rate applied against the trade direction.

    Returns:
        dict with equity_curve (pd.Series), returns (pd.Series), and
        per-symbol weight history (pd.DataFrame).
    """
    # Enforce lock-box: truncate all series before building the common index
    if end_date is not None:
        multi_data = {sym: df[df.index <= end_date] for sym, df in multi_data.items()}

    # Align all symbols to a common index; forward-fill to handle late-start tickers.
    close_dict = {sym: df["close"] for sym, df in multi_data.items()}
    close_df = pd.DataFrame(close_dict).ffill().dropna(how="all")
    open_dict = {sym: df["open"] for sym, df in multi_data.items()}
    open_df = pd.DataFrame(open_dict).reindex(close_df.index).ffill()

    # Drop symbols that have no valid data at all
    close_df = close_df.dropna(axis=1, how="all")
    symbols = list(close_df.columns)
    open_df = open_df.reindex(columns=symbols)
    n = len(close_df)
    lookback = alpha.lookback

    # Track portfolio state
    cash = float(init_cash)
    positions: dict[str, float] = {s: 0.0 for s in symbols}
    equity_history: list[tuple[pd.Timestamp, float]] = []
    weight_history: list[dict] = []

    for i in range(lookback, n - 1):
        ts = close_df.index[i]
        close_prices = close_df.iloc[i]

        # Current equity — skip NaN prices (asset not yet listed / data gap)
        pos_value = sum(
            positions[s] * float(close_prices[s])
            for s in symbols
            if pd.notna(close_prices[s])
        )
        portfolio_value = cash + pos_value
        equity_history.append((ts, portfolio_value))

        # Rebalance on schedule
        if (i - lookback) % rebalance_freq != 0:
            continue

        window = close_df.iloc[i - lookback: i + 1]
        new_weights = alpha.compute_signal(window)
        if not new_weights:
            continue
        weight_history.append({"ts": ts, **new_weights})

        # Execute on the next bar's open, not the same close used for the signal.
        exec_prices = open_df.iloc[i + 1]
        exec_equity = cash + sum(
            positions[s] * float(exec_prices[s])
            for s in symbols
            if pd.notna(exec_prices[s])
        )

        # Execute rebalance: adjust each symbol toward target weight.
        for sym in symbols:
            market_price = float(exec_prices[sym]) if pd.notna(exec_prices[sym]) else float("nan")
            if not (market_price > 0):  # NaN, zero, or negative — skip
                continue
            target_w = new_weights.get(sym, 0.0)
            target_val = exec_equity * target_w
            current_val = positions[sym] * market_price
            delta = target_val - current_val

            if abs(delta) < 1.0:
                continue

            direction = np.sign(delta)
            fill_price = market_price * (1 + slippage_pct * direction)
            current_val_at_fill = positions[sym] * fill_price
            delta = target_val - current_val_at_fill
            if abs(delta) < 1.0:
                continue

            commission = abs(delta) * commission_pct
            cash -= delta + commission
            positions[sym] += delta / fill_price

    if n > lookback:
        last_prices = close_df.iloc[-1]
        last_value = cash + sum(
            positions[s] * float(last_prices[s])
            for s in symbols
            if pd.notna(last_prices[s])
        )
        equity_history.append((close_df.index[-1], last_value))

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
