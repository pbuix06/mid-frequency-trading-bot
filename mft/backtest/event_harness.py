"""
Event-driven validation harness — bar-by-bar simulation.

Purpose: prove parity with the vectorbt research harness.
The same alpha.compute_signal() is called here. If equity curves match,
you have confidence that vectorbt's vectorized semantics aren't introducing
artifacts that would break live.

Phase 0: lightweight Python simulator.
Phase 2: swap SimulatedBroker for NautilusTrader BacktestEngine
         (see mft/backtest/nautilus_adapter.py — not yet implemented).

Usage:
    alpha = SMACrossover("SPY", fast=20, slow=50)
    state = run_event_driven(alpha, data, symbol="SPY")
    returns = equity_to_returns(state)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mft.alphas.base import AlphaBase


@dataclass
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    qty: float        # positive = buy, negative = sell
    price: float
    commission: float


@dataclass
class SimState:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    is_halted: bool = False


def run_event_driven(
    alpha: AlphaBase,
    data: pd.DataFrame,
    symbol: str,
    *,
    init_cash: float = 100_000,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.001,
) -> SimState:
    """
    Bar-by-bar event loop. Signals on bar close; orders fill on next bar open.

    This is the canonical validation engine. Gate 0 requires its equity curve
    to reconcile with run_research() on the same alpha and data.
    """
    state = SimState(cash=init_cash)
    n = len(data)
    lookback = alpha.lookback

    prev_signal: float | None = None

    for i in range(lookback, n - 1):
        bar = data.iloc[i]
        next_bar = data.iloc[i + 1]

        # --- Compute signal on current closed bar (PIT) ---
        window = data.iloc[i - lookback : i + 1]
        signals = alpha.compute_signal(window)
        target_weight = signals.get(symbol, 0.0)

        # --- Mark equity at current close ---
        current_pos = state.positions.get(symbol, 0.0)
        equity = state.cash + current_pos * bar["close"]
        state.equity_curve.append((bar.name, equity))

        # --- Only rebalance when signal changes (matches vectorbt from_signals semantics) ---
        if target_weight == prev_signal:
            continue
        prev_signal = target_weight

        # --- Execute on next bar open with slippage ---
        direction = np.sign(target_weight - (1.0 if current_pos > 0 else 0.0))
        exec_price = next_bar["open"] * (1 + slippage_pct * direction)
        target_value = equity * target_weight
        current_value = current_pos * exec_price
        delta_value = target_value - current_value

        if abs(delta_value) < 1.0:
            continue

        qty = delta_value / exec_price
        commission = abs(delta_value) * commission_pct

        state.cash -= delta_value + commission
        state.positions[symbol] = current_pos + qty
        state.fills.append(
            Fill(
                timestamp=next_bar.name,
                symbol=symbol,
                qty=qty,
                price=exec_price,
                commission=commission,
            )
        )

    # Final bar mark-to-market
    if len(data) > lookback:
        last_bar = data.iloc[-1]
        last_pos = state.positions.get(symbol, 0.0)
        state.equity_curve.append((last_bar.name, state.cash + last_pos * last_bar["close"]))

    return state


def equity_to_returns(state: SimState) -> pd.Series:
    """Convert SimState equity curve to daily returns Series."""
    if not state.equity_curve:
        return pd.Series(dtype=float, name="returns")
    timestamps, values = zip(*state.equity_curve)
    equity = pd.Series(list(values), index=list(timestamps), name="equity")
    return equity.pct_change().dropna()
