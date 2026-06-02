"""
Event-driven validation harness — bar-by-bar simulation.

Purpose: prove parity with the vectorbt research harness.
The same alpha.compute_signal() is called here. If equity curves match,
you have confidence that vectorbt's vectorized semantics aren't introducing
artifacts that would break live.

Supports crash recovery via the `initial_state` parameter: pass a
TradingState saved by StateStore and the run resumes from that checkpoint
rather than restarting from scratch.

Usage:
    alpha = SMACrossover("SPY", fast=20, slow=50)
    state = run_event_driven(alpha, data, symbol="SPY")
    returns = equity_to_returns(state)

    # Resume after crash:
    checkpoint = store.load(path)
    state = run_event_driven(alpha, data, symbol="SPY", initial_state=checkpoint)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mft.alphas.base import AlphaBase

if TYPE_CHECKING:
    from mft.execution.state import TradingState


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
    last_signals: dict[str, float] = field(default_factory=dict)
    last_bar_ts: pd.Timestamp | None = None

    def to_trading_state(self) -> "TradingState":
        """Snapshot current state for crash recovery."""
        from mft.execution.state import TradingState
        return TradingState(
            cash=self.cash,
            positions=dict(self.positions),
            last_signals=dict(self.last_signals),
            last_bar_ts=self.last_bar_ts,
        )


def run_event_driven(
    alpha: AlphaBase,
    data: pd.DataFrame,
    symbol: str,
    *,
    init_cash: float = 100_000,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.001,
    initial_state: "TradingState | None" = None,
    end_date: "pd.Timestamp | None" = None,
) -> SimState:
    """
    Bar-by-bar event loop. Signals on bar close; orders fill on next bar open.

    This is the canonical validation engine. Gate 0 requires its equity curve
    to reconcile with run_research() on the same alpha and data.

    Args:
        end_date:      If set, data is truncated to bars <= end_date before
                       running. Pass LOCKBOX_CUTOFF for all research runs to
                       keep lock-box data untouched.
        initial_state: If provided, restores cash/positions/last_signals and
                       resumes processing from after initial_state.last_bar_ts.
                       The full data DataFrame must still be passed (the lookback
                       window before the resume point is needed for signal computation).
    """
    # Enforce lock-box: never use data beyond end_date in any research run
    if end_date is not None:
        data = data[data.index <= end_date]

    state = SimState(cash=init_cash)
    n = len(data)
    lookback = alpha.lookback

    # Determine start index and restore state for crash recovery
    start_i = lookback
    prev_signal: float | None = None

    if initial_state is not None:
        state.cash = initial_state.cash
        state.positions = dict(initial_state.positions)
        state.last_signals = dict(initial_state.last_signals)
        prev_signal = initial_state.last_signals.get(symbol)

        if initial_state.last_bar_ts is not None:
            bars_after = data.index > initial_state.last_bar_ts
            if not bars_after.any():
                return state  # nothing left to process
            first_after = int(bars_after.argmax())
            start_i = max(lookback, first_after)

    for i in range(start_i, n - 1):
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

        # --- Track state for crash recovery ---
        state.last_signals[symbol] = target_weight
        state.last_bar_ts = bar.name

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
    if len(data) > start_i:
        last_bar = data.iloc[-1]
        last_pos = state.positions.get(symbol, 0.0)
        state.equity_curve.append((last_bar.name, state.cash + last_pos * last_bar["close"]))
        state.last_bar_ts = last_bar.name

    return state


def equity_to_returns(state: SimState) -> pd.Series:
    """Convert SimState equity curve to daily returns Series."""
    if not state.equity_curve:
        return pd.Series(dtype=float, name="returns")
    timestamps, values = zip(*state.equity_curve)
    equity = pd.Series(list(values), index=list(timestamps), name="equity")
    return equity.pct_change().dropna()
