"""
Risk limits and kill switches.

Wire these in BEFORE the first live dollar. Test each in paper.
Many systems neglect real-time post-trade surveillance and let slow losses
compound until catastrophic. Don't be one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class RiskLimits:
    """Risk control configuration. Set once at startup; do not change live."""

    max_position_pct: float = 0.25       # Single-position cap (≤ 25% equity)
    max_gross_exposure: float = 1.0      # Sum of |weights|
    per_trade_risk_pct: float = 0.02     # Risk per trade (1–2% of equity)
    daily_loss_limit_pct: float = 0.02   # Halt if daily loss exceeds this
    weekly_loss_limit_pct: float = 0.05  # Halt if weekly loss exceeds this
    max_drawdown_pct: float = 0.10       # Halt if drawdown exceeds this
    pnl_velocity_window_min: int = 15    # Minutes for P&L velocity check
    pnl_velocity_limit_pct: float = 0.01 # Max loss in velocity window


@dataclass
class RiskState:
    """Mutable risk state. Survives restarts (serialize to disk in production)."""

    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    weekly_start_equity: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""
    halt_timestamp: Optional[pd.Timestamp] = None


class RiskManager:
    """
    Pre-trade and post-trade risk enforcement.

    Pre-trade: enforces position limits before orders are sent.
    Post-trade: checks loss limits after each bar; fires kill switch if breached.

    Kill switch: once halted, manual_resume() required. By design.
    """

    def __init__(self, limits: RiskLimits, init_equity: float):
        self.limits = limits
        self.state = RiskState(
            peak_equity=init_equity,
            daily_start_equity=init_equity,
            weekly_start_equity=init_equity,
        )

    def check_pre_trade(
        self,
        target_weights: dict[str, float],
        current_equity: float,
    ) -> tuple[dict[str, float], list[str]]:
        """
        Enforce position limits before order generation.

        Returns:
            (approved_weights, list_of_violations)
            If halted, returns ({}, [halt message]).
        """
        if self.state.is_halted:
            return {}, [f"HALTED: {self.state.halt_reason}"]

        violations: list[str] = []
        approved: dict[str, float] = {}

        for symbol, w in target_weights.items():
            if abs(w) > self.limits.max_position_pct:
                clipped = max(-self.limits.max_position_pct, min(self.limits.max_position_pct, w))
                violations.append(f"{symbol}: clipped {w:.3f} → {clipped:.3f}")
                w = clipped
            approved[symbol] = w

        gross = sum(abs(w) for w in approved.values())
        if gross > self.limits.max_gross_exposure:
            scale = self.limits.max_gross_exposure / gross
            approved = {s: w * scale for s, w in approved.items()}
            violations.append(f"Gross {gross:.2f} > {self.limits.max_gross_exposure:.2f}; scaled by {scale:.3f}")

        return approved, violations

    def update_post_trade(
        self,
        current_equity: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> list[str]:
        """
        Check loss limits after each bar. Trigger kill switch if breached.

        Returns list of triggered alerts (empty = all clear).
        """
        alerts: list[str] = []

        # Update high-water mark
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity

        # Drawdown
        if self.state.peak_equity > 0:
            dd = (current_equity - self.state.peak_equity) / self.state.peak_equity
            if dd < -self.limits.max_drawdown_pct:
                self._halt(f"Max drawdown {dd:.2%} exceeded {self.limits.max_drawdown_pct:.2%}", timestamp)
                alerts.append(f"KILL SWITCH: drawdown {dd:.2%}")

        # Daily loss
        if self.state.daily_start_equity > 0:
            daily_pnl = (current_equity - self.state.daily_start_equity) / self.state.daily_start_equity
            if daily_pnl < -self.limits.daily_loss_limit_pct:
                self._halt(f"Daily loss {daily_pnl:.2%}", timestamp)
                alerts.append(f"KILL SWITCH: daily loss {daily_pnl:.2%}")

        return alerts

    def reset_daily(self, current_equity: float) -> None:
        """Call at the start of each trading day."""
        self.state.daily_start_equity = current_equity

    def manual_resume(self) -> None:
        """Manual intervention required to resume. Intentional friction."""
        self.state.is_halted = False
        self.state.halt_reason = ""
        self.state.halt_timestamp = None

    def _halt(self, reason: str, timestamp: Optional[pd.Timestamp]) -> None:
        if not self.state.is_halted:
            self.state.is_halted = True
            self.state.halt_reason = reason
            self.state.halt_timestamp = timestamp
