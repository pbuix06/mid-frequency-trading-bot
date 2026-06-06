"""
Pre-trade risk gate — the last check before a (paper) position is opened.

A signal only becomes a hypothetical trade if EVERY limit passes. The gate is pure: it
reads a `FeatureSnapshot`, a `PortfolioView` (the paper book's current state), and the
clock, and returns a `RiskDecision(allowed, reasons)`. It never mutates anything and never
places an order — it only permits or blocks the paper-decision step.

Checks (all must pass):
  1. data not stale          (snapshot age <= stale_seconds)
  2. spread below threshold  (if top-of-book is known)
  3. max open positions not exceeded
  4. per-symbol notional cap not exceeded
  5. daily loss limit not hit (halt new entries)
  6. volatility not extreme  (unless explicitly allowed)
  7. no duplicate trade if already in a position for that symbol
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mft.live_monitor.feature_stream import FeatureSnapshot
from mft.live_monitor.signal_engine import Signal


@dataclass
class RiskLimits:
    max_positions: int = 3
    max_symbol_notional: float = 1_000.0      # quote-ccy cap per symbol
    max_daily_loss: float = 200.0             # quote-ccy; new entries halt at -this
    max_spread_bps: float = 10.0
    stale_seconds: float = 120.0
    extreme_vol_24h: float = 0.05             # 1m-return std deemed "extreme"
    allow_extreme_vol: bool = False


@dataclass
class PortfolioView:
    """Minimal read-only view of the paper book the gate needs (no methods, no orders)."""
    open_symbols: set = field(default_factory=set)
    n_positions: int = 0
    symbol_notional: dict = field(default_factory=dict)
    realized_pnl_today: float = 0.0


@dataclass
class RiskDecision:
    allowed: bool
    reasons: list[str]


class RiskGate:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def check(self, signal: Signal, snap: FeatureSnapshot, pf: PortfolioView,
              now: pd.Timestamp, notional: float) -> RiskDecision:
        if not signal.is_trade:
            return RiskDecision(False, ["no actionable signal (flat)"])

        L = self.limits
        reasons: list[str] = []
        already = signal.symbol in pf.open_symbols

        age = (pd.Timestamp(now) - pd.Timestamp(snap.ts)).total_seconds()
        if age > L.stale_seconds:
            reasons.append(f"stale data: snapshot {age:.0f}s old > {L.stale_seconds:.0f}s")

        if snap.spread_bps is not None and snap.spread_bps > L.max_spread_bps:
            reasons.append(f"spread {snap.spread_bps:.1f}bps > {L.max_spread_bps:.1f}bps")

        if not already and pf.n_positions >= L.max_positions:
            reasons.append(f"max positions reached ({pf.n_positions} >= {L.max_positions})")

        projected = pf.symbol_notional.get(signal.symbol, 0.0) + notional
        if projected > L.max_symbol_notional + 1e-9:
            reasons.append(f"symbol exposure {projected:.0f} > cap {L.max_symbol_notional:.0f}")

        if pf.realized_pnl_today <= -abs(L.max_daily_loss):
            reasons.append(f"daily loss limit hit ({pf.realized_pnl_today:+.2f} <= -{abs(L.max_daily_loss):.2f})")

        if (not L.allow_extreme_vol) and snap.vol_24h == snap.vol_24h and snap.vol_24h > L.extreme_vol_24h:
            reasons.append(f"extreme vol {snap.vol_24h:.4f} > {L.extreme_vol_24h:.4f} (not allowed)")

        if already:
            reasons.append(f"duplicate: already holding {signal.symbol}")

        return RiskDecision(len(reasons) == 0, reasons if reasons else ["ok"])
