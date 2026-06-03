"""
Portfolio — the one object that turns a set of sleeves into a tradeable book.

It composes the Stage-5 pieces:
  inverse_vol_alloc (capital across sleeves)  ->  net_book (one position per
  instrument)  ->  RiskManager (position caps, gross limit, kill switch).

It is PURE and deterministic: feed it each sleeve's CURRENT target weights and
the CURRENT equity, and it returns the approved net book plus any risk
violations. It does NOT fetch data or call alphas — the harness (Stage 6) does
the heterogeneous data->signal plumbing and hands the per-sleeve targets here.
That keeps the portfolio logic testable and identical in backtest, paper, and
live (the one-code-path principle, applied to the book).

Usage:
    pf = Portfolio(sleeve_returns=hist, risk=RiskManager(limits, equity))
    res = pf.construct(sleeve_targets={"trend": {...}, "ls": {...}}, equity=1e5)
    res.approved          # {instrument: weight} to trade, post risk checks
    alerts = pf.mark_to_market(equity, ts)   # post-trade kill-switch check
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mft.portfolio.book import book_exposure, inverse_vol_alloc, net_book
from mft.risk.limits import RiskManager


@dataclass
class BookResult:
    approved: dict[str, float]                       # risk-clipped target book
    raw: dict[str, float]                             # pre-risk net book
    violations: list[str] = field(default_factory=list)
    alloc: dict[str, float] = field(default_factory=dict)

    @property
    def exposure(self) -> dict[str, float]:
        return book_exposure(self.approved)


class Portfolio:
    """
    Combine weakly-correlated sleeves into one risk-managed book.

    Args:
        sleeve_returns: DataFrame of historical per-sleeve daily returns
                        (columns = sleeve names). Drives inverse-vol allocation.
        risk:           a configured RiskManager (caps, limits, kill switch).
        vol_window:     trailing window for the vol estimate (None = full sample).
    """

    def __init__(
        self,
        sleeve_returns: pd.DataFrame,
        risk: RiskManager,
        vol_window: int | None = None,
    ):
        if sleeve_returns.shape[1] == 0:
            raise ValueError("need at least one sleeve")
        self.sleeve_names = list(sleeve_returns.columns)
        self.risk = risk
        self._vol_window = vol_window
        self.alloc = inverse_vol_alloc(sleeve_returns, vol_window)

    def refresh_alloc(self, sleeve_returns: pd.DataFrame) -> None:
        """Re-estimate sleeve allocation from updated history (periodic refresh)."""
        self.alloc = inverse_vol_alloc(sleeve_returns, self._vol_window)

    def construct(
        self,
        sleeve_targets: dict[str, dict[str, float]],
        equity: float,
    ) -> BookResult:
        """
        Net the sleeves' current target weights into one book and risk-check it.
        Unknown sleeves (no allocation) contribute nothing. If the risk manager
        is halted, returns an empty book with the halt reason.
        """
        raw = net_book(sleeve_targets, self.alloc)
        approved, violations = self.risk.check_pre_trade(raw, equity)
        return BookResult(approved=approved, raw=raw, violations=violations, alloc=dict(self.alloc))

    def mark_to_market(self, equity: float, ts: pd.Timestamp | None = None) -> list[str]:
        """Post-trade loss/drawdown check. Returns kill-switch alerts (empty = ok)."""
        return self.risk.update_post_trade(equity, ts)

    def start_day(self, equity: float) -> None:
        self.risk.reset_daily(equity)

    @property
    def halted(self) -> bool:
        return self.risk.state.is_halted
