"""
Transaction-cost assumptions — single source of truth.

Costs are NOT equal across instruments (see CLAUDE.md / handoff). Two layers:

  - Flat default (DEFAULT_COMMISSION_PCT / DEFAULT_SLIPPAGE_PCT): one canonical
    per-leg rate every engine defaults to. Conservative for ultra-liquid ETFs
    (SPY real cost ~1-2 bp), optimistic for illiquid single names.

  - liquidity_tiered_cost(adv): a crude per-name cost that scales DOWN with
    trailing dollar volume, so a ~600-name cross-sectional book is not charged
    the same rate on a mega-cap and a micro-cap near delisting. Use this for the
    XS equity book, where the flat rate otherwise flatters the result.

These are deliberately simple research assumptions. Real per-instrument impact
(square-root law / Almgren-Chriss) is calibrated against LIVE FILLS in Phase 6
(implementation shortfall) and fed back here — building it on guessed
coefficients now would be false precision. Until then, the 2x-3x cost-STRESS
test is how we handle the uncertainty, not a fancier point estimate.
"""

from __future__ import annotations

# Flat per-leg rates (commission and slippage charged separately by the engines).
DEFAULT_COMMISSION_PCT = 0.001   # 10 bps per leg
DEFAULT_SLIPPAGE_PCT = 0.001     # 10 bps per leg
# Combined per-trade cost (both legs of the model) — the flat default a single
# trade pays on its traded notional.
DEFAULT_COST_PCT = DEFAULT_COMMISSION_PCT + DEFAULT_SLIPPAGE_PCT  # 20 bps/trade


def liquidity_tiered_cost(avg_dollar_vol: float, multiplier: float = 1.0) -> float:
    """
    Combined per-trade cost (commission + slippage) as a function of a name's
    trailing average daily dollar volume. Rough tiers; calibrate to fills later.

    Tiers (one-way, charged on traded notional):
        >= $1B ADV   : 10 bps  (mega-cap, tight spread)
        >= $100M ADV : 20 bps  (large-cap; ~= the flat default)
        >= $20M ADV  : 40 bps  (mid / small-cap)
        <  $20M ADV  : 80 bps  (illiquid — where the flat rate most flatters)

    `multiplier` scales every tier (used by the cost-stress test: 1x / 2x / 3x).
    """
    if avg_dollar_vol >= 1_000_000_000:
        base = 0.0010
    elif avg_dollar_vol >= 100_000_000:
        base = 0.0020
    elif avg_dollar_vol >= 20_000_000:
        base = 0.0040
    else:
        base = 0.0080
    return base * multiplier
