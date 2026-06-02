"""
Research diagnostics — the robustness lenses required before Phase 4 validation.

These are NOT the formal Phase 4 gate (that is CPCV + DSR in cpcv.py / dsr.py).
They are the pre-validation health checks the playbook calls for:

  - cost_stress_curve : re-run economics at 1x / 2x / 3x costs. If the edge
                        dies when costs double, "it wasn't an edge, it was a
                        cost illusion" (playbook §4.4 / Gate 4).
  - turnover          : annualized two-way turnover from a weight history.
                        Drives cost and capacity; a high-turnover sleeve with a
                        thin Sharpe will not survive realistic costs.
  - rolling_sharpe    : trailing Sharpe series — reveals whether one period
                        carries the whole curve (decay / regime dependence).

All functions are pure and operate on returns / weight Series already produced
by the harnesses, so they add no new look-ahead surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.validation.metrics import max_drawdown, sharpe


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Trailing annualized Sharpe over a rolling window.

    A healthy alpha has a rolling Sharpe that stays mostly positive; one that
    spikes in a single window and is flat/negative elsewhere is regime-dependent
    and likely to decay live.
    """
    if len(returns) < window:
        return pd.Series(dtype=float, name="rolling_sharpe")
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std()
    rs = (mu / sd.replace(0.0, np.nan)) * np.sqrt(252)
    rs.name = "rolling_sharpe"
    return rs.dropna()


def rolling_sharpe_summary(returns: pd.Series, window: int = 252) -> dict:
    """Fraction of rolling windows that are positive, plus min/median/max."""
    rs = rolling_sharpe(returns, window)
    if rs.empty:
        return {"frac_positive": float("nan"), "min": float("nan"),
                "median": float("nan"), "max": float("nan"), "n_windows": 0}
    return {
        "frac_positive": float((rs > 0).mean()),
        "min": float(rs.min()),
        "median": float(rs.median()),
        "max": float(rs.max()),
        "n_windows": int(len(rs)),
    }


def turnover_from_weights(weights: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """
    Annualized two-way turnover from a per-rebalance weight history.

    weights: DataFrame indexed by rebalance timestamp, columns = symbols,
             values = target weights at each rebalance.

    Turnover at each rebalance = sum(|w_t - w_{t-1}|) (two-way: a full swap of
    a 100% book = 2.0). Annualized by the average rebalance frequency.

    Returns dict: per_rebalance_mean, annualized, n_rebalances.
    """
    if weights is None or weights.empty or len(weights) < 2:
        return {"per_rebalance_mean": float("nan"), "annualized": float("nan"),
                "n_rebalances": 0}

    w = weights.fillna(0.0).sort_index()
    deltas = w.diff().abs().sum(axis=1).iloc[1:]   # drop first (no prior weights)
    per_rebalance_mean = float(deltas.mean())

    # Annualize using observed average spacing between rebalances
    span_days = (w.index[-1] - w.index[0]).days
    if span_days <= 0:
        rebals_per_year = float(len(w))
    else:
        rebals_per_year = len(w) / (span_days / 365.25)

    return {
        "per_rebalance_mean": per_rebalance_mean,
        "annualized": float(per_rebalance_mean * rebals_per_year),
        "n_rebalances": int(len(w)),
    }


def cost_stress_curve(run_fn, multipliers=(1.0, 2.0, 3.0), base_cost: float = 0.001) -> list[dict]:
    """
    Re-run a strategy at escalating cost multipliers and report Sharpe decay.

    Args:
        run_fn:      callable(commission_pct, slippage_pct) -> pd.Series of returns.
                     The caller closes over the alpha/data; this fn only varies cost.
        multipliers: cost multipliers to test (1x baseline, 2x stress, 3x extreme).
        base_cost:   the 1x commission/slippage rate (each leg).

    Returns:
        list of dicts: {multiplier, cost_pct, sharpe, max_drawdown, survives}
        'survives' is True if Sharpe stays > 0 at that cost level.
    """
    rows = []
    for m in multipliers:
        cost = base_cost * m
        rets = run_fn(commission_pct=cost, slippage_pct=cost)
        s = sharpe(rets) if len(rets) else float("nan")
        dd = max_drawdown(rets) if len(rets) else float("nan")
        rows.append({
            "multiplier": m,
            "cost_pct": cost,
            "sharpe": float(s),
            "max_drawdown": float(dd),
            "survives": bool(s > 0) if not np.isnan(s) else False,
        })
    return rows
