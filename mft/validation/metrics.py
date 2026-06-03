"""
Performance metrics.

Always evaluate on a VECTOR of metrics, never a single number.
A Sharpe of 1.5 with Calmar of 0.3 is a flag, not a win.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio.

    Always specify rf. The 2021→2023 cash-yield rise from ~1% to >5% mechanically
    shrank Sharpe numerators — leaving rf=0 inflates recent Sharpes.
    """
    excess = returns - rf / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Annualized Sortino ratio (downside deviation only).
    Sortino >> Sharpe = healthy upside skew (trend-following).
    Sortino ≈/< Sharpe = red flag.
    """
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown. Negative number, e.g. -0.25 = -25% drawdown."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    CAGR / abs(max drawdown).
    Target > 1.0; elite > 2.0; flatters in calm periods.
    """
    n = len(returns)
    if n == 0:
        return np.nan
    annual = (1 + returns).prod() ** (periods_per_year / n) - 1
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return float(annual / mdd)


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    n = len(returns)
    if n == 0:
        return np.nan
    return float((1 + returns).prod() ** (periods_per_year / n) - 1)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive return. Needs ≥ 30 observations."""
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Gross profit / gross loss. > 1.0 required; > 1.5 solid."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf
    return float(gains / losses)


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical Value-at-Risk: the alpha-quantile daily loss (a negative number).
    VaR(5%) = -0.03 means "on the worst 5% of days, you lose at least 3%".
    """
    if returns.empty:
        return float("nan")
    return float(np.quantile(returns, alpha))


def conditional_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Conditional VaR / Expected Shortfall: the MEAN loss on days worse than the
    VaR threshold (a negative number). Captures tail severity that VaR misses.
    """
    if returns.empty:
        return float("nan")
    var = np.quantile(returns, alpha)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else float(var)


def full_metrics(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute the full metric vector. Use this — never a single point estimate.
    Interpret Sharpe alongside Calmar, drawdown, and regime analysis.
    """
    return {
        "sharpe": sharpe(returns, rf, periods_per_year),
        "sortino": sortino(returns, rf, periods_per_year),
        "calmar": calmar(returns, periods_per_year),
        "cagr": cagr(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "volatility": float(returns.std() * np.sqrt(periods_per_year)),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
        "n_periods": int(len(returns)),
    }
