"""
Deflated Sharpe Ratio (DSR) and Minimum Backtest Length (MinBTL).

Bailey & López de Prado (2014), "Pseudo-Mathematics and Financial Charlatanism."

Key results:
  - With N=10 trials you expect max in-sample Sharpe of 1.57 on pure noise.
  - A 7-parameter binary search = 128 trials = expected max Sharpe > 2.6, zero true edge.
  - MinBTL: with only 5 years of daily data, max 45 independent configurations.

Use:
    n = trial_log.count()
    budget = min_backtest_length(n)
    dsr = deflated_sharpe_ratio(observed_sharpe, returns, n_trials=n)
    assert dsr >= 0.95, "Not significant — do not advance to paper trading"
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """
    E[max(SR)] for n_trials IID normal Sharpe ratios with std = sr_std.

    This is the benchmark the observed Sharpe must beat.
    More trials = higher expected max = higher hurdle.
    """
    if n_trials <= 0:
        return 0.0
    euler_gamma = 0.5772156649
    # Approximation via expected maximum of normal order statistics
    z1 = stats.norm.ppf(1 - 1 / n_trials) if n_trials > 1 else 0.0
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e)) if n_trials > 1 else 0.0
    e_max = (1 - euler_gamma) * z1 + euler_gamma * z2
    return float(sr_std * max(e_max, 0.0))


def deflated_sharpe_ratio(
    sr: float,
    returns: np.ndarray | list,
    n_trials: int,
    sr_benchmark: float | None = None,
) -> float:
    """
    Probability that the observed Sharpe is real given trial count.

    Corrects for:
      - Selection bias (multiple testing)
      - Non-normality of returns (skew, kurtosis)
      - Sample length

    Args:
        sr: Observed annualized Sharpe ratio.
        returns: Period returns array (not annualized).
        n_trials: Total configs tried INCLUDING this one.
        sr_benchmark: Sharpe to beat (default: expected max under null).

    Returns:
        Probability in [0, 1]. Advance to paper only if >= 0.95.
    """
    returns = np.asarray(returns)
    T = len(returns)
    if T < 5:
        return 0.0

    mu = returns.mean()
    sigma = returns.std(ddof=1) + 1e-16
    skew = float(np.mean(((returns - mu) / sigma) ** 3))
    kurt = float(np.mean(((returns - mu) / sigma) ** 4))

    if sr_benchmark is None:
        sr_benchmark = expected_max_sharpe(n_trials)

    # Standard error of Sharpe, adjusted for non-normality
    sr_se = np.sqrt(
        (1 + 0.5 * sr**2 - skew * sr + (kurt - 3) / 4 * sr**2) / max(T - 1, 1)
    )

    dsr = float(stats.norm.cdf((sr - sr_benchmark) / (sr_se + 1e-16)))
    return float(np.clip(dsr, 0.0, 1.0))


def min_backtest_length(n_trials: int, periods_per_year: int = 252) -> int:
    """
    Minimum number of daily bars (MinBTL) needed to support n_trials.

    MinBTL = periods_per_year × E[max_N]²

    Where E[max_N] is the expected maximum of N iid standard normal variates
    (approximated via the Euler-Mascheroni formula).

    Worked examples matching Bailey, Borwein, López de Prado & Zhu (2014):
      N=7  → ~485 bars ≈ 2 years   (paper: "2-year backtest, 7 configs")
      N=45 → 1260 bars = 5 years   (paper: "5 years → max 45 configs")

    With 5 years (1260 bars): max ~45 independent trials.
    With 2 years (504 bars): max ~7 trials before E[IS Sharpe] = 1 on noise.

    Returns bars (not years). Divide by periods_per_year for years.
    """
    if n_trials <= 1:
        return periods_per_year
    e_max = expected_max_sharpe(n_trials)
    min_bars = int(np.ceil(periods_per_year * e_max**2))
    return max(min_bars, periods_per_year)
