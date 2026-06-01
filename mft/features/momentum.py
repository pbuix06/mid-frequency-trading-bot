"""
Momentum and volatility features. Pure functions — no side effects.

These are building blocks for alpha compute_signal() methods.
They must be pure so the look-ahead unit test can verify them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Log returns over n periods."""
    return np.log(prices / prices.shift(periods))


def rolling_volatility(prices: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualized rolling volatility of log returns."""
    ret = log_returns(prices)
    return ret.rolling(window).std() * np.sqrt(periods_per_year)


def time_series_momentum(prices: pd.Series, lookback: int, skip: int = 1) -> pd.Series:
    """
    Time-series momentum signal: return over [skip, lookback] bars.
    skip=1 skips the most recent bar to avoid short-term reversal contamination.
    Moskowitz, Ooi & Pedersen (2012).
    """
    return prices.shift(skip) / prices.shift(lookback) - 1


def z_score(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score. Foundation for mean-reversion signals."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / (sigma + 1e-10)


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank assets cross-sectionally at each timestamp.
    Returns percentile ranks in [0, 1]. Used for cross-sectional momentum.
    """
    return df.rank(axis=1, pct=True)


def dollar_volume(df: pd.DataFrame) -> pd.Series:
    """Dollar volume: proxy for liquidity. Use for universe filtering."""
    return df["close"] * df["volume"]
