"""
Portfolio construction: combine alpha signals into target weights.

Start with inverse volatility (simple, robust).
Add Ledoit-Wolf shrinkage, mean-variance, HRP as you mature.

Grinold's Fundamental Law: IR ≈ IC × √BR × TC.
Protect the transfer coefficient — over-constraining or over-trading drags it to zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_weights(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    vol_window: int = 60,
) -> pd.DataFrame:
    """
    Weight alpha signals by inverse realized volatility.

    Higher vol → lower weight → approximately equal risk contribution.
    Simple and robust for a portfolio of uncorrelated alphas.

    Args:
        signals: Target weights per alpha, shape (T, n_alphas). Values in [-1, 1].
        returns: Returns per alpha/asset, shape (T, n_alphas).
        vol_window: Rolling window for volatility estimate.

    Returns:
        Reweighted signals normalized to unit gross exposure.
    """
    vol = returns.rolling(vol_window).std()
    inv_vol = 1.0 / (vol + 1e-10)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return signals * weights


def risk_parity_weights(cov: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """
    Equal risk contribution (risk parity) weights.

    Each asset contributes equally to portfolio variance.
    Robust to estimation error because it doesn't require expected returns.

    Args:
        cov: Covariance matrix of shape (n, n).

    Returns:
        Weight vector of shape (n,) summing to 1.
    """
    n = cov.shape[0]
    w = np.ones(n) / n

    for _ in range(max_iter):
        marginal = cov @ w
        rc = w * marginal
        target_rc = rc.sum() / n
        w = w * target_rc / (marginal + 1e-12)
        w = np.maximum(w, 1e-8)
        w /= w.sum()

    return w


def apply_position_limits(
    weights: pd.Series,
    max_position: float = 0.25,
) -> pd.Series:
    """
    Clip positions to max_position and renormalize gross exposure.

    Per-position cap ≤ 25% of equity; reduce for correlated positions.
    """
    clipped = weights.clip(-max_position, max_position)
    gross = clipped.abs().sum()
    if gross > 1.0:
        clipped = clipped / gross
    return clipped
