"""
Signal lab — grade a signal's predictive content WITHOUT assuming any fill or cost.

This is the IC-first heart of the engine. Before spending a backtest trial (which
raises the DSR bar for everything), ask the cheap question: does this signal's
cross-sectional ranking actually predict the next move?

  - IC / rank-IC : per-timestamp cross-sectional correlation(signal, forward return).
                   Rank-IC (Spearman) is the robust default — it ignores outliers and
                   only asks "did higher-ranked names out-return lower-ranked ones?".
  - IC t-stat    : mean(IC) / std(IC) * sqrt(#periods). |t| > ~2 is the usual screen.
  - buckets      : average forward return by signal quantile; a real signal is
                   MONOTONIC across buckets, not just good at one extreme.
  - alpha decay  : IC across horizons (15/30/60/120m) — where does the edge live and
                   how fast does it die? Tells you the holding period before you backtest.

Caveat baked in: with a 33-name universe each cross-sectional IC is a correlation over
<=33 points — noisy per period. Significance comes from MANY periods, and the universe
is survivorship-biased, so treat IC as evidence of *mechanism*, not a tradeable number.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mft.research.targets import forward_return_panel

BARS_PER_YEAR_DAILY = 252


def _align(signal: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in signal.columns if c in target.columns]
    idx = signal.index.intersection(target.index)
    return signal.loc[idx, cols], target.loc[idx, cols]


def _rowwise_pearson(a: np.ndarray, b: np.ndarray, min_names: int) -> np.ndarray:
    valid = ~(np.isnan(a) | np.isnan(b))
    n = valid.sum(axis=1)
    a = np.where(valid, a, np.nan)
    b = np.where(valid, b, np.nan)
    with warnings.catch_warnings():  # all-NaN rows are filtered by min_names below
        warnings.simplefilter("ignore", RuntimeWarning)
        ma = np.nanmean(a, axis=1, keepdims=True)
        mb = np.nanmean(b, axis=1, keepdims=True)
    da, db = a - ma, b - mb
    cov = np.nansum(da * db, axis=1)
    va = np.nansum(da * da, axis=1)
    vb = np.nansum(db * db, axis=1)
    denom = np.sqrt(va * vb)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / denom
    corr[(n < min_names) | (denom == 0)] = np.nan
    return corr


def ic_series(signal: pd.DataFrame, target: pd.DataFrame,
              method: str = "spearman", min_names: int = 5) -> pd.Series:
    """Per-timestamp cross-sectional IC. method='spearman' (rank-IC) or 'pearson'."""
    s, t = _align(signal, target)
    if method == "spearman":
        s = s.rank(axis=1)
        t = t.rank(axis=1)
    corr = _rowwise_pearson(s.to_numpy(float), t.to_numpy(float), min_names)
    return pd.Series(corr, index=s.index, name=f"{method}_ic").dropna()


def ic_summary(ic: pd.Series) -> dict:
    """Mean IC, its dispersion, t-stat, IR, and fraction of positive periods."""
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_tstat": np.nan,
                "ic_ir": np.nan, "ic_pct_pos": np.nan, "n_periods": n}
    mean, std = float(ic.mean()), float(ic.std())
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_tstat": float(mean / std * np.sqrt(n)) if std > 0 else np.nan,
        "ic_ir": float(mean / std) if std > 0 else np.nan,
        "ic_pct_pos": float((ic > 0).mean()),
        "n_periods": n,
    }


def ic_by_month(ic: pd.Series) -> pd.Series:
    """Mean IC per calendar month — stability check (is it carried by one stretch?)."""
    return ic.dropna().groupby(ic.dropna().index.tz_convert("UTC").to_period("M")).mean()


def bucket_returns(signal: pd.DataFrame, target: pd.DataFrame,
                   n_buckets: int = 5, min_names: int = 5) -> pd.Series:
    """
    Mean forward return by signal quantile bucket (0 = lowest signal). A genuine
    signal is monotonic: bucket return should rise (or fall) steadily across buckets.
    """
    s, t = _align(signal, target)
    pct = s.rank(axis=1, pct=True)
    enough = pct.notna().sum(axis=1) >= min_names
    pct, tv = pct[enough], t[enough]
    bucket = np.minimum((pct.to_numpy(float) * n_buckets).astype("float"), n_buckets - 1)
    tvals = tv.to_numpy(float)
    out = {}
    for b in range(n_buckets):
        mask = bucket == b
        vals = tvals[mask]
        out[b] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    return pd.Series(out, name="bucket_fwd_ret")


def bucket_monotonicity(bucket_ret: pd.Series) -> float:
    """Spearman corr of bucket index vs bucket return in [-1, 1]. +1 = perfectly monotone up."""
    br = bucket_ret.dropna()
    if len(br) < 3:
        return np.nan
    return float(pd.Series(br.index, index=br.index).corr(br, method="spearman"))


def alpha_decay(signal: pd.DataFrame, close_wide: pd.DataFrame,
                horizons_bars: list[int], method: str = "spearman") -> pd.Series:
    """Mean rank-IC of the signal against forward returns at each horizon."""
    out = {}
    for h in horizons_bars:
        tgt = forward_return_panel(close_wide, h)
        out[h] = ic_summary(ic_series(signal, tgt, method=method))["ic_mean"]
    return pd.Series(out, name="ic_by_horizon")
