"""
Combinatorial Purged Cross-Validation (CPCV).

López de Prado (2018), "Advances in Financial Machine Learning."

Superior to K-fold and walk-forward:
  - Generates C(N,k) backtest paths → a Sharpe DISTRIBUTION, not one number
  - Lower probability of backtest overfitting
  - Higher deflated Sharpe

Standard K-fold is invalid in finance: it ignores temporal dependence.
Purging and embargoing are mandatory whenever labels are forward-looking.

Usage:
    splits = cpcv_splits(n=len(data), n_groups=6, n_test_groups=2)
    sharpes = []
    for train_idx, test_idx in splits:
        # fit on data.iloc[train_idx], evaluate on data.iloc[test_idx]
        sharpes.append(compute_sharpe(data.iloc[test_idx]))
    # sharpes is now a distribution — report it, don't just take the mean
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def purge_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo_bars: int = 5,
) -> np.ndarray:
    """
    Remove training samples near the test set to prevent leakage.

    Purge: remove train samples whose label horizon overlaps the test period.
    Embargo: remove train samples within embargo_bars after the test period ends.

    Args:
        train_idx: Training bar indices.
        test_idx: Test bar indices.
        embargo_bars: Bars to embargo after test set end.

    Returns:
        Purged and embargoed training indices.
    """
    test_start = int(test_idx.min())
    test_end = int(test_idx.max())
    embargo_end = test_end + embargo_bars

    # Keep train samples that don't overlap with [test_start, embargo_end]
    clean = train_idx[(train_idx < test_start) | (train_idx > embargo_end)]
    return clean


def cpcv_splits(
    n: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_bars: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate CPCV train/test splits.

    Args:
        n: Total number of time-series samples.
        n_groups: N in C(N, k) — number of groups to split data into.
        n_test_groups: k in C(N, k) — groups held out per split.
        embargo_bars: Embargo period between train and test.

    Returns:
        List of (train_idx, test_idx) pairs. Length = C(n_groups, n_test_groups).
        Concatenate test_idx results across all splits to cover the full dataset.
    """
    indices = np.arange(n)
    boundaries = np.linspace(0, n, n_groups + 1, dtype=int)
    groups = [indices[boundaries[i] : boundaries[i + 1]] for i in range(n_groups)]

    splits = []
    for test_group_ids in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_group_ids])
        train_idx = np.concatenate(
            [groups[g] for g in range(n_groups) if g not in test_group_ids]
        )
        train_idx = purge_embargo(train_idx, test_idx, embargo_bars)
        splits.append((train_idx, test_idx))

    return splits
