"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv():
    """200 bars of synthetic OHLCV. No survivorship bias risk; use freely."""
    rng = np.random.default_rng(42)
    n = 200
    log_ret = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices * rng.uniform(0.998, 1.000, n),
            "high": prices * rng.uniform(1.000, 1.005, n),
            "low": prices * rng.uniform(0.995, 1.000, n),
            "close": prices,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )
