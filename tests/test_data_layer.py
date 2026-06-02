"""Data-layer contract tests."""

from __future__ import annotations

import pandas as pd

from mft.data_layer.loader import _normalize


def test_normalize_preserves_ohlcv_column_order():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "Close": [100.0],
            "Volume": [1_000_000.0],
            "Open": [99.0],
            "High": [101.0],
            "Low": [98.0],
        }
    )

    normalized = _normalize(df)

    assert list(normalized.columns[:5]) == ["open", "high", "low", "close", "volume"]
