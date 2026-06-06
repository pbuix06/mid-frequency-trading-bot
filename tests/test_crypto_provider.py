"""
Crypto interface contract: normalize works (24/7, no session masking, perp columns
created), and the placeholder provider REFUSES to return data (no fake results).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mft.data_layer.crypto_provider import (
    NORMALIZED_COLUMNS,
    _UnimplementedCryptoProvider,
    normalize_crypto_bars,
)


def test_normalize_creates_schema_and_keeps_24_7():
    # tz-naive UTC-ish index spanning a weekend overnight (crypto trades through it)
    idx = pd.date_range("2023-01-06 22:00", periods=6, freq="5min")  # Fri night -> Sat
    df = pd.DataFrame({
        "open": np.arange(6.0), "high": np.arange(6.0) + 1,
        "low": np.arange(6.0) - 1, "close": np.arange(6.0) + 0.5,
        "volume": np.ones(6),
    }, index=idx)
    out = normalize_crypto_bars(df, "BTCUSDT", "binance", market="perp", exchange="binance")
    assert list(out.columns) == NORMALIZED_COLUMNS          # perp cols added as NaN
    assert out.index.tz is not None and str(out.index.tz) == "UTC"
    assert len(out) == 6                                    # NOTHING dropped for weekend
    assert out.attrs["market"] == "perp" and out["funding_rate"].isna().all()


def test_missing_required_columns_raises():
    df = pd.DataFrame({"open": [1.0], "close": [1.0]})
    with pytest.raises(ValueError):
        normalize_crypto_bars(df, "ETHUSDT", "binance")


def test_placeholder_refuses_to_fabricate():
    p = _UnimplementedCryptoProvider()
    with pytest.raises(NotImplementedError):
        p.get_bars("BTCUSDT", "2023-01-01", "2023-02-01")
    with pytest.raises(NotImplementedError):
        p.get_funding("BTCUSDT", "2023-01-01", "2023-02-01")
