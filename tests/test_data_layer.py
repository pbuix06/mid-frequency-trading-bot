"""Data-layer contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.data_layer.cleaning import remove_outliers
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


def _ohlcv(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c,
         "volume": np.full(n, 1e6)},
        index=idx,
    )


def test_remove_outliers_is_causal_not_full_sample():
    """
    A spike late in the series must NOT change how an identical earlier bar is
    treated. Full-sample stats would let the late spike inflate sigma and leak
    backward; the causal (trailing) version must be unaffected.
    """
    base = [100.0] * 200
    df_clean = _ohlcv(base)

    # Same series but with a huge spike injected at bar 180 (the "future")
    spiked = list(base)
    spiked[180] = 100.0 * 1.5  # +50% one-day jump
    df_spiked = _ohlcv(spiked)

    out_clean = remove_outliers(df_clean, z_threshold=5.0, window=63)
    out_spiked = remove_outliers(df_spiked, z_threshold=5.0, window=63)

    # Bars BEFORE the spike must be byte-identical in both runs (no backward leak)
    pre = df_clean.index[:180]
    pd.testing.assert_frame_equal(out_clean.loc[pre], out_spiked.loc[pre])


def test_remove_outliers_flags_a_genuine_spike():
    """A clear outlier inside the series is neutralized (ffill from prior bar)."""
    closes = [100.0] * 100
    closes[80] = 100.0 * 2.0  # +100% spike — unambiguous outlier
    df = _ohlcv(closes)
    out = remove_outliers(df, z_threshold=5.0, window=63, fill="ffill")
    # The spike close should be replaced by the prior (100.0), not left at 200.0
    assert out["close"].iloc[80] == 100.0


def test_remove_outliers_ignores_tiny_move_after_flat_window():
    """
    After a perfectly flat (sigma≈0) window, a tiny resumption move must NOT be
    falsely flagged — the sigma floor prevents the divide-by-~0 blowup.
    """
    closes = [100.0] * 100
    closes[80] = 100.05  # +0.05% — a normal move, not an outlier
    df = _ohlcv(closes)
    out = remove_outliers(df, z_threshold=5.0, window=63, fill="ffill")
    assert out["close"].iloc[80] == 100.05, "tiny move wrongly flagged after flat window"
