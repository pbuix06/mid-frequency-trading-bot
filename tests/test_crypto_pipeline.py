"""
Crypto pipeline logic: 24/7 resampler + validation checks, on synthetic fixtures.
(These test the CODE, not market results — no fake research data is produced.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.data_layer.crypto_validate import (
    ohlcv_is_clean,
    validate_funding,
    validate_ohlcv,
    validate_oi,
)
from mft.research.crypto_panel import compute_basis, resample_crypto_bars


def _min_bars(start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = pd.Series(100 + np.arange(n) * 0.1, index=idx)
    return pd.DataFrame({
        "open": close, "high": close + 0.05, "low": close - 0.05, "close": close,
        "volume": np.ones(n) * 10.0, "quote_volume": close * 10.0,
        "trades": np.full(n, 5.0), "taker_buy_base": np.ones(n) * 6.0,
        "taker_buy_quote": close * 6.0,
    }, index=idx)


def test_resample_is_24_7_and_aggregates():
    # bars crossing midnight — crypto keeps them all (no RTH/overnight drop)
    df = _min_bars("2023-01-01 23:58", 10)  # 23:58 .. 00:07 next day
    out = resample_crypto_bars(df, "5min")
    midnight = pd.Timestamp("2023-01-02 00:00", tz="UTC")
    assert midnight in out.index                 # overnight bucket kept (24/7)
    first = out.iloc[0]
    assert first["volume"] == 20.0               # 2 one-min bars summed
    assert first["open"] == df["open"].iloc[0]   # first
    assert "vwap" in out.columns and np.isfinite(out["vwap"]).all()


def test_validate_clean_frame_passes():
    rep = validate_ohlcv(_min_bars("2023-03-01 00:00", 120))
    assert ohlcv_is_clean(rep)
    assert rep["tz_is_utc"] and rep["monotonic_increasing"]
    assert rep["duplicate_timestamps"] == 0 and rep["n_gaps"] == 0
    assert rep["coverage_pct"] == 100.0


def test_validate_catches_ohlc_inconsistency_and_neg_price():
    df = _min_bars("2023-03-01 00:00", 50)
    df.iloc[10, df.columns.get_loc("low")] = df.iloc[10]["high"] + 5  # low > high
    df.iloc[20, df.columns.get_loc("close")] = -1.0                   # negative price
    rep = validate_ohlcv(df)
    assert rep["ohlc_inconsistent_bars"] >= 1
    assert rep["negative_or_zero_prices"] >= 1
    assert not ohlcv_is_clean(rep)


def test_validate_catches_duplicates_and_gaps():
    df = _min_bars("2023-03-01 00:00", 60)
    dup = pd.concat([df, df.iloc[[30]]]).sort_index()       # duplicate timestamp
    assert validate_ohlcv(dup)["duplicate_timestamps"] >= 1
    gapped = df.drop(df.index[20:25])                        # 5-minute outage
    rep = validate_ohlcv(gapped)
    assert rep["n_gaps"] >= 1 and rep["missing_bars"] == 5
    assert rep["coverage_pct"] < 100.0


def test_validate_funding_8h_alignment():
    idx = pd.date_range("2023-03-01", periods=12, freq="8h", tz="UTC")
    ser = pd.Series(np.linspace(-1e-4, 1e-4, 12), index=idx, name="funding_rate")
    rep = validate_funding(ser)
    assert rep["present"] and rep["looks_8h_aligned"] and rep["median_spacing_hours"] == 8.0


def test_validate_oi_spacing():
    idx = pd.date_range("2023-03-01", periods=20, freq="5min", tz="UTC")
    oi = pd.DataFrame({"open_interest": np.arange(20.0) + 1, "oi_value": np.arange(20.0) + 1}, index=idx)
    rep = validate_oi(oi)
    assert rep["present"] and rep["median_spacing_minutes"] == 5.0 and rep["negative_oi_bars"] == 0


def test_compute_basis():
    idx = pd.date_range("2023-03-01", periods=5, freq="5min", tz="UTC")
    spot = pd.Series([100, 100, 100, 100, 100.0], index=idx)
    perp = pd.Series([101, 100.5, 99, 100, 102.0], index=idx)
    basis = compute_basis(spot, perp)
    assert np.allclose(basis.values, [0.01, 0.005, -0.01, 0.0, 0.02])
