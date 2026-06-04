"""Alpaca parser tests — network-free, using synthetic bar JSON."""

from __future__ import annotations

import pandas as pd

from mft.data_layer.alpaca_ingest import _parse_bars


def _bars() -> list[dict]:
    return [
        {"t": "2023-01-03T14:30:00Z", "o": 100.0, "h": 100.5, "l": 99.8, "c": 100.2,
         "v": 12000, "n": 85, "vw": 100.1},
        {"t": "2023-01-03T14:31:00Z", "o": 100.2, "h": 100.3, "l": 99.9, "c": 100.0,
         "v": 9000, "n": 60, "vw": 100.05},
    ]


def test_parse_produces_utc_ohlcv():
    df = _parse_bars(_bars())
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "trades", "vwap"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[0] == 100.2
    assert df["trades"].iloc[0] == 85.0
    assert df["vwap"].iloc[1] == 100.05


def test_parse_empty_returns_empty_schema():
    df = _parse_bars([])
    assert df.empty
    assert "vwap" in df.columns


def test_parse_drops_nonpositive_close():
    bars = _bars() + [{"t": "2023-01-03T14:32:00Z", "o": 0, "h": 0, "l": 0, "c": 0, "v": 0}]
    df = _parse_bars(bars)
    assert len(df) == 2  # the zero-close bar dropped


def test_parse_handles_missing_optional_fields():
    # n (trades) and vw (vwap) absent -> defaults, no crash
    bars = [{"t": "2023-01-03T14:30:00Z", "o": 50.0, "h": 51.0, "l": 49.5, "c": 50.5, "v": 100}]
    df = _parse_bars(bars)
    assert df["vwap"].iloc[0] == 50.5   # falls back to close
    assert df["trades"].iloc[0] == 0.0
    assert isinstance(df.index, pd.DatetimeIndex)
