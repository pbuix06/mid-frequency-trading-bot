"""
Data loader: Parquet (production) or CSV (dev/Yahoo).

All output DataFrames:
  - DatetimeIndex in UTC
  - Columns: [open, high, low, close, volume]
  - Sorted ascending by timestamp

Production path: load_parquet() with DuckDB for fast columnar reads.
Dev path: load_csv() for Yahoo/broker exports.

DO NOT use Yahoo data to validate strategies. It is survivorship-biased
and adjusted post-hoc. Use it only for learning the plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REQUIRED = {"open", "high", "low", "close", "volume"}
_OHLCV = ["open", "high", "low", "close", "volume"]


def load_parquet(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    """
    Load OHLCV from partitioned Parquet. Primary production path.
    Supports both single files and directory partitions (date=YYYY-MM-DD/...).
    """
    path = Path(path)
    try:
        import duckdb

        if path.is_dir():
            query = f"SELECT * FROM read_parquet('{path}/**/*.parquet', hive_partitioning=true) ORDER BY timestamp"
        else:
            query = f"SELECT * FROM read_parquet('{path}') ORDER BY timestamp"
        df = duckdb.query(query).df()
    except ImportError:
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        df = dataset.to_table().to_pandas()

    df = _normalize(df)
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol].drop(columns=["symbol"])
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV from CSV (Yahoo / broker export). Dev only."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalize(df)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, ensure UTC DatetimeIndex, validate schema."""
    # Lowercase all column names
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    col_map = {"adj_close": "close", "date": "timestamp"}
    df = df.rename(columns=col_map)

    # Promote timestamp column to index if needed
    for candidate in ("timestamp", "date", "datetime"):
        if candidate in df.columns:
            df = df.set_index(candidate)
            break

    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"

    # Ensure UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()

    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")

    # Return only OHLCV plus any extras (e.g. symbol, adjusted_close)
    extras = [c for c in df.columns if c not in _REQUIRED]
    return df[_OHLCV + extras]
