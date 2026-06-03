"""
EDGAR PIT-correctness tests — as important as the look-ahead test.

Fundamentals are the #1 source of look-ahead bias: a number must be stamped when
it was KNOWABLE (the SEC `filed` date), not when the period it describes ended,
and a later RESTATEMENT must never overwrite what you originally knew. These
tests use synthetic companyfacts (no network) to lock that in.
"""

from __future__ import annotations

import pandas as pd

from mft.data_layer.edgar_ingest import extract_fundamentals, extract_pit_series, pit_value


def _facts() -> dict:
    """Synthetic companyfacts: two quarters + a later restatement of Q1."""
    return {
        "facts": {
            "us-gaap": {
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"end": "2020-03-31", "val": 100.0, "filed": "2020-05-01", "form": "10-Q"},
                            {"end": "2020-06-30", "val": 110.0, "filed": "2020-08-01", "form": "10-Q"},
                            # restatement of Q1 filed a year later — must be IGNORED
                            {"end": "2020-03-31", "val": 999.0, "filed": "2021-05-01", "form": "10-K/A"},
                        ]
                    }
                },
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [
                        {"end": "2020-03-31", "val": 1000.0, "filed": "2020-05-01", "form": "10-Q"},
                    ]}
                }
            },
        }
    }


def test_extract_keeps_earliest_filing_per_period():
    s = extract_pit_series(_facts(), "StockholdersEquity", "us-gaap", "USD")
    # The 2020-03-31 period must carry its ORIGINAL value (100), not the restatement (999).
    assert 100.0 in s.values
    assert 999.0 not in s.values
    assert s.index.is_monotonic_increasing


def test_pit_value_does_not_leak_future_filings():
    df = extract_fundamentals(_facts())
    # Before anything is filed -> unknown
    assert pd.isna(pit_value(df, "book_equity", pd.Timestamp("2020-04-01", tz="UTC")))
    # After Q1 filing but before Q2 filing -> Q1 value
    assert pit_value(df, "book_equity", pd.Timestamp("2020-06-01", tz="UTC")) == 100.0
    # After Q2 filing -> Q2 value
    assert pit_value(df, "book_equity", pd.Timestamp("2020-09-01", tz="UTC")) == 110.0


def test_restatement_never_changes_what_was_known():
    df = extract_fundamentals(_facts())
    # Even a year AFTER the Q1 restatement was filed, the Q1 figure we acted on
    # back in 2020 stays the originally-reported 100 (earliest-filing rule).
    known = df[(df["item"] == "book_equity")].sort_values("filed")
    assert (known["value"] == 999.0).sum() == 0


def test_index_is_filed_date_not_period_end():
    s = extract_pit_series(_facts(), "StockholdersEquity", "us-gaap", "USD")
    # First knowable date is the FILED date (2020-05-01), not the period end (2020-03-31).
    assert s.index[0] == pd.Timestamp("2020-05-01", tz="UTC")
