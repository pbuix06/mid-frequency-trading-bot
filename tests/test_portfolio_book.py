"""Book construction tests — sleeve allocation + per-instrument netting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.portfolio.book import book_exposure, inverse_vol_alloc, net_book


def test_inverse_vol_alloc_favors_low_vol_sleeve():
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    r = pd.DataFrame({
        "calm":  rng.normal(0, 0.005, 300),   # low vol -> bigger weight
        "wild":  rng.normal(0, 0.02, 300),     # high vol -> smaller weight
    }, index=idx)
    a = inverse_vol_alloc(r)
    assert abs(sum(a.values()) - 1.0) < 1e-9
    assert a["calm"] > a["wild"]


def test_inverse_vol_alloc_degenerate_returns_equal_weight():
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    r = pd.DataFrame({"a": np.zeros(10), "b": np.zeros(10)}, index=idx)
    a = inverse_vol_alloc(r)
    assert abs(a["a"] - 0.5) < 1e-9 and abs(a["b"] - 0.5) < 1e-9


def test_net_book_combines_overlapping_instruments():
    # Two sleeves both touch SPY; one long, one short -> they net.
    targets = {
        "trend": {"SPY": 1.0},
        "ls":    {"SPY": -0.5, "AAPL": 0.5},
    }
    alloc = {"trend": 0.5, "ls": 0.5}
    book = net_book(targets, alloc)
    assert abs(book["SPY"] - (0.5 * 1.0 + 0.5 * -0.5)) < 1e-9   # 0.25
    assert abs(book["AAPL"] - (0.5 * 0.5)) < 1e-9               # 0.25


def test_net_book_preserves_dollar_neutrality():
    # A dollar-neutral sleeve stays neutral after allocation scaling.
    targets = {"ls": {"A": 0.5, "B": -0.5}}
    book = net_book(targets, {"ls": 0.3})
    assert abs(sum(book.values())) < 1e-9


def test_net_book_drops_canceling_positions():
    targets = {"x": {"SPY": 0.5}, "y": {"SPY": -0.5}}
    book = net_book(targets, {"x": 0.5, "y": 0.5})
    assert "SPY" not in book   # exactly cancels -> dropped


def test_book_exposure_metrics():
    book = {"A": 0.5, "B": -0.3, "C": 0.2}
    e = book_exposure(book)
    assert abs(e["gross"] - 1.0) < 1e-9
    assert abs(e["net"] - 0.4) < 1e-9
    assert abs(e["long"] - 0.7) < 1e-9
    assert abs(e["short"] - (-0.3)) < 1e-9
