"""IntradayReversal tests — direction, bounds, and look-ahead safety."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mft.alphas.intraday_reversal import IntradayReversal
from mft.data_layer.pit import make_pit_window


def _minutes(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2023-01-03 14:30", periods=n, freq="1min", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e4}, index=idx)


def test_fades_an_up_spike_short():
    # Flat then a sharp spike up -> z high -> SHORT
    closes = [100.0] * 40 + [105.0]
    df = _minutes(closes)
    alpha = IntradayReversal("AAPL", window=30, threshold=1.5, skip=0)
    sig = alpha.compute_signal(df.iloc[-alpha.lookback - 1:])
    assert sig["AAPL"] == -1.0


def test_fades_a_down_spike_long():
    closes = [100.0] * 40 + [95.0]
    df = _minutes(closes)
    alpha = IntradayReversal("AAPL", window=30, threshold=1.5, skip=0)
    sig = alpha.compute_signal(df.iloc[-alpha.lookback - 1:])
    assert sig["AAPL"] == 1.0


def test_flat_when_within_band():
    rng = np.random.default_rng(0)
    closes = list(100 + rng.normal(0, 0.05, 60))  # tiny noise -> |z| stays small
    df = _minutes(closes)
    alpha = IntradayReversal("AAPL", window=30, threshold=3.0)
    sig = alpha.compute_signal(df.iloc[-alpha.lookback - 1:])
    assert sig["AAPL"] == 0.0


def test_signal_in_bounds():
    rng = np.random.default_rng(1)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 200))))
    df = _minutes(closes)
    alpha = IntradayReversal("AAPL")
    sig = alpha.compute_signal(df.iloc[-alpha.lookback - 1:])
    assert sig["AAPL"] in (-1.0, 0.0, 1.0)


def test_no_look_ahead():
    rng = np.random.default_rng(2)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 300))))
    df = _minutes(closes)
    alpha = IntradayReversal("AAPL")
    as_of = df.index[200]
    w = make_pit_window(df, as_of, alpha.lookback)
    sig_before = alpha.compute_signal(w)
    poisoned = df.copy()
    poisoned.loc[df.index[201:], "close"] = 1e6
    w2 = make_pit_window(poisoned, as_of, alpha.lookback)
    assert alpha.compute_signal(w2) == sig_before
