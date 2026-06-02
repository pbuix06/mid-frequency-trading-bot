"""
Survivorship-free cross-sectional research harness.

The fixed-universe `run_research_xs` is fine for a handful of hand-picked
sleeves, but ranking cross-sectional momentum on ~20 names that all survived to
today is survivorship-biased: every company that went to zero is silently
excluded from the ranking, which inflates the apparent edge.

This harness fixes that:
  - Universe membership is decided POINT-IN-TIME at each rebalance using only
    trailing data (recent bar + trailing liquidity), so delisted names are in
    the universe while they were alive and leave it when they die.
  - No cross-death forward-fill: once a ticker's data ends, its prices stay NaN.
  - Delisting liquidation: a held name that loses eligibility is sold at its
    last known close at the next rebalance (its decline is realized, not erased).

Known caveat: daily data does not capture the final delisting gap (last traded
price -> 0). This slightly flatters longs that delist and the short side that
covers them; net effect is small and documented, not hidden.

This is research tooling (vectorbt track). Validation parity stays in the
event/Nautilus harnesses on the surviving fixed sleeves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mft.alphas.base import AlphaBase
from mft.data_layer.eodhd_ingest import load_ticker

# ── Panel construction ────────────────────────────────────────────────────────

def build_panels(
    tickers: list[str],
    pit_dir: Path,
    end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build aligned close and dollar-volume panels (columns=tickers, rows=dates).

    Critically does NOT forward-fill across a ticker's death: after a ticker's
    last real bar its columns stay NaN, so the simulator can detect delisting.
    Small intra-life holiday gaps are left as NaN too (the as-of checks tolerate
    a few missing days).
    """
    closes, dvols = {}, {}
    for t in tickers:
        try:
            df = load_ticker(t, pit_dir)
        except FileNotFoundError:
            continue
        if end_date is not None:
            df = df[df.index <= end_date]
        if df.empty:
            continue
        closes[t] = df["close"]
        dvols[t] = df["close"] * df["volume"]

    close_df = pd.DataFrame(closes).sort_index()
    dvol_df = pd.DataFrame(dvols).reindex(close_df.index)
    return close_df, dvol_df


def eligible_as_of(
    close_df: pd.DataFrame,
    dvol_df: pd.DataFrame,
    i: int,
    *,
    lookback: int,
    min_dollar_vol: float,
    liq_window: int = 63,
    recent_tol: int = 5,
) -> list[str]:
    """
    Tickers eligible to trade at row index i, using only data up to i.

    A ticker is eligible if:
      - it has a real (non-NaN) close within the last `recent_tol` bars
        (still trading — filters delisted names and phantom stale tails),
      - it has >= lookback non-NaN closes in its history up to i,
      - its trailing `liq_window` mean dollar volume >= min_dollar_vol.
    """
    if i < lookback:
        return []

    recent = close_df.iloc[max(0, i - recent_tol): i + 1]
    has_recent = recent.notna().any(axis=0)

    hist = close_df.iloc[: i + 1]
    enough_hist = hist.notna().sum(axis=0) >= lookback

    liq = dvol_df.iloc[max(0, i - liq_window): i + 1].mean(axis=0)
    liquid = liq >= min_dollar_vol

    mask = has_recent & enough_hist & liquid
    return [t for t in close_df.columns if bool(mask.get(t, False))]


# ── Result ──────────────────────────────────────────────────────────────────

@dataclass
class XSResult:
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_delistings: int = 0
    avg_universe_size: float = 0.0

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else float("nan")


# ── Simulator ─────────────────────────────────────────────────────────────────

def run_survivorship_xs(
    alpha: AlphaBase,
    close_df: pd.DataFrame,
    dvol_df: pd.DataFrame,
    *,
    rebalance_freq: int = 21,
    min_dollar_vol: float = 5_000_000,
    liq_window: int = 63,
    init_cash: float = 100_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.001,
    max_daily_return: float = 1.0,
) -> XSResult:
    """
    Weight-based, multiplicative-equity cross-sectional simulator.

    Why weight-based (not share-accounting): tracking share counts with per-name
    cash deltas blows up when shorting low-priced names crashing toward delisting
    (target_value / tiny_price -> enormous share counts). The standard, stable
    approach holds *weights* and compounds equity by the daily portfolio return
    r_p = Σ wᵢ·rᵢ, which is bounded and cannot produce negative equity for
    sane weights.

    Mechanics:
      - Target weights are set at each rebalance on the PIT-eligible universe.
      - Between rebalances weights DRIFT with realized returns (buy-and-hold
        between rebalances); turnover cost is charged only at rebalances on the
        traded weight change |w_target - w_drifted|.
      - A name that delists mid-period stops contributing returns; its decline
        up to delisting is realized. It is traded out at the next rebalance.
        (Caveat: the final last-price -> 0 gap is not modeled — daily data.)

    The alpha must be constructed with the FULL candidate universe; each
    rebalance it receives a window of only the currently-eligible columns.
    """
    lookback = alpha.lookback
    n = len(close_df)
    idx = close_df.index

    # Daily simple returns; NaN where a name has no price on either day.
    ret_df = close_df.pct_change(fill_method=None)

    equity = float(init_cash)
    weights: dict[str, float] = {}        # current (drifted) weights by ticker
    equity_hist: list[tuple[pd.Timestamp, float]] = [(idx[0], equity)]
    weight_hist: list[dict] = []
    n_delistings = 0
    universe_sizes: list[int] = []

    for i in range(1, n):
        ts = idx[i]
        rets_today = ret_df.iloc[i]

        # 1) Apply today's return to drifted weights -> portfolio return
        port_ret = 0.0
        if weights:
            new_w: dict[str, float] = {}
            for t, w in weights.items():
                r = rets_today.get(t, np.nan)
                r = 0.0 if (pd.isna(r) or abs(r) > max_daily_return) else float(r)
                port_ret += w * r
                new_w[t] = w * (1.0 + r)
            # Renormalize drifted weights by the gross they grew into
            gross = 1.0 + port_ret
            if gross > 0:
                weights = {t: wv / gross for t, wv in new_w.items()}
            else:
                weights = new_w
        equity *= (1.0 + port_ret)
        equity_hist.append((ts, equity))

        # 2) Rebalance?
        if i < lookback or (i - lookback) % rebalance_freq != 0:
            continue

        eligible = eligible_as_of(
            close_df, dvol_df, i,
            lookback=lookback, min_dollar_vol=min_dollar_vol, liq_window=liq_window,
        )

        # Count delistings: names we still hold that have died (no future bar)
        for t, w in weights.items():
            if w != 0.0 and t not in eligible and not _has_future_bar(close_df, t, i):
                n_delistings += 1

        if len(eligible) < 4:
            continue
        universe_sizes.append(len(eligible))

        window = close_df[eligible].iloc[i - lookback: i + 1]
        targets = alpha.compute_signal(window)
        if not targets:
            continue
        target_w = {t: v for t, v in targets.items() if v != 0.0}
        weight_hist.append({"ts": ts, **target_w})

        # Turnover cost on the traded change from drifted -> target weights
        names = set(weights) | set(target_w)
        turnover = sum(abs(target_w.get(t, 0.0) - weights.get(t, 0.0)) for t in names)
        cost = turnover * (commission_pct + slippage_pct)
        equity *= (1.0 - cost)
        equity_hist[-1] = (ts, equity)

        weights = target_w

    equity_s = pd.Series(
        [e for _, e in equity_hist],
        index=pd.DatetimeIndex([t for t, _ in equity_hist]),
        name="equity",
    )
    returns = equity_s.pct_change(fill_method=None).dropna()
    weights_df = pd.DataFrame(weight_hist).set_index("ts") if weight_hist else pd.DataFrame()

    return XSResult(
        equity=equity_s,
        returns=returns,
        weights=weights_df,
        n_delistings=n_delistings,
        avg_universe_size=float(np.mean(universe_sizes)) if universe_sizes else 0.0,
    )


def _has_future_bar(close_df: pd.DataFrame, ticker: str, i: int) -> bool:
    """True if `ticker` has any non-NaN close after row i (i.e. not yet dead)."""
    future = close_df[ticker].iloc[i + 1:]
    return bool(future.notna().any())
