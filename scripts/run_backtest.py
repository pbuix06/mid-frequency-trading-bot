"""
Phase 3 research backtest runner.

Runs a single alpha on one or more tickers, prints the metric vector, and
appends one row to trials/trials.csv. Research runs are logged; --dry-run is
diagnostic plumbing only and must not be treated as research evidence.

Usage:
    # Single-asset alphas (TSMomentum, ShortReversion, SMACrossover):
    python scripts/run_backtest.py --alpha TSMomentum --tickers SPY
    python scripts/run_backtest.py --alpha ShortReversion --tickers AAPL --params '{"window": 10}'

    # Cross-sectional alpha (XSMomentum) — pass a universe:
    python scripts/run_backtest.py --alpha XSMomentum --tickers SPY QQQ IWM DIA XLK XLF XLE

    # Diagnostic dry run (print metrics but do NOT log trial/evidence):
    python scripts/run_backtest.py --alpha TSMomentum --tickers SPY --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from mft.alphas import (
    LongShortMomentum,
    LowVolAnomaly,
    PairsMeanReversion,
    ShortReversion,
    SMACrossover,
    TSMomentum,
    XSMomentum,
)
from mft.backtest.vectorbt_harness import run_research_xs
from mft.data_layer.eodhd_ingest import LOCKBOX_CUTOFF, load_ticker
from mft.monitoring.trial_log import TrialLog
from mft.validation.dsr import min_backtest_length
from mft.validation.metrics import full_metrics

PIT_DIR = Path(__file__).parents[1] / "data" / "pit"
TRIALS_FILE = Path(__file__).parents[1] / "trials" / "trials.csv"

ALPHA_REGISTRY = {
    "SMACrossover": SMACrossover,
    "TSMomentum": TSMomentum,
    "ShortReversion": ShortReversion,
    "XSMomentum": XSMomentum,
    "LongShortMomentum": LongShortMomentum,
    "PairsMeanReversion": PairsMeanReversion,
    "LowVolAnomaly": LowVolAnomaly,
}

SINGLE_ASSET_ALPHAS = {"SMACrossover", "TSMomentum", "ShortReversion"}
PAIRS_ALPHAS = {"PairsMeanReversion"}


def load_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for t in tickers:
        path = PIT_DIR / f"{t}.parquet"
        if not path.exists():
            print(f"  [warn] {t} not found in data/pit/ — skipping")
            continue
        df = load_ticker(t, PIT_DIR)
        if df.empty:
            print(f"  [warn] {t} has no data — skipping")
            continue
        data[t] = df
    return data


def build_alpha(alpha_name: str, tickers: list[str], params: dict):
    cls = ALPHA_REGISTRY[alpha_name]
    if alpha_name in ("XSMomentum", "LowVolAnomaly", "LongShortMomentum"):
        return cls(universe=tickers, **params)
    elif alpha_name == "PairsMeanReversion":
        if len(tickers) < 2:
            raise ValueError("PairsMeanReversion requires exactly 2 tickers")
        return cls(
            asset1=tickers[0], asset2=tickers[1],
            lookback=params.get("lookback", 252),
            z_entry=params.get("z_entry", 1.5),
            z_exit=params.get("z_exit", 0.5),
            z_stop=params.get("z_stop", 3.5),
        )
    elif alpha_name == "SMACrossover":
        return cls(symbol=tickers[0], fast=params.get("fast", 20), slow=params.get("slow", 50))
    elif alpha_name == "TSMomentum":
        return cls(
            symbol=tickers[0],
            lookback=params.get("lookback", 252),
            skip=params.get("skip", 21),
            vol_window=params.get("vol_window", 63),
            target_vol=params.get("target_vol", 0.15),
        )
    elif alpha_name == "ShortReversion":
        return cls(
            symbol=tickers[0],
            window=params.get("window", 5),
            threshold=params.get("threshold", 1.0),
        )
    raise ValueError(f"Unknown alpha: {alpha_name}")


def run_single_asset(alpha, data: dict, ticker: str, args) -> dict:
    from mft.backtest.event_harness import equity_to_returns, run_event_driven

    df = data[ticker]
    research_df = df if args.end_date is None else df[df.index <= args.end_date]
    if research_df.empty:
        raise ValueError(f"{ticker} has no bars inside the selected research window")
    ev_state = run_event_driven(
        alpha, df, symbol=ticker,
        slippage_pct=args.slippage, commission_pct=args.commission,
        end_date=args.end_date,
    )
    returns = equity_to_returns(ev_state)
    metrics = full_metrics(returns)
    data_window = f"{research_df.index[0].date()}:{research_df.index[-1].date()}"
    return {**metrics, "data_window": data_window, "n_fills": len(ev_state.fills)}


def run_xs(alpha, data: dict, args) -> dict:
    result = run_research_xs(
        alpha, data,
        rebalance_freq=args.rebalance_freq,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        end_date=args.end_date,
    )
    metrics = full_metrics(result["returns"])
    if len(result["returns"]):
        data_window = f"{result['returns'].index[0].date()}:{result['returns'].index[-1].date()}"
    else:
        start = min(df.index[0] for df in data.values())
        end = args.end_date if args.end_date is not None else max(df.index[-1] for df in data.values())
        data_window = f"{pd.Timestamp(start).date()}:{pd.Timestamp(end).date()}"
    return {**metrics, "data_window": data_window, "n_fills": 0}


def print_metrics(alpha_name: str, tickers: list[str], m: dict, params: dict):
    print(f"\n{'='*60}")
    print(f"  Alpha:    {alpha_name}")
    print(f"  Tickers:  {tickers}")
    print(f"  Params:   {params}")
    print(f"  Window:   {m.get('data_window', '?')}")
    print(f"{'='*60}")
    print(f"  Sharpe:      {m.get('sharpe', float('nan')):.4f}  (target 0.5–1.5)")
    print(f"  Sortino:     {m.get('sortino', float('nan')):.4f}")
    print(f"  Calmar:      {m.get('calmar', float('nan')):.4f}")
    print(f"  Max DD:      {m.get('max_drawdown', float('nan')):.2%}")
    print(f"  CAGR:        {m.get('cagr', float('nan')):.2%}")
    print(f"  Hit rate:    {m.get('hit_rate', float('nan')):.2%}")
    sharpe = m.get("sharpe", 0)
    if sharpe > 2:
        print(f"\n  ⚠  Sharpe {sharpe:.2f} > 2 — check for bug or overfit")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", required=True, choices=list(ALPHA_REGISTRY))
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--params", default="{}", help="JSON param overrides")
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--commission", type=float, default=0.001)
    p.add_argument("--rebalance-freq", type=int, default=21, dest="rebalance_freq")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Diagnostic only: print metrics but skip trial log",
    )
    p.add_argument("--include-lockbox", action="store_true",
                   help="DANGER: use data past LOCKBOX_CUTOFF. Only for the Phase 4 final exam.")
    args = p.parse_args()

    # Lock-box enforced by default. Every research run stops at LOCKBOX_CUTOFF
    # unless --include-lockbox is explicitly passed (the final exam, once).
    args.end_date = None if args.include_lockbox else LOCKBOX_CUTOFF
    if args.include_lockbox:
        print("\n  ⚠  --include-lockbox set: USING LOCK-BOX DATA. This is the final exam.")
        print("     Do NOT use this for ordinary research.\n")
    else:
        print(f"\n  Lock-box enforced: data truncated at {LOCKBOX_CUTOFF.date()}")

    params = json.loads(args.params)
    print(f"\nLoading data for: {args.tickers}")
    data = load_data(args.tickers)
    if not data:
        print("No data found. Run scripts/ingest_universe.py first.")
        sys.exit(1)

    alpha = build_alpha(args.alpha, list(data.keys()), params)
    print(f"Alpha: {alpha}")

    if args.alpha in SINGLE_ASSET_ALPHAS:
        ticker = list(data.keys())[0]
        m = run_single_asset(alpha, data, ticker, args)
    elif args.alpha in PAIRS_ALPHAS:
        m = run_xs(alpha, data, args)   # run_research_xs handles long/short weights
    else:
        m = run_xs(alpha, data, args)

    print_metrics(args.alpha, list(data.keys()), m, params)

    if args.dry_run:
        print("  [dry-run] Trial NOT logged.")
        print("            Diagnostic only; do not treat these metrics as research evidence.")
        return

    # Log trial
    log = TrialLog(TRIALS_FILE)
    trial_id = log.log(
        strategy=args.alpha,
        asset_universe=list(data.keys()),
        params=params or {"default": True},
        data_window=m.get("data_window", ""),
        is_sharpe=round(m.get("sharpe", float("nan")), 4),
        max_dd=round(m.get("max_drawdown", float("nan")), 4),
        notes=f"Phase 3 research — {args.alpha}",
    )
    n_trials = log.count()
    min_bars = min_backtest_length(n_trials)
    min_years = min_bars / 252
    print(f"  Trial {trial_id} logged to trials/trials.csv")
    print(
        f"  MinBTL: {n_trials} trials used; "
        f"need at least {min_bars} bars ({min_years:.2f} years) of data\n"
    )


if __name__ == "__main__":
    main()
