"""
Run the crypto live-monitor in PAPER mode (no live trading, no real orders).

Sources:
  --source replay   : replay stored 1m bars (data/crypto/spot_1m or perp_1m) minute by
                      minute. Offline, deterministic, the safe default. Use --days to bound.
  --source rest     : near-live. Poll Binance public REST for the latest CLOSED 1m kline
                      per symbol (+ bookTicker spread). Network, read-only. --minutes to bound.
  --source websocket: seconds-level. Public kline_1m/bookTicker websocket -> 1m bars. Needs
                      the optional `websocket-client` package; falls back message points to rest.

Always: governance gate runs first (LIVE_TRADING_APPROVED must be False), every fill is
simulated, and a daily markdown report + parquet ledger are written under reports/live_monitor/.

Examples:
    python scripts/run_live_monitor.py --source replay --days 7
    python scripts/run_live_monitor.py --source rest --minutes 120
    python scripts/run_live_monitor.py --source replay --symbols BTCUSDT ETHUSDT SOLUSDT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.automation.registry import CRYPTO_MAJORS, assert_no_live_approved  # noqa: E402
from mft.live_monitor.feature_stream import FeatureConfig  # noqa: E402
from mft.live_monitor.report import daily_monitor_report, save_ledger  # noqa: E402
from mft.live_monitor.runner import LiveMonitor, LiveMonitorConfig  # noqa: E402
from mft.live_monitor.websocket_client import (  # noqa: E402
    BinanceRestPoller,
    BinanceWebSocketClient,
    ReplayStream,
)

SPOT_1M = ROOT / "data" / "crypto" / "spot_1m"
PERP_1M = ROOT / "data" / "crypto" / "perp_1m"


def _load_replay(symbols: list[str], days: int | None) -> ReplayStream:
    src_dir = SPOT_1M if (SPOT_1M / f"{symbols[0]}.parquet").exists() else PERP_1M
    bars: dict[str, pd.DataFrame] = {}
    for s in symbols:
        p = src_dir / f"{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()
        if days:
            df = df[df.index >= df.index[-1] - pd.Timedelta(days=days)]
        bars[s] = df
    if not bars:
        raise SystemExit(f"No crypto bars under {src_dir}. Run: python scripts/ingest_crypto.py --days 30")
    return ReplayStream(bars)


def main(argv=None):
    assert_no_live_approved()
    ap = argparse.ArgumentParser(prog="run-live-monitor", description="Crypto live monitor (PAPER ONLY).")
    ap.add_argument("--source", choices=["replay", "rest", "websocket"], default="replay")
    ap.add_argument("--symbols", nargs="+", default=list(CRYPTO_MAJORS[:3]))
    ap.add_argument("--days", type=int, default=7, help="replay lookback window")
    ap.add_argument("--minutes", type=int, default=120, help="rest/websocket: minutes to run")
    ap.add_argument("--quiet", action="store_true", help="suppress per-trade stdout")
    args = ap.parse_args(argv)

    # For replay over a short --days window, shrink warmup so the smoke test can actually fire.
    feat = FeatureConfig()
    if args.source == "replay" and args.days is not None and args.days < 2:
        feat = FeatureConfig(min_history_bars=120)

    cfg = LiveMonitorConfig(symbols=args.symbols, btc_symbol="BTCUSDT" if "BTCUSDT" in args.symbols else args.symbols[0],
                            feature=feat)
    monitor = LiveMonitor(cfg)

    if args.source == "replay":
        stream = _load_replay(args.symbols, args.days)
    elif args.source == "rest":
        stream = BinanceRestPoller(args.symbols, max_minutes=args.minutes)
    else:
        stream = BinanceWebSocketClient(args.symbols, max_minutes=args.minutes)

    print(f"=== LIVE MONITOR (PAPER ONLY) | source={args.source} | symbols={args.symbols} ===")
    print("    LIVE_TRADING_APPROVED=False · no real orders · simulated fills only\n")

    def on_step(res):
        if args.quiet:
            return
        if res.signals_fired or res.outages:
            tag = []
            if res.signals_fired:
                tag.append(f"signals={res.signals_fired}")
            if res.outages:
                tag.append(f"OUTAGE={res.outages}")
            print(f"  {res.now}  regime={res.regime:<8} open={res.n_trades_open}  " + "  ".join(tag))

    monitor.run(stream, on_step=on_step)

    eng = monitor.paper
    print(f"\nProcessed {len(monitor.minute_log)} minutes | signals evaluated {len(eng.signals)} | "
          f"actionable {sum(s['is_trade'] for s in eng.signals)} | rejected {len(eng.rejected)} | "
          f"paper trades {len(eng.trades)}")
    print(f"Net paper PnL after costs: {eng.realized_total:+.2f} (expected <= 0 for a zero-edge smoke test)")
    ledger = save_ledger(eng)
    report = daily_monitor_report(monitor)
    print(f"Ledger -> {ledger}")
    print(f"Report -> {report}   (LIVE NOT APPROVED — research verdict unchanged)")


if __name__ == "__main__":
    main()
