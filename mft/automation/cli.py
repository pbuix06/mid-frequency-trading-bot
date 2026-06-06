"""
Research automation CLI — orchestrates the registry, paper engine, monitors, and reports.

Subcommands:
  status         governance gate + registry overview
  registry       list registered strategies + expected backtest stats
  slippage       show cost/slippage assumptions
  validate-data  run data-integrity checks on stored crypto perp bars
  paper-run      simulate ONE strategy forward (simulated fills), report + monitor warnings
  paper-all      paper-run all `paper_watch` strategies, emit the weekly research report
  update-data    (network) refresh crypto data via scripts/ingest_crypto.py

NO subcommand can place an order or enable live trading. `assert_no_live_approved()` runs first, always.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from mft.automation import registry as REG  # noqa: E402
from mft.automation.monitor import DataMonitor, DeviationMonitor, RiskCostMonitor  # noqa: E402
from mft.automation.paper_engine import PaperConfig, run_paper  # noqa: E402
from mft.automation.reporting import (  # noqa: E402
    daily_paper_report,
    save_ledger,
    weekly_research_report,
)


def _load_crypto(days: int | None):
    from mft.research.crypto_eval import regimes
    from mft.research.crypto_panel import (
        FUNDING_DIR,
        PERP_1M,
        cumulative_funding_series,
        funding_zscore_series,
        load_crypto_panel,
        load_funding,
        map_to_bars,
    )
    syms = list(REG.CRYPTO_MAJORS)
    panel = load_crypto_panel(syms, freq="5min", market="BTCUSDT", spot_dir=PERP_1M)
    close = panel.to_wide("close", syms)
    if days:
        close = close[close.index >= close.index[-1] - pd.Timedelta(days=days)]
    idx = close.index
    funding = {s: load_funding(s, FUNDING_DIR) for s in syms}
    fz = pd.DataFrame({s: map_to_bars(funding_zscore_series(funding[s], 10), idx) for s in syms}).reindex(idx)
    cum = pd.DataFrame({s: map_to_bars(cumulative_funding_series(funding[s]), idx) for s in syms}).reindex(idx)
    return close, fz, cum, regimes(close)


def _crypto_ready() -> bool:
    from mft.research.crypto_panel import PERP_1M
    return (PERP_1M / "BTCUSDT.parquet").exists()


def cmd_status(_):
    REG.assert_no_live_approved()
    print("=== Research automation — governance status ===")
    print(f"LIVE_TRADING_APPROVED : {REG.LIVE_TRADING_APPROVED}  (must be False)")
    print(f"Registered strategies : {len(REG.all_specs())} — all status in {sorted(REG.VALID_STATUS)}")
    print("Live deployment       : NOT APPROVED. Paper/research only. See docs/NO_LIVE_DEPLOYMENT.md")
    for s in REG.all_specs().values():
        print(f"  - {s.name:<28} [{s.status}] verdict={s.expected.get('verdict')}")


def cmd_registry(_):
    for s in REG.all_specs().values():
        e = s.expected
        print(f"{s.name:<28} {s.asset_class:<6} {s.family:<16} status={s.status:<11} "
              f"exp_ic={e.get('ic'):+.4f} exp_net5={e.get('net_bps_5'):+.2f}bps  trials={s.trial_ids}")


def cmd_slippage(_):
    rc = RiskCostMonitor()
    for k, v in rc.slippage_assumptions().items():
        print(f"  {k}: {v}")
    print(f"  cost floors (bps/side): {rc.COST_FLOOR_BPS}")


def cmd_validate_data(args):
    if not _crypto_ready():
        print("No crypto data. Run: python scripts/ingest_crypto.py --days 30")
        return
    close, *_ = _load_crypto(args.days)
    dm = DataMonitor()
    from mft.research.crypto_panel import PERP_1M, _load_parquet
    print(f"Validating {len(REG.CRYPTO_MAJORS)} perp symbols ...")
    for s in REG.CRYPTO_MAJORS:
        bars = _load_parquet(PERP_1M / f"{s}.parquet")
        rep, warns = dm.validate(bars)
        fresh = dm.freshness(bars)
        flag = "CLEAN" if not warns else ",".join(w.code for w in warns)
        print(f"  {s:<9} bars={rep['n_bars']:>7} cov={rep['coverage_pct']}% "
              f"last={fresh['last_bar'][:16]} age={fresh['age_minutes']}m {'STALE' if fresh['stale'] else ''} [{flag}]")


def _run_one(name, args, close, fz, cum, reg):
    spec = REG.get(name)
    cfg = PaperConfig(cost_bps_per_side=args.cost, mode=args.mode)
    res = run_paper(spec, close, reg, cfg, funding_z=fz, cum_funding=cum)
    warns = (RiskCostMonitor().check_cost_assumption(spec.asset_class, args.cost)
             + RiskCostMonitor().check_exposure(res.ledger)
             + DeviationMonitor().compare(spec, res.summary))
    save_ledger(res)
    daily_paper_report(spec, res, warns, args.mode)
    return spec, res, warns


def cmd_paper_run(args):
    if not _crypto_ready():
        print("No crypto data. Run: python scripts/ingest_crypto.py --days 30")
        return
    close, fz, cum, reg = _load_crypto(args.days)
    spec, res, warns = _run_one(args.strategy, args, close, fz, cum, reg)
    s = res.summary
    print(f"=== PAPER (simulated) {spec.name} | mode={args.mode} {args.cost}bps/side | {s['n_trades']} trades ===")
    print(f"  net {s['mean_net_bps']:+.2f} (gross {s['mean_gross_bps']:+.2f} + fund {s['mean_funding_bps']:+.2f} "
          f"- cost {s['mean_cost_bps']:.2f}) bps/trade | IC {s['ic']:+.4f} | backtest expected net "
          f"{spec.expected.get('net_bps_5'):+.2f}, IC {spec.expected.get('ic'):+.4f}")
    print(f"  by regime: {s.get('net_bps_by_regime')}")
    print("  warnings:")
    for w in warns:
        print(f"    {w}")
    print(f"  -> reports/paper/{pd.Timestamp.now().date()}_{spec.name}.md  (LIVE NOT APPROVED)")


def cmd_paper_all(args):
    if not _crypto_ready():
        print("No crypto data. Run: python scripts/ingest_crypto.py --days 30")
        return
    close, fz, cum, reg = _load_crypto(args.days)
    results, warns_by = {}, {}
    for name, spec in REG.all_specs().items():
        if spec.status != "paper_watch":
            continue
        _, res, warns = _run_one(name, args, close, fz, cum, reg)
        results[name], warns_by[name] = res, warns
        fp = [w for w in warns if w.code == "SUSPECTED_FALSE_POSITIVE"]
        print(f"  {name:<28} net {res.summary['mean_net_bps']:+.2f} bps/trade  IC {res.summary['ic']:+.4f}"
              f"  {'⚠ FALSE-POSITIVE FLAG' if fp else 'consistent w/ REJECTED'}")
    p = weekly_research_report(results, warns_by)
    print(f"\nWeekly research report -> {p}  (no live deployment approved)")


def cmd_update_data(args):
    print(f"Refreshing crypto data (network) — scripts/ingest_crypto.py --days {args.days} ...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "ingest_crypto.py"), "--days", str(args.days)], check=False)


def main(argv=None):
    REG.assert_no_live_approved()
    ap = argparse.ArgumentParser(prog="research-cli", description="Paper/research automation (NO live trading).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("registry").set_defaults(fn=cmd_registry)
    sub.add_parser("slippage").set_defaults(fn=cmd_slippage)
    p = sub.add_parser("validate-data")
    p.add_argument("--days", type=int, default=None)
    p.set_defaults(fn=cmd_validate_data)
    p = sub.add_parser("paper-run")
    p.add_argument("--strategy", required=True)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--mode", choices=["taker", "maker_optimistic"], default="taker")
    p.add_argument("--cost", type=float, default=5.0)
    p.set_defaults(fn=cmd_paper_run)
    p = sub.add_parser("paper-all")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--mode", choices=["taker", "maker_optimistic"], default="taker")
    p.add_argument("--cost", type=float, default=5.0)
    p.set_defaults(fn=cmd_paper_all)
    p = sub.add_parser("update-data")
    p.add_argument("--days", type=int, default=30)
    p.set_defaults(fn=cmd_update_data)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
