"""
Live-monitor daily report (markdown) + the deviation / false-positive guard.

Restates, every time, that this is PAPER/OBSERVATION ONLY and that the breakout trigger is
a plumbing smoke test with ZERO expected edge — so a profitable paper run is flagged as a
SUSPECTED FALSE POSITIVE, never a discovery. Reuses mft.automation.monitor.Warning_ so the
warning vocabulary matches the research-automation framework.

Sections: signals triggered, signals rejected (by reason), paper trades, PnL after costs,
performance by symbol, performance by regime, data outages, deviation warnings.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from mft.automation.monitor import Warning_
from mft.automation.registry import LIVE_TRADING_APPROVED
from mft.live_monitor.paper_decision import LIVE_TRADING_ENABLED, PaperDecisionEngine
from mft.live_monitor.runner import LiveMonitor

ROOT = Path(__file__).parents[2]
REPORT_DIR = ROOT / "reports" / "live_monitor"
LEDGER_DIR = REPORT_DIR / "ledgers"
for _d in (REPORT_DIR, LEDGER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_BANNER = ("> ⚠️ **PAPER / OBSERVATION ONLY — NO LIVE TRADING.** No real orders are placed. "
           "The 24h-breakout trigger is a monitoring smoke test with **zero expected edge**; "
           "it does **not** revive any rejected strategy and does **not** change the research "
           "verdict (no production-ready alpha found). A profitable paper run here is a "
           "SUSPECTED FALSE POSITIVE, not a green light.")


def trades_frame(engine: PaperDecisionEngine) -> pd.DataFrame:
    if not engine.trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in engine.trades])


def deviation_warnings(engine: PaperDecisionEngine, fp_net_tol: float = 0.0) -> list[Warning_]:
    """Smoke-test expectation is net <= 0. Positive realized PnL ⇒ suspected false positive."""
    warns: list[Warning_] = []
    df = trades_frame(engine)
    if df.empty:
        warns.append(Warning_("info", "NO_TRADES", "no paper trades closed this run"))
        return warns
    net = float(df["net_pnl"].sum())
    if net > fp_net_tol:
        warns.append(Warning_("warn", "SUSPECTED_FALSE_POSITIVE",
                              f"breakout smoke test net +{net:.2f} after costs > expected (<=0). "
                              f"Zero-edge trigger showing profit = regime luck/overfit, NOT alpha. Do not act."))
    else:
        warns.append(Warning_("info", "STAYS_REJECTED",
                              f"net {net:+.2f} after costs <= 0 — consistent with the no-edge expectation."))
    return warns


def _by(df: pd.DataFrame, key: str) -> str:
    if df.empty or key not in df.columns:
        return "| (none) | | | |"
    g = df.groupby(key).agg(trades=("net_pnl", "size"), net=("net_pnl", "sum"),
                            gross=("gross_pnl", "sum"), cost=("cost", "sum"))
    return "\n".join(f"| {k} | {r.trades:.0f} | {r.net:+.2f} | {r.gross:+.2f} | {r.cost:.2f} |"
                     for k, r in g.iterrows())


def save_ledger(engine: PaperDecisionEngine, tag: str = "breakout_smoketest") -> Path:
    p = LEDGER_DIR / f"{date.today().isoformat()}_{tag}.parquet"
    df = trades_frame(engine)
    if not df.empty:
        df.to_parquet(p)
    return p


def daily_monitor_report(monitor: LiveMonitor, tag: str = "breakout_smoketest") -> Path:
    engine = monitor.paper
    df = trades_frame(engine)
    warns = deviation_warnings(engine)

    sig_df = pd.DataFrame(engine.signals) if engine.signals else pd.DataFrame()
    n_actionable = int(sig_df["is_trade"].sum()) if not sig_df.empty else 0
    rej = engine.rejected
    rej_reasons = Counter(r for row in rej for r in row["reasons"])

    n_trades = len(df)
    net = float(df["net_pnl"].sum()) if n_trades else 0.0
    gross = float(df["gross_pnl"].sum()) if n_trades else 0.0
    cost = float(df["cost"].sum()) if n_trades else 0.0
    win = float((df["net_pnl"] > 0).mean()) if n_trades else float("nan")
    outages = monitor.outage_log
    n_outage_min = len(outages)

    warns_md = "\n".join(f"- {w}" for w in warns) or "- (none)"
    rej_md = "\n".join(f"| {reason} | {cnt} |" for reason, cnt in rej_reasons.most_common()) or "| (none) | 0 |"
    outage_md = "\n".join(f"| {o['ts']} | {', '.join(o['missing'])} |" for o in outages[:25]) or "| (none) | |"

    body = f"""# Live-monitor daily report — {tag} — {date.today().isoformat()}

{_BANNER}

**Live approved:** {LIVE_TRADING_APPROVED} · **Live order path:** {LIVE_TRADING_ENABLED} (none exists) · \
**Minutes processed:** {len(monitor.minute_log)} · **Last BTC regime:** {monitor.last_regime}

## Activity
| metric | value |
|---|---:|
| symbols monitored | {len(monitor.cfg.symbols)} |
| signals evaluated | {len(engine.signals)} |
| actionable breakout signals | {n_actionable} |
| signals rejected by risk gate | {len(rej)} |
| paper trades closed | {n_trades} |
| open positions (end) | {len(engine.positions)} |

## Paper PnL after costs (simulated fills)
| metric | value |
|---|---:|
| net PnL | {net:+.2f} |
| gross PnL | {gross:+.2f} |
| total cost (fees+slippage) | {cost:.2f} |
| win rate | {win:.2%} |
| realized PnL (total) | {engine.realized_total:+.2f} |

## Rejected signals by reason
| reason | count |
|---|---:|
{rej_md}

## Performance by symbol
| symbol | trades | net | gross | cost |
|---|---:|---:|---:|---:|
{_by(df, 'symbol')}

## Performance by BTC regime (at entry)
| regime | trades | net | gross | cost |
|---|---:|---:|---:|---:|
{_by(df, 'regime_at_entry')}

## Data outages (subscribed symbol missing in a minute frame)
| minute | missing symbols |
|---|---|
{outage_md}
_({n_outage_min} minute(s) with at least one missing symbol.)_

## Deviation / false-positive guard
{warns_md}

_Ledger: `reports/live_monitor/ledgers/{date.today().isoformat()}_{tag}.parquet`. \
This report does not approve live trading and does not alter the research verdict._
"""
    p = REPORT_DIR / f"{date.today().isoformat()}_{tag}.md"
    p.write_text(body)
    return p
