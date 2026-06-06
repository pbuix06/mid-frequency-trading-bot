"""
Reporting — daily paper-trading report + weekly research report (markdown).

Every report restates that NO live deployment is approved and that paper results that look good are
treated as suspected false positives. Performance is broken down by strategy and regime, with the
signal-vs-realized (IC) check and all monitor warnings surfaced front-and-centre.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from mft.automation.monitor import Warning_
from mft.automation.paper_engine import PaperResult
from mft.automation.registry import LIVE_TRADING_APPROVED, StrategySpec

ROOT = Path(__file__).parents[2]
PAPER_DIR = ROOT / "reports" / "paper"
WEEKLY_DIR = ROOT / "reports" / "weekly"
LEDGER_DIR = ROOT / "reports" / "paper" / "ledgers"
for _d in (PAPER_DIR, WEEKLY_DIR, LEDGER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_BANNER = ("> ⚠️ **PAPER / RESEARCH ONLY — NO LIVE TRADING.** No real orders are placed. "
           "This project found no production-ready alpha; every strategy here is REJECTED and "
           "paper-watched to detect false positives. A good-looking paper result is a red flag, not a green light.")


def save_ledger(result: PaperResult) -> Path:
    p = LEDGER_DIR / f"{date.today().isoformat()}_{result.strategy}.parquet"
    if not result.ledger.empty:
        result.ledger.to_parquet(p)
    return p


def daily_paper_report(spec: StrategySpec, result: PaperResult, warns: list[Warning_],
                       cost_mode: str) -> Path:
    s = result.summary
    warns_md = "\n".join(f"- {w}" for w in warns) or "- (none)"
    regime_md = "\n".join(f"| {k} | {v:+.2f} |" for k, v in s.get("net_bps_by_regime", {}).items()) or "| n/a | n/a |"
    body = f"""# Daily paper report — {spec.name} — {date.today().isoformat()}

{_BANNER}

**Status:** `{spec.status}` · **Verdict (backtest):** {spec.expected.get('verdict')} · **Trials:** {spec.trial_ids}
**Live approved:** {LIVE_TRADING_APPROVED} · **Fill mode:** {cost_mode}

## Paper summary (simulated fills)
| metric | paper | backtest expected |
|---|---:|---:|
| net bps/trade | {s.get('mean_net_bps', float('nan')):+.2f} | {spec.expected.get('net_bps_5', float('nan')):+.2f} |
| gross bps/trade | {s.get('mean_gross_bps', float('nan')):+.2f} | {spec.expected.get('gross_bps', float('nan')):+.2f} |
| funding bps/trade | {s.get('mean_funding_bps', float('nan')):+.2f} | — |
| cost bps/trade | {s.get('mean_cost_bps', float('nan')):+.2f} | — |
| IC | {s.get('ic', float('nan')):+.4f} | {spec.expected.get('ic', float('nan')):+.4f} |
| win rate | {s.get('win_rate', float('nan')):.2%} | — |
| hypothetical trades | {s.get('n_trades', 0)} | — |

## Net bps/trade by BTC regime
| regime | net bps |
|---|---:|
{regime_md}

## Monitor warnings (false-positive / data / cost)
{warns_md}

## Signal-vs-realized
IC paper {s.get('ic', float('nan')):+.4f} vs backtest {spec.expected.get('ic', float('nan')):+.4f}. A sign flip or
collapse means the effect is unstable. A paper net materially above the (negative) backtest is a SUSPECTED
FALSE POSITIVE — recorded, never acted on.

_Ledger: `reports/paper/ledgers/{date.today().isoformat()}_{spec.name}.parquet` (every hypothetical trade:
signal, entry, exit, cost, regime, reason)._
"""
    p = PAPER_DIR / f"{date.today().isoformat()}_{spec.name}.md"
    p.write_text(body)
    return p


def weekly_research_report(results: dict[str, PaperResult], warns_by: dict[str, list[Warning_]]) -> Path:
    rows = []
    for name, res in results.items():
        s = res.summary
        rows.append({"strategy": name, "status_net_bps": s.get("mean_net_bps", float("nan")),
                     "gross_bps": s.get("mean_gross_bps", float("nan")), "ic": s.get("ic", float("nan")),
                     "trades": s.get("n_trades", 0),
                     "warnings": len([w for w in warns_by.get(name, []) if w.level in ("warn", "critical")])})
    df = pd.DataFrame(rows)
    tbl = df.to_markdown(index=False, floatfmt=".3f") if not df.empty else "(no strategies run)"
    fp = [str(w) for ws in warns_by.values() for w in ws if w.code == "SUSPECTED_FALSE_POSITIVE"]
    fp_md = "\n".join(f"- {w}" for w in fp) or "- (none — all paper results consistent with REJECTED verdicts)"
    body = f"""# Weekly research report — {date.today().isoformat()}

{_BANNER}

## Paper performance by strategy
{tbl}

## Suspected false positives (paper >> rejected backtest)
{fp_md}

## Interpretation
All registered strategies are REJECTED; the healthy outcome is paper net ≤ 0 (consistent with backtest).
Any positive paper drift is scrutinised as regime luck/overfit, not edge. No strategy is a candidate for
live deployment; `LIVE_TRADING_APPROVED = {LIVE_TRADING_APPROVED}`.

See `docs/NO_LIVE_DEPLOYMENT.md` and `docs/project_review_current_state.md`.
"""
    p = WEEKLY_DIR / f"{date.today().isoformat()}_weekly.md"
    p.write_text(body)
    return p
