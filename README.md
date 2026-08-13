# MFT — a mid-frequency trading **research** system

> A rigorous, honest quantitative research framework for finding and **falsifying** mid-frequency
> trading signals. It is **not** a profitable bot, and **no live trading is approved.** Its value is
> the discipline and the negative results: across ~130 logged trials and 8 signal families, it found
> **no production-ready alpha** — and proved it cleanly, cheaply, and reproducibly.

**Status:** 241 tests passing · 129 trials logged (`trials/trials.csv`, append-only) · 0 deployable
strategies · 0 capital at risk · live trading **NOT approved** (`docs/NO_LIVE_DEPLOYMENT.md`).

---

## What this project is

A from-scratch mid-frequency (minutes-to-days) systematic trading research engine, built solo, with
one organising principle: **finding a signal that looks profitable is easy and usually wrong; the
entire skill is proving whether it's real after costs, execution, and multiple-testing.** The
repository is the machine that does that proving, plus the documented results of pointing it at
daily equities, intraday equities, and crypto.

## The problem it solves

Most retail "backtests" lie — via look-ahead bias, survivorship bias, ignored transaction costs,
optimistic fills, and overfitting from unlogged search. This project is engineered to make those
lies impossible to tell by accident: one code path from research to paper, append-only trial
logging, hard-coded lock-boxes, pre-registered splits, regime validation, realistic cost models,
and a paper/forward-validation layer whose explicit job is to **flag false positives**.

## Headline verdict (honest)

**No production-ready alpha was found.** Every signal family accessible on free/retail data —
reversal, mean-reversion, breakouts, volume, lead-lag, maker execution, funding/positioning, and
momentum — was tested and **rejected**. The recurring killers were *transaction costs* (edges too
small at minute-to-hourly frequency), *adverse selection* (naive maker fills the wrong side), and
*regime confounding* (a "winner" that was just a crash artifact). Full detail:
[`docs/RESEARCH_VERDICT.md`](docs/RESEARCH_VERDICT.md).

## What was built (the durable deliverable)

- **Equity IC-first research engine** (`mft/research/`): panel → features → targets → signal_lab
  (IC / rank-IC / buckets / alpha-decay) → cross-sectional backtest → splits → reports, with
  strict past-only features and no-leakage tests.
- **Crypto data layer** (`mft/data_layer/`): Binance spot/perp/funding/OI providers + validation;
  **365 days of clean, validated minute data** for 10 USDT majors.
- **24/7 crypto backtest** (`xs_backtest.py` continuous mode): no equity-session logic.
- **Maker-fill simulator** (`maker_sim.py`): touch / trade-through / probabilistic fills + adverse
  selection — never assumes a fill without price touching the limit.
- **Funding-carry attribution** (`funding_backtest.py`): splits PnL into price vs funding.
- **Validation toolkit** (`mft/validation/`): Deflated Sharpe, CPCV, metrics, regime analysis.
- **Research/paper-trading automation** (`mft/automation/`): strategy registry, simulated-fill
  paper engine, a **false-positive monitor**, risk/cost monitor, and reporting — with a hard
  no-live governance gate.

## Repository layout

```
mft/
  research/      IC-first engine: panel, features, targets, signal_lab, xs_backtest, funding_backtest,
                 maker_sim, crypto_eval, crypto_panel, splits, report
  data_layer/    eodhd/alpaca/edgar ingest, crypto_provider (Binance), crypto_validate
  validation/    dsr (Deflated Sharpe), cpcv, metrics, diagnostics
  execution/     costs (single source of truth), intraday_sim, paper_trader, state
  automation/    registry, paper_engine, monitor, reporting, cli   (paper/research only — NO live)
  backtest/      vectorbt + event-driven + nautilus + survivorship harnesses
  alphas/, features/, portfolio/, risk/, monitoring/
scripts/         runnable studies + research_cli.py (the automation entry point)
tests/           241 tests (no-lookahead, parity, engine, crypto, funding, automation, ...)
docs/            PROJECT_SUMMARY, RESEARCH_VERDICT, HOW_TO_REPRODUCE, NO_LIVE_DEPLOYMENT, audits
RESEARCH_LOG.md  the lab notebook — every phase, finding, and rejection, top to bottom
trials/trials.csv  append-only trial ledger (never edited)
```

## Quickstart

```bash
# install (editable, with dev deps)
uv pip install -e ".[dev]"          # or: pip install -e ".[dev]"

# run the full test suite (241 tests)
pytest tests/ -q

# research/paper automation CLI (NO live trading)
python scripts/research_cli.py status            # governance gate + registry
python scripts/research_cli.py registry          # strategies + expected backtest stats
python scripts/research_cli.py validate-data     # data freshness + integrity
python scripts/research_cli.py paper-all --days 120   # simulate all watched strategies + weekly report
```

See [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md) for data ingestion and full reproduction.

## Why no live trading is approved

There is nothing to deploy: zero strategies survived out-of-sample + lock-box + regime validation +
realistic costs. The automation framework is for **research validation**, not trading — it has no
real-order code path, `registry.LIVE_TRADING_APPROVED` is `False`, and a `"live"` strategy status is
structurally impossible. A *rejected* strategy that looks good on recent data is treated as regime
luck and **flagged, never acted on**. See [`docs/NO_LIVE_DEPLOYMENT.md`](docs/NO_LIVE_DEPLOYMENT.md).

## Where the key logs and reports are

| Artifact | Location |
|---|---|
| Master lab notebook (all findings + rejections) | `RESEARCH_LOG.md` |
| Append-only trial ledger | `trials/trials.csv` |
| Per-study research logs | `research_logs/*.md` |
| Backtest result tables | `results/alpha_tests/*.csv`, `results/leaderboard.csv` |
| Figures | `results/figures/*.png` |
| Paper-trading reports + ledgers | `reports/paper/`, `reports/paper/ledgers/` |
| Weekly research reports | `reports/weekly/` |
| Project review & verdict | `docs/project_review_current_state.md`, `docs/RESEARCH_VERDICT.md` |

`results/` and `reports/` are generated artifacts and deliberately not committed — regenerate them
locally by following [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md).

## Documentation

- [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) — goal, outcome, components, lessons.
- [`docs/RESEARCH_VERDICT.md`](docs/RESEARCH_VERDICT.md) — the direct negative result.
- [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md) — environment, tests, data, reports.
- [`docs/NO_LIVE_DEPLOYMENT.md`](docs/NO_LIVE_DEPLOYMENT.md) — governance.
- [`docs/project_review_current_state.md`](docs/project_review_current_state.md) — sober checkpoint.

---

> **Disclaimer.** Research and educational project. Not investment advice. Not a profitable trading
> system. No live-trading readiness is claimed or implied. Historical/simulated results do not imply
> future performance. No real orders are placed by any code in this repository.
