# How to reproduce

Everything below is research / paper / simulation. **No command in this repository places a real
order or enables live trading.**

## 1. Environment setup

Requirements: Python 3.13, a virtual environment, and the package installed editable.

```bash
cd mid-frequency-trading-bot

# create + activate a venv (the repo uses .venv/)
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# install the package with dev dependencies
uv pip install -e ".[dev]"           # or: pip install -e ".[dev]"
```

Notes:
- Activate the venv before running anything (`python` then resolves to the venv interpreter).
- Crypto ingestion needs outbound network access to Binance public endpoints (no API key required).
  Some networks/regions geo-block `api.binance.com`; if so, ingestion will fail and you can still run
  the tests and the CLI's non-data commands.

## 2. Run the tests (225)

```bash
pytest tests/ -q                     # full suite
pytest tests/test_look_ahead.py -v   # the most important test (no look-ahead leakage)
pytest tests/test_automation.py -q   # the paper/governance framework
```

A green suite confirms the engine, the no-leakage guarantees, and the no-live governance gate.

## 3. Ingest / update crypto data

Fetches Binance spot + perp minute bars, funding, and open interest into `data/crypto/`, validates
them, and regenerates `docs/crypto_data_audit.md`.

```bash
python scripts/ingest_crypto.py --days 30        # quick: last 30 days
python scripts/ingest_crypto.py --days 365       # full backfill used in the research (~25–35 min)

# or via the CLI wrapper:
python scripts/research_cli.py update-data --days 365
```

Storage: `data/crypto/{spot_1m,perp_1m,funding,open_interest,metadata}/` (git-ignored).
Open interest is limited to ~30 days by the Binance API (documented in the audit).

## 4. Validate data

```bash
python scripts/research_cli.py validate-data
```

Reports per-symbol bar counts, coverage %, gaps, OHLC integrity, last-bar age, and staleness.

## 5. Run the paper / research automation

```bash
python scripts/research_cli.py status                    # governance gate + registry overview
python scripts/research_cli.py registry                  # strategies + expected backtest stats
python scripts/research_cli.py slippage                  # cost / slippage assumptions

# simulate one watched strategy forward (simulated fills only):
python scripts/research_cli.py paper-run --strategy crypto_momentum_24h --days 120

# simulate ALL watched strategies and write the weekly research report:
python scripts/research_cli.py paper-all --days 120
```

Each paper run records every hypothetical trade (signal, entry, exit, cost, regime, reason) to a
ledger and runs the data, false-positive, and risk/cost monitors. A rejected strategy that looks good
is flagged `SUSPECTED_FALSE_POSITIVE` and never acted on.

## 6. Generate the weekly report

`paper-all` writes the weekly research report automatically. It aggregates paper performance by
strategy, surfaces any suspected false positives, and restates that no live deployment is approved:

```bash
python scripts/research_cli.py paper-all --days 120
# -> reports/weekly/<date>_weekly.md
```

## 7. Re-run the original research studies (optional)

The pre-registered studies live in `scripts/` and write to `results/` and `research_logs/`:

```bash
python scripts/run_crypto_funding_backfill.py    # funding reversal — 365d OOS / regime validation
python scripts/run_crypto_momentum.py            # cross-sectional momentum — 365d OOS / regime
python scripts/run_reversal_research.py          # equity intraday residual reversal (needs data/intraday/)
```

Each appends one row per config to the append-only ledger `trials/trials.csv` (never edit it).

## 8. Where outputs are saved

| Output | Path |
|---|---|
| Master lab notebook | `RESEARCH_LOG.md` |
| Append-only trial ledger | `trials/trials.csv` |
| Per-study research logs | `research_logs/*.md` |
| Result tables + leaderboard | `results/alpha_tests/*.csv`, `results/leaderboard.csv` |
| Figures | `results/figures/*.png` |
| Crypto data audit | `docs/crypto_data_audit.md` |
| Paper reports + per-trade ledgers | `reports/paper/`, `reports/paper/ledgers/` |
| Weekly research reports | `reports/weekly/` |

## 9. What you cannot do

There is intentionally no command to place an order, connect a broker, or enable live trading.
`mft.automation.registry.LIVE_TRADING_APPROVED` is `False`, a `"live"` strategy status is rejected by
construction, and there is no order-execution code path in `mft/automation/`. See
`docs/NO_LIVE_DEPLOYMENT.md`.
