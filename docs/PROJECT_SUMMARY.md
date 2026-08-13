# Project summary

## Original goal

Build a **mid-frequency trading model / bot** — a systematic strategy holding positions from minutes
to a few days — and take it from research through validation toward (eventually) live deployment.
The intended path: build an honest research engine, find economically-motivated alpha signals,
validate them rigorously, then paper-trade and stage to live.

## Final outcome

**No deployable alpha was found.** After ~130 logged trials across daily equities, intraday
equities, and crypto, every tested signal family was rejected. The project never reached live
deployment — not because the machinery failed, but because the machinery worked: it repeatedly,
honestly demonstrated that the candidate edges were not tradable. **Live trading is not approved and
no real capital was ever at risk.**

This is a *successful* research outcome in the scientific sense — a thorough, well-documented set of
negative results produced under strict anti-self-deception discipline — even though it is not a
profitable product.

## Durable deliverable

A **rigorous, reusable quant research framework plus a paper/forward-validation system**, and a
complete, honest record of what was tested and why it was rejected. Concretely:

- A research engine that turns a new signal into a validated verdict in roughly one runner.
- A validated 365-day crypto data stack (spot/perp/funding/OI) with integrity checks.
- A methodology — pre-registration, append-only trial logging, lock-boxes, Deflated Sharpe, CPCV,
  regime validation, realistic cost and fill modelling — that makes overfitting and look-ahead hard
  to commit by accident.
- A paper-trading automation layer whose explicit purpose is to **catch false positives**, not to
  trade.

## Components built

| Component | Module | Role |
|---|---|---|
| **Equity research engine** | `mft/research/` | IC-first pipeline: panel → features → targets → signal_lab → cross-sectional backtest → splits → reports. Strict past-only features, no-leakage tests. |
| **Crypto data ingestion** | `mft/data_layer/crypto_provider.py`, `crypto_validate.py`, `scripts/ingest_crypto.py` | Binance spot/perp/funding/open-interest providers + validation; 365 days of clean minute data for 10 USDT majors. |
| **24/7 crypto backtest** | `mft/research/xs_backtest.py` (continuous mode), `crypto_panel.py` | Continuous UTC rebalance grid, no equity-session logic, holds span midnight, weekend bars kept. |
| **Maker simulator** | `mft/research/maker_sim.py` | Passive-fill simulation (touch / trade-through / probabilistic) with adverse selection; never fills without price touching the limit. |
| **Funding attribution** | `mft/research/funding_backtest.py` | Splits PnL into price reversion vs funding carry (known-only 8h→bar mapping). |
| **Paper-trading simulator** | `mft/automation/paper_engine.py` | Forward, simulated-fill paper trading with a per-trade ledger (signal/entry/exit/cost/regime/reason). No real orders. |
| **False-positive monitor** | `mft/automation/monitor.py` (`DeviationMonitor`) | Compares paper results to backtest expectations; a rejected strategy that looks good is flagged as suspected regime luck / overfit. |
| **No-live governance** | `mft/automation/registry.py`, `docs/NO_LIVE_DEPLOYMENT.md` | Hard gate: no live-approved flag, no `"live"` status, no order code path; checked on every run and in tests. |
| **Validation toolkit** | `mft/validation/` | Deflated Sharpe (multiple-testing correction), CPCV, metrics, diagnostics, regime analysis. |
| **Cost discipline** | `mft/execution/costs.py` | Single source of truth for transaction costs; cost sweeps everywhere. |

## Key technical lessons

1. **At mid-frequency on retail data, execution economics dominate signal discovery.** Real
   predictive effects exist (some with strong in-sample IC), but the per-trade edge (~0.5–2 bps) is
   smaller than realistic round-trip taker cost (~3–10 bps). The cost wall is the recurring killer.
2. **Naive maker execution does not rescue a taker-dead reversal.** Passive orders are adversely
   selected: they fill the continued-losers and miss the fast bounces (filled gross negative while
   the missed trades carried the edge). Earning the spread needs order-book/queue modelling.
3. **A short, lucky sample fabricates "edges."** The funding-reversal signal looked like the first
   cost-surviving alpha on 30 days — but that window was a −25% crash; on 365 days with a
   pre-registered split it was net-negative out-of-sample and made money only in down months
   (crash/liquidation beta, not alpha). **Pre-registered OOS + regime validation is non-negotiable.**
4. **Cross-sectional sign matters, and "go slower" is not a free fix.** Crypto cross-sectional
   momentum at 4h–24h was *wrong-signed* (negative IC ⇒ the cross-section mean-reverts even at
   slower horizons). The longer-hold escape hatch closed.
5. **"BTC-relative" can be a no-op.** Subtracting a per-timestamp constant (the market return) does
   not change a cross-sectional ranking — BTC-relative momentum/lead-lag was identical to raw.
6. **The most valuable artifact is the rejection pipeline.** Pre-registration, lock-boxes, trial
   budgets, Deflated Sharpe, and regime cuts cheaply killed two strategies that *looked* like the
   project's first edge. That discipline is the transferable asset.

## Where to read more

- `docs/RESEARCH_VERDICT.md` — the direct verdict.
- `docs/project_review_current_state.md` — sober checkpoint with the full branch table.
- `RESEARCH_LOG.md` — the complete chronological lab notebook (sections 0–20).
- `docs/HOW_TO_REPRODUCE.md` — how to re-run everything.
