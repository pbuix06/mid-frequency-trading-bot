# NO LIVE DEPLOYMENT — governance

**Status: live trading is NOT approved. No real capital is at risk. No real orders can be placed.**

This document is the deployment gate for the research/paper-trading automation framework
(`mft/automation/`). It is deliberately blunt.

## Why
The project's research program tested every signal family accessible on free/retail data —
reversal, mean-reversion, breakouts, volume, lead-lag, maker execution, funding/positioning, and
momentum — across daily equities, intraday equities, and crypto (119+ logged trials, pre-registered
splits, lock-box, regime validation). **It found no production-ready alpha.** See
`docs/project_review_current_state.md` and `RESEARCH_LOG.md`.

A paper-trading framework for *rejected* strategies exists for ONE reason: to **validate the
rejections forward** and **catch false positives** — a rejected strategy that suddenly "works" on
near-live data is treated as regime luck / overfit, recorded, and **not acted on.**

## Hard guarantees (enforced in code, tested)
1. `mft.automation.registry.LIVE_TRADING_APPROVED is False`. No code path sets it True.
2. Strategy status is restricted to `{rejected, research, paper_watch}` — **`"live"` is not a valid
   status**; `StrategySpec.__post_init__` rejects anything else.
3. `assert_no_live_approved()` runs at the start of every CLI invocation.
4. The paper engine (`paper_engine.py`) has **no real-order adapter** — there is no broker/exchange
   order call anywhere in `mft/automation/`. Every "trade" is a simulated fill written to a ledger.
5. The CLI has **no `live` subcommand**.
6. Tests (`tests/test_automation.py`) assert all of the above.

## What the framework DOES do
- **Strategy registry** — pre-registered specs with the backtest's expected IC / gross / net and verdict.
- **Paper engine** — simulates a strategy forward on near-live data, simulated fills only (taker or a
  flagged-optimistic maker), recording signal / entry / exit / cost / regime / reason for every
  hypothetical trade.
- **Monitors** — data freshness & integrity; a **deviation / false-positive guard** (paper-vs-backtest);
  and a risk/cost monitor (cost-floor, exposure, slippage assumptions).
- **Reports** — daily paper report + weekly research report (by strategy, by regime, signal-vs-realized,
  warnings), each re-stating that no live deployment is approved.

## What would have to change before any live consideration (NOT planned)
A green light would require, at minimum: a genuine candidate alpha that survived OOS + lock-box +
regime validation + realistic costs (none exists); an independent risk sign-off; a real
execution/queue/adverse-selection model on paid order-book data; and a staged, capital-capped rollout
with kill-switches. **None of this is in scope.** This framework is the opposite: a machine for proving,
honestly and forward, that the rejected edges stay rejected.
