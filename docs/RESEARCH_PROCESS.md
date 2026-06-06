# Research process & discipline

The methodology this project is built around. The point of these rules is to make
self-deception hard: finding a signal that looks profitable is easy and usually wrong; the
work is proving whether it survives costs, execution, and multiple-testing.

## Core discipline

- **One code path.** The same signal logic runs in research, paper, and (hypothetical) live.
  No strategy is ever rewritten "for production." This is what lets research numbers and
  paper numbers be trusted to come from the same logic.
- **Count every trial.** Every backtest gets exactly one row in `trials/trials.csv`
  (append-only, never edited). Untracked search is guaranteed overfit; the trial count drives
  the multiple-testing correction (Deflated Sharpe).
- **Lock-box, hard-coded.** A cutoff date is fixed in advance; research may read bars only up
  to it. The lock-box is opened exactly once, at a final exam, and never used to tune.
  (Daily: `LOCKBOX_CUTOFF = 2022-07-01`; intraday: `INTRADAY_LOCKBOX = 2023-07-01`; crypto
  validation uses a pre-registered train/validation/lock-box split.)
- **Economic rationale before statistics.** If the reason an edge should exist can't be stated
  in one sentence, it's probably noise.
- **Realistic post-cost Sharpe.** Mid-frequency post-cost Sharpe lives around 0.5–1.5. Anything
  ≥ 2 is assumed to be a bug or overfit until proven otherwise.
- **A failed gate does not graduate.** No rationalising a pass, no lowering the bar.

## Validation gates

- **Parity** — research and event-driven engines must produce the same returns for the same
  strategy, so numbers reflect logic, not engine quirks.
- **No look-ahead** — every feature is computable from data available at its bar's close;
  enforced by tests (`tests/test_look_ahead.py`, `tests/test_*_no_lookahead.py`).
- **Survivorship-free** — delisted instruments are included where the data allows; cross-sectional
  studies use a point-in-time, delisting-aware engine.
- **Multiple-testing** — Deflated Sharpe / minimum-backtest-length bound the honest search budget.
- **Robustness** — CPCV (purged cross-validation), cost stress (1×/2×/3×), and regime analysis.
- **Out-of-sample + regime** — pre-registered train/validation/lock-box splits with PnL attributed
  by market regime; a "winner" confined to one regime (e.g. a crash) is labelled beta, not alpha.

## Cost discipline

Transaction costs are the single biggest reason apparent edges disappear. A single cost model is
the source of truth (`mft/execution/costs.py`); every result is reported gross **and** net, with a
cost sweep (e.g. 0/2/5/10 bps per side). Costs are not equal across instruments — flat rates are
conservative for liquid ETFs/majors and optimistic for illiquid names — so uncertainty is handled
with cost *stress*, not false per-name precision.

## Anti-overfitting workflow for a new signal

1. State the economic hypothesis in one sentence.
2. Pre-register the config grid, the splits, and the regime cuts **before** running.
3. Measure information coefficient (IC) first — signal predictivity, fill-free — before any backtest.
4. Only then backtest: gross and net, cost sweep, leg/symbol/regime attribution.
5. Log every config to the trial ledger.
6. Apply the decision rules; reject honestly. Most ideas die — that is the expected outcome.

## Provenance

The complete chronological record — every phase, finding, and rejection — is `RESEARCH_LOG.md`.
The current sober assessment is `docs/project_review_current_state.md`; the conclusion is
`docs/RESEARCH_VERDICT.md`.
