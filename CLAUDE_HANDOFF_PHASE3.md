# Claude Handoff: Phase 0-3 Fixes And Research Quality

Context for Claude:

- User has NOT started Phase 4.
- Do not open or use post-lockbox data.
- Lockbox cutoff is `LOCKBOX_CUTOFF = 2022-07-01 UTC`; research must pass `end_date=LOCKBOX_CUTOFF`.
- Current goal is to keep Phase 0-3 aligned with the research plan, then improve pre-lockbox research quality before any honest validation.

## What Codex Fixed

Core simulator semantics:

- `mft/backtest/vectorbt_harness.py`
  - Replaced binary entry/exit simulation with target-percent order simulation.
  - Preserves fractional weights such as TSMomentum volatility sizing.
  - Computes signals on closed bars and applies target weights on the next bar open.
  - Treats `{}` as hold current position.
  - Added slippage support to `run_research_xs`.

- `mft/backtest/event_harness.py`
  - Treats `{}` as hold, matching `AlphaBase`.
  - Rebalances using fractional target-weight deltas, not "current position means 100 percent long."
  - Uses next-bar open execution with slippage based on trade direction.

Alpha behavior:

- `mft/alphas/pairs_mean_reversion.py`
  - Hold band now returns `{}` instead of flat weights.
  - Explicit zeros mean exit; `{}` means hold.
  - Direct calls stay flat until the declared 252-bar lookback is available.
  - Ignores older bars outside the declared `lookback + 1` window if a caller passes too much history.

- `mft/alphas/xs_momentum.py` and `mft/alphas/long_short_momentum.py`
  - Fixed a one-bar-short denominator in the return lookback.
  - Signals now rank from exactly `t-lookback` to `t-skip`.

- `mft/alphas/low_vol_anomaly.py`
  - Direct calls now respect the declared strategy lookback, not just `vol_window`.

Research scripts and lockbox discipline:

- `scripts/check_correlations.py`
  - Now passes `LOCKBOX_CUTOFF` into every sleeve.
  - Uses the same `LongShortMomentum` universe as `scripts/check_regimes.py`.
  - Wording changed from "final suite" to candidate/correlation screen language.
  - Prints each sleeve's actual pre-lockbox window instead of a stale hard-coded start year.

- `scripts/check_regimes.py`
  - Uses the same candidate universe and explicit 1x slippage as the correlation screen.

- `scripts/parity_check.py`
  - Real-data parity checks now enforce `LOCKBOX_CUTOFF` by default.
  - Added a dangerous `--include-lockbox` escape hatch only for the Phase 4 final exam.

- `scripts/run_backtest.py`
  - Added `LongShortMomentum` to the alpha registry.
  - Passes slippage into cross-sectional runs.
  - `--dry-run` now warns that printed metrics are diagnostic only and not research evidence.
  - Logs actual return/data windows more accurately and prints MinBTL in bars plus years.

Config/dependency cleanup:

- `configs/sma_crossover.yaml`
  - Path fixed to `data/pit/SPY.parquet`.
  - End date fixed to `2022-07-01`.

- `mft/data_layer/loader.py`
  - `_normalize()` now returns OHLCV in the documented order: `open, high, low, close, volume`.

- `pyproject.toml`
  - Added optional `ib` extra for `ib-insync`.

Tests and hygiene:

- Added `tests/test_harness_semantics.py` for fractional target-weight rebalancing.
- Tightened no-look-ahead tests in `tests/test_phase3_alphas.py`.
- Cleaned repo-wide ruff issues.
- Updated `CLAUDE.md` with current Phase 3 wording and metrics.

## Current Verification

Commands run after the fixes:

```bash
.venv/bin/ruff check mft tests scripts
.venv/bin/python -m pytest
.venv/bin/python scripts/check_correlations.py
.venv/bin/python scripts/check_regimes.py
```

Results:

- Ruff: all checks passed.
- Pytest: 71 passed.
- Correlation screen: passed, all pairwise absolute correlations below 0.3.
- Regime check: runs pre-lockbox only.

Current pre-lockbox correlation screen metrics:

| Sleeve | Sharpe | CAGR | Max DD | Note |
|---|---:|---:|---:|---|
| TSMOM_SPY | 0.5513 | 5.89% | -22.26% | Acceptable candidate |
| TSMOM_GLD | 0.5075 | 5.46% | -29.84% | Acceptable candidate |
| TSMOM_TLT | 0.1887 | 1.46% | -27.23% | Diversifying but weak |
| LongShort_Eq | 0.1100 | 0.44% | -42.62% | Very weak, drawdown-heavy |

Pairwise correlations:

- SPY vs GLD: +0.069
- SPY vs TLT: -0.286
- SPY vs LongShort_Eq: +0.130
- GLD vs TLT: +0.102
- GLD vs LongShort_Eq: +0.086
- TLT vs LongShort_Eq: +0.131

## Final Audit Notes Before Phase 4

- Gate 3 still passes on correlation only: all candidate pairwise absolute correlations are below 0.3.
- `TSMOM_SPY` and `TSMOM_GLD` are the only decent standalone candidates right now.
- `TSMOM_TLT` is a weak diversifier, not a strong alpha.
- `LongShortMomentum` became weaker after the exact-lookback fix; treat it as a candidate to improve or replace.
- Early trial-log rows include post-lockbox or pre-fix experiments. They still count against the trial budget, but they should not be cited as clean evidence.
- `mft/backtest/nautilus_harness.py` is still a Phase 2 single-asset parity harness. It does not yet support the full fractional/multi-asset/hold semantics used by the Phase 3 research harnesses, so do not use it to validate these candidate sleeves until it is upgraded.
- `mft/data_layer/cleaning.py::remove_outliers()` is full-sample logic. Do not use it in PIT research unless it is changed to rolling/refit-only behavior.

## Research Quality Diagnosis

The code is now closer to the research plan, but the research quality is not equally strong across sleeves.

Main issues:

- `TSMOM_TLT` passes the correlation screen but has weak standalone Sharpe and loses in the 2020-2022 rate-hike regime.
- `LongShortMomentum` is diversifying but very weak and drawdown-heavy. It failed the financial-crisis regime and has a large max drawdown.
- Phase 3 should be treated as a candidate suite, not a ready final suite.

## How To Improve Research Quality Without Cheating

Do this only on pre-lockbox data and log every backtest/trial:

Recommended priority: improve the existing Phase 3 candidates before adding many new alphas. The current problem is weak candidate quality, not a shortage of ideas.

1. Improve or replace weak sleeves before Phase 4.
   - Keep `TSMOM_SPY` and `TSMOM_GLD` as current strongest candidates.
   - Treat `TSMOM_TLT` as optional diversification, not a proven alpha.
   - Rework or replace `LongShortMomentum` before taking it seriously.

2. For `LongShortMomentum`, test structural improvements, not random parameter hunting.
   - Use a larger, liquid, survivorship-aware equity universe.
   - Add sector neutrality or beta neutrality.
   - Add volatility scaling by leg or by name.
   - Track turnover and cost sensitivity.
   - Compare equal-weight legs vs inverse-vol legs.

3. Promote or reject `LowVolAnomaly` with a cleaner structure.
   - Test sector-neutral low-vol or low-beta/high-beta long-short.
   - Measure hidden SPY beta; do not accept it if it is just defensive equity beta.
   - Keep the rationale tied to the low-vol / low-beta anomaly, not parameter tuning.

4. Turn the single-asset TSMOM sleeves into a cleaner multi-asset trend sleeve.
   - Keep SPY and GLD as the strongest current trend candidates.
   - Compare bond alternatives only as pre-lockbox research, not lockbox validation.
   - Compare TLT, IEF, AGG/BND if data quality exists.
   - Require stable regime behavior, not just lower correlation.
   - If the sleeve only diversifies but does not earn, mark it as portfolio hedge candidate rather than alpha.

5. Defer `PairsMeanReversion` unless the missing research checks are added.
   - Require cointegration testing, rolling hedge-ratio stability, and structural-break checks.
   - Log pair selection as trials; pair mining has many hidden degrees of freedom.

6. Do not prioritize `ShortReversion` on daily data.
   - Current daily evidence is weak and cost-sensitive.
   - Revisit it later if the project moves to intraday data.

7. Add a research-quality gate before Phase 4.
   Suggested minimums:
   - Sharpe above 0.3 for diversifiers, above 0.5 for core sleeves.
   - Max drawdown not absurd relative to CAGR.
   - Positive or near-flat in at least 3 of 4 regimes.
   - Pairwise absolute correlation below 0.3.
   - Cost stress does not destroy the result.

8. Add diagnostics before more alpha work.
   Useful next scripts:
   - parameter sensitivity grid, pre-lockbox only
   - rolling Sharpe/drawdown table
   - cost stress table
   - turnover report
   - universe membership/data-quality report

Important: do not "fix" poor research quality by tuning until Sharpe improves. Fix it by improving economic structure, robustness checks, and rejecting weak candidates.

Short version for Claude: do not add a large batch of new alphas yet. First improve `LongShortMomentum`, make `LowVolAnomaly` a real sector/beta-aware candidate, and consolidate TSMOM into a cleaner multi-asset trend sleeve. Add new alphas only after these existing ideas are either upgraded or rejected.
