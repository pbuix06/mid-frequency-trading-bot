# Phase 4 Handoff — Honest Validation verdict

## Outcome: the candidate suite FAILS Gate 4 (Deflated Sharpe Ratio)

Tools: `scripts/run_validation.py` (per-sleeve DSR), `scripts/run_cpcv.py`
(portfolio DSR + CPCV distribution). Both lock-box-safe (≤ 2022-07-01).

The "one canonical engine" question dissolved: the sleeves are two strategy
TYPES (single-asset trend via event harness; cross-sectional dollar-neutral via
the survivorship harness), so the common currency is the daily RETURN SERIES.
DSR/CPCV operate on returns. Per-sleeve uses each sleeve's maximal window;
portfolio uses the common 2010-2022 window.

### Numbers
Deflated bar ≈ **1.02 annualized Sharpe** (N = 46 logged trials; empirical
cross-trial Sharpe dispersion ≈ 0.45 ann). DSR pass threshold = 0.95.

Per-sleeve DSR (max windows):
| Sleeve | Sharpe | DSR |
|---|---|---|
| TSMOM_SPY | 0.55 | 0.016 |
| TSMOM_GLD | 0.51 | 0.019 |
| TSMOM_TLT | 0.19 | 0.000 |
| LongShort | 0.41 | 0.001 |

Combined book (common 2010-2022 window):
| Portfolio | Sharpe | MaxDD | DSR | CPCV median | frac+ |
|---|---|---|---|---|---|
| equal-weight | 0.61 | -14% | 0.010 | 0.67 | 100% |
| inverse-vol | **0.73** | **-11%** | **0.134** | 0.72 | 96% |

### Interpretation
Two things are both true:
- The book PASSES robustness: CPCV stability (96% of sub-period paths positive),
  2x cost stress (Phase 3.5), survivable drawdown (-11%), mostly-positive
  regimes. In absolute terms it is an attractive, stable, diversified book in
  the realistic 0.5-1.5 Sharpe range.
- It FAILS the multiple-testing-corrected significance gate: after honestly
  accounting for 46 trials, Sharpe 0.73 cannot be distinguished from the best of
  46 lucky draws at 95%. Sensitivity-checked: even a much smaller effective N
  does not rescue it (would need N≈2).

Per the playbook, a failed gate does not graduate. **Do not risk real capital on
this suite.** The gate did its job — it prevented deploying capital on an
unproven edge.

### Root cause — data, not infrastructure
Daily price-only data yields real-but-modest momentum edges (0.5-0.7) that can't
clear an honest bar after a realistic search. The families that would add
genuinely uncorrelated breadth need data we lack:
- short-horizon reversal: real gross edge (0.39) destroyed by 76x turnover on
  daily bars -> needs INTRADAY data
- value / quality / carry: need FUNDAMENTAL data

### Decision: option 1 — pursue better data, then re-run this pipeline
The infrastructure (survivorship harness, DSR, CPCV, cost model, lock-box
discipline, one-code-path) is the durable asset and is ready. When better data
arrives, re-run `run_validation.py` + `run_cpcv.py` unchanged.

Status of the four sleeves: not rejected as ideas, but not graduated. They
remain valid building blocks for a future book that also includes uncorrelated
non-momentum families.

### What NOT to do
- Do not lower the 95% bar to rationalize a pass.
- Do not mine more momentum variants (raises N, worsens the bar).
- Do not deploy real capital. Paper-trading for plumbing validation (Phase 6,
  zero capital) is acceptable ONLY if explicitly not treated as edge validation.

---

## Update: EDGAR survivorship "fix" backfired — selection bias (REJECTED)

Attempted to de-bias the value/quality universe by matching delisted tickers to
historical SEC CIKs (cik-lookup-data.txt) and ingesting their fundamentals
(385 recovered). Result looked great — value 0.29->0.77, portfolio DSR
0.35->0.82 — and was REJECTED as a selection-bias artifact:

  - 67% of recovered delisted names ended within 30% of their 1-yr peak
    (median last/peak 0.94) => the matchable delistings are overwhelmingly
    orderly ACQUISITIONS (which end high and target cheap/value names), not
    bankruptcies. The name-match recovers acquisitions (clean filings/names) but
    misses the offsetting bankruptcies (messy/none) -> value spuriously inflated.
  - Two tells: (1) adding failed firms RAISED value (wrong direction for
    survivorship); (2) 2010-2022 was a value-winter, so 0.77 is implausible.

Lesson: free EDGAR gives clean PIT fundamentals for SURVIVORS but cannot produce
a clean survivorship-free cross-section. That needs paid survivorship-free
fundamentals (CRSP/Compustat). Do NOT trust the de-biased fundamental numbers.

Trustworthy state stands: value ~0.29, quality ~0.16 (survivor-only, mildly
biased but right ballpark); both real + uncorrelated with momentum; portfolio
DSR ~0.35. Still below Gate 4. The artifact (DSR 0.82) is NOT counted.

Next lever: genuinely new DIMENSIONS that don't depend on a clean equity
delisting universe — FX (no survivorship issue; EODHD FX data is free/on-plan)
and intraday (MT5/Alpaca/paid). Pursue cross-asset breadth.
