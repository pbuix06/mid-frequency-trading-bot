"""
LESSON 2 — Re-derive the Gate 4 verdict yourself.

Goal: by the end of this file you have reproduced, from raw sleeve returns and the
append-only trial ledger, the exact numbers that killed the daily suite at Gate 4:
book Sharpe 0.73, deflated bar 1.02, DSR 0.134 vs the 0.95 pass line. Not "the
system says it failed" — YOU say it failed, because you computed it.

Every number comes from YOUR real data (data/pit/) and YOUR real code (mft/...,
scripts/run_validation.py, scripts/run_cpcv.py). Nothing is invented, and nothing
here writes to trials/trials.csv — re-examining logged results is free; only NEW
configs cost overfitting budget.

HOW TO RUN
    source .venv/bin/activate
    python learn/02_gate4_rederivation.py

    # or, in VS Code: click "Run Cell" above any `# %%` block to step through it.

The sleeve build takes ~20 seconds (it replays four real backtests, including an
800-name survivorship-free panel). Read the prose between the numbers.

Pairs with RESEARCH_LOG.md sec.6 (the Phase 4 verdict) and docs/RESEARCH_VERDICT.md.
"""

# %% ──────────────────────────────────────────────────────────────────────────
# Setup: rebuild the four sleeves through the REAL validation scripts.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cpcv as rc  # noqa: E402  (the official Gate-4 step-2 script — we reuse, not reimplement)

from mft.validation.dsr import deflated_sharpe_ratio, expected_max_sharpe  # noqa: E402
from mft.validation.metrics import max_drawdown, sharpe  # noqa: E402

TRIALS = ROOT / "trials" / "trials.csv"


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


print("Rebuilding the four sleeves (TSMOM SPY/GLD/TLT + LongShort momentum) through")
print("the same code Gate 4 ran — event-driven, lock-box-enforced, tiered costs.")
print("~20 seconds ...")
df = rc.common_window_returns()          # daily returns, common 2010->lockbox window
print(f"done: {df.shape[0]} common days, {df.index[0].date()} -> {df.index[-1].date()}")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 1 — Four modest sleeves, and why LOW CORRELATION is the raw material.
banner("LESSON 1 — the four sleeves on the common window (each modest alone)")

print(f"  {'sleeve':<12}{'Sharpe':>8}{'MaxDD':>9}")
for c in df.columns:
    print(f"  {c:<12}{sharpe(df[c]):>8.3f}{max_drawdown(df[c]):>9.2%}")

print("\n  Pairwise correlation of daily returns:")
print(df.corr().round(2).to_string())

print("""
No sleeve is impressive alone — that IS the honest post-cost reality (0.5-1.5 is
the realistic band; see docs/RESEARCH_PROCESS.md). But look at the correlations:
near zero, some negative. Uncorrelated modest edges are the raw material of
portfolio construction: their returns add, their noise partially cancels. That
was the entire Phase 3 point of the |rho|<0.3 correlation gate.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 2 — Combine them: diversification works exactly as advertised...
banner("LESSON 2 — the combined book (diversification's free lunch, by hand)")

inv_vol = 1.0 / df.std()                     # risk-balance: weight ~ 1/vol
iv_w = inv_vol / inv_vol.sum()
book = (df * iv_w).sum(axis=1)               # the book IS just a weighted sum of returns

book_sr = sharpe(book)
book_dd = max_drawdown(book)
best_sleeve = max(sharpe(df[c]) for c in df.columns)

print(f"  inverse-vol weights: {dict(iv_w.round(2))}")
print(f"  book Sharpe = {book_sr:.3f}   (best single sleeve: {best_sleeve:.3f})")
print(f"  book MaxDD  = {book_dd:.2%}   (sleeves ranged to ~-50%)")
assert abs(book_sr - 0.725) < 0.01, "book Sharpe should reproduce the recorded 0.73"

print("""
Higher Sharpe than ANY sleeve, and roughly half the drawdown — diversification
did exactly what the textbooks promise. This is the strongest book the daily
suite could field. Hold that number: 0.73. Now comes the question that actually
decides Gate 4: is 0.73 EVIDENCE of edge, or the expected trophy of a search?

(Note: inverse-vol uses full-sample vol — a mild look-ahead the project flags
openly; the equal-weight book has zero estimation and fails the same way.)
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 3 — The bar: what LUCK alone produces after 46 recorded attempts.
banner("LESSON 3 — reconstruct the luck bar from the append-only ledger")

ledger = pd.read_csv(TRIALS)
frozen = ledger.iloc[:46]                                    # the ledger AS OF Gate 4
sig_ann = pd.to_numeric(frozen["is_sharpe"], errors="coerce").dropna().std(ddof=1)
sig_daily = sig_ann / np.sqrt(252)
N = 46

bar_ann = expected_max_sharpe(N, sr_std=sig_daily) * np.sqrt(252)
print(f"  ledger rows at Gate-4 time : {len(frozen)} (T0001..T0046 — append-only, so")
print("                               the historical state is reconstructable forever)")
print(f"  cross-trial Sharpe std     : {sig_ann:.4f} annualized")
print(f"  E[max Sharpe | 46 tries]   : {bar_ann:.3f} annualized  <- the bar")
assert abs(sig_ann - 0.4547) < 0.001 and abs(bar_ann - 1.02) < 0.01

print(f"""
The logic, in one breath: we made 46 recorded attempts whose Sharpes scatter with
std {sig_ann:.2f}. Even if EVERY attempt were pure noise, the best of 46 draws from
that distribution is expected to show Sharpe ~{bar_ann:.2f}. So a book at 0.73 is
BELOW what luck alone would hand the winner of our own search. That single
sentence is why "count every trial" is a core discipline: an unlogged search has
no bar, and no bar means any number can masquerade as edge.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 4 — DSR: the probability the book is real. The verdict, re-derived.
banner("LESSON 4 — the Deflated Sharpe Ratio (the number that failed the gate)")

sr_daily = float(book.mean() / book.std())                   # per-bar Sharpe units
bench_daily = expected_max_sharpe(N, sr_std=sig_daily)
dsr = deflated_sharpe_ratio(sr_daily, book.dropna().values, n_trials=N,
                            sr_benchmark=bench_daily)

print(f"  book Sharpe (ann.)  = {book_sr:.3f}")
print(f"  luck bar    (ann.)  = {bar_ann:.3f}")
print(f"  DSR                 = {dsr:.3f}     pass line = 0.95")
print(f"  verdict             = {'PASS' if dsr >= 0.95 else 'FAIL — do not risk real capital'}")
assert abs(dsr - 0.134) < 0.005, "must reproduce the recorded Gate-4 DSR of 0.134"

print("""
DSR asks: given the search we ACTUALLY ran, what is the probability this Sharpe
reflects real edge rather than selection luck? 0.134 means ~13% — nowhere near
the 95% the playbook demands before real money. You just reproduced the exact
recorded verdict (RESEARCH_LOG sec.6: "DSR 0.134 << 0.95").

Read it precisely: this is NOT "the book loses money." It is "after an honest
accounting of the search, we cannot distinguish 0.73 from luck." A significance
failure, not an execution failure — which is why the decision was better DATA
(new signal families), never more mining of the same data.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 5 — CPCV: judge a DISTRIBUTION, never a point estimate.
banner("LESSON 5 — CPCV: 28 held-out paths instead of one flattering number")

dist = rc.cpcv_sharpe_distribution(book)     # Sharpe on every C(8,2) held-out combo
print(f"  paths = C(8,2) = {len(dist)} held-out sub-period combinations")
print(f"  median Sharpe = {np.median(dist):.3f}")
print(f"  [5, 95] band  = [{np.percentile(dist, 5):.3f}, {np.percentile(dist, 95):.3f}]")
print(f"  frac positive = {np.mean(dist > 0):.0%}")

print("""
One backtest Sharpe is a single draw; CPCV slices the sample into 8 groups and
evaluates every 2-group holdout -> 28 paths -> a distribution. The book's median
(0.72) sits right on the full-sample number and ~96% of paths are positive: the
edge is STABLE, not one lucky regime. And that is the mature reading of Gate 4:
stable + modest + below the luck bar = "real but too small to prove after the
search we ran." Robustness passed; significance failed; the gate stays shut.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 6 — Why re-running the official script TODAY fails even harder.
banner("LESSON 6 — the bar only rises: N=46 then, N=129 now (and a polluted sigma)")

all_sr = pd.to_numeric(ledger["is_sharpe"], errors="coerce").dropna()
print(f"  trials logged now            : {len(ledger)}")
print(f"  naive std of ALL is_sharpe   : {all_sr.std(ddof=1):.2f} ann.   (was {sig_ann:.2f})")
print(f"  most extreme logged Sharpes  : {sorted(all_sr)[:3]}")
print(f"  E[max] at N=129, frozen sigma: "
      f"{expected_max_sharpe(129, sr_std=sig_daily) * np.sqrt(252):.2f} ann. (bar was {bar_ann:.2f})")

print("""
Two things happened to the ledger since Gate 4, and both teach:

1. N grew 46 -> 129. E[max] grows with N, so the SAME book faces a HIGHER bar
   today. Every trial you log raises the bar for every strategy forever — the
   overfitting budget is real, and spending it "just to check" is not free.

2. The naive dispersion estimate exploded (0.45 -> ~29) because later intraday /
   crypto trials logged deliberately-cost-dead books with annualized Sharpes
   like -157 (high-turnover strategies annualize catastrophically). Feed THAT
   sigma to the formula and DSR prints 0.000 — directionally right, numerically
   meaningless. Estimator hygiene: the luck bar needs the dispersion of
   COMPARABLE trials, which is why the verdict is stated at the frozen N=46
   state. Either way the conclusion is the same — the fail only gets harder.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# CLOSING — the Gate-4 questions you can now ask of ANY strategy suite.
banner("YOU CAN NOW JUDGE A SUITE, NOT JUST A SLEEVE — ask these 5 questions")
print("""
  1. COMMON WINDOW  — are all sleeves evaluated on the same aligned period?
  2. THE BOOK       — does combining beat the best sleeve (Sharpe up, DD down)?
     If not, the "portfolio" is decoration.
  3. THE BAR        — how many trials produced this, with what dispersion, and
     what would luck's best look like at that N? (expected_max_sharpe)
  4. DSR >= 0.95    — probability of real edge AFTER the search. Below the line,
     a failed gate does not graduate. No exceptions, no "but it's stable."
  5. CPCV           — is the median near the point estimate with frac+ ~100%?
     Stable-but-insignificant means "collect better data", not "tune harder."

You have now personally re-derived the project's central negative result:
0.73 / bar 1.02 / DSR 0.134. When someone asks "how do you know you didn't just
overfit?", THIS file is the answer. Next: learn/03 — watch the same discipline
unmask a 30-day crypto "edge" as crash beta.
""")
