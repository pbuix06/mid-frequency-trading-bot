"""
LESSON 1 — Learn to drive your own factory.

Goal: by the end of this file you can independently VALIDATE any backtest result
in this project — re-derive a Sharpe by hand, watch costs and honest fills kill an
edge, and see why the trial count raises the significance bar. Every number below
comes from YOUR real data (data/intraday/) and YOUR real code (mft/...). Nothing
is invented for the lesson.

HOW TO RUN
    source .venv/bin/activate
    python learn/01_validation_walkthrough.py

    # or, in VS Code: click "Run Cell" above any `# %%` block to step through it
    # interactively (needs the Python extension's Interactive Window).

It runs top-to-bottom in a few seconds and prints a guided lesson. Read the prose
between the numbers — the numbers are the point, but the *why* is the skill.

Pairs with RESEARCH_LOG.md (the lab notebook) and docs/RESEARCH_PROCESS.md (the discipline).
"""

# %% ──────────────────────────────────────────────────────────────────────────
# Setup: imports and the one universe we'll study.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from mft.backtest.intraday_session import (  # noqa: E402
    BASE_COST_PER_TRADE,
    opening_range_break_returns,
)
from mft.data_layer.alpaca_ingest import INTRADAY_LOCKBOX, load_intraday  # noqa: E402
from mft.features.intraday import daily_session_features  # noqa: E402
from mft.validation.dsr import deflated_sharpe_ratio, expected_max_sharpe  # noqa: E402
from mft.validation.metrics import full_metrics, max_drawdown, sharpe  # noqa: E402

INTRADAY_DIR = ROOT / "data" / "intraday"

# The "ex-ante high-beta" book from T0053/T0054 in the trial ledger — the project's
# best taker-viable intraday candidate. We'll reproduce its rise and fall ourselves.
HIGH_BETA = ["TSLA", "NVDA", "META", "AMZN", "XOM", "AAPL", "GOOGL"]


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def load_features(symbol: str) -> pd.DataFrame:
    """Minute bars -> per-day session features, lock-box enforced. The real path."""
    df = load_intraday(symbol, INTRADAY_DIR)
    feat = daily_session_features(df)            # gap, opening range, breakout level
    feat = feat[feat.index <= INTRADAY_LOCKBOX]  # DISCIPLINE: never touch post-lockbox
    return feat


if not list(INTRADAY_DIR.glob("*.parquet")):
    raise SystemExit("No intraday data found. Run scripts/ingest_alpaca.py first.")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 1 — What the raw data actually is, and the #1 leakage trap.
banner("LESSON 1 — what a minute bar is, and why the close is dangerous")

raw = load_intraday("AAPL", INTRADAY_DIR)
print(f"AAPL minute bars: {len(raw):,} rows, {raw.index[0]} -> {raw.index[-1]}")
print(f"columns: {list(raw.columns)}")
print("\nfirst 3 bars:")
print(raw[["open", "high", "low", "close", "volume"]].head(3).to_string())

print("""
A "bar" is a 60-second summary: Open (first trade), High, Low, Close (last trade),
Volume. The timestamp is UTC (every timestamp in this project is UTC internally).

THE TRAP that kills most beginner backtests: a bar's CLOSE is only knowable when
the 60 seconds are OVER. If your signal at 10:00:00 uses the 10:00 close, you used
information from the future — a "look-ahead bias." It inflates backtests into
fantasy. tests/test_intraday_no_lookahead.py exists to guarantee this never
happens in the real code. We'll *see* what leakage looks like in Lesson 6.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 2 — From minute bars to a strategy's daily P&L.
banner("LESSON 2 — minute bars -> one decision per day -> a return series")

feat = load_features("AAPL")
print(f"AAPL session features: {len(feat)} trading days (one row per day)\n")
print(feat[["open", "close", "or_high", "or_low", "brk_dir", "intraday_ret"]].head(5).to_string())

print("""
Each row compresses a whole day. The opening-range breakout (ORB) strategy:
  - watch the first 30 minutes -> that's the "opening range" [or_low, or_high]
  - if price later breaks ABOVE or_high -> go long (brk_dir=+1); below -> short
  - hold to the close, then go flat. ~one round-trip per day.
Economic rationale (say it in one sentence, per docs/RESEARCH_PROCESS.md): a decisive move out of
the morning range tends to continue intraday in high-energy names.
""")

ret = opening_range_break_returns(feat, cost_per_trade=BASE_COST_PER_TRADE)
print(f"AAPL ORB daily returns: {len(ret)} days. "
      f"A return series is ALL you need to evaluate any strategy.")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 3 — Compute the Sharpe BY HAND, then trust the harness.
banner("LESSON 3 — the Sharpe ratio, by hand vs the harness")

mu = ret.mean()          # average daily return
sd = ret.std()           # daily volatility (std of returns)
by_hand = mu / sd * np.sqrt(252)   # annualize: 252 trading days/year
from_harness = sharpe(ret)         # mft/validation/metrics.py

print(f"  mean daily return   = {mu:.6f}")
print(f"  daily std (vol)     = {sd:.6f}")
print(f"  Sharpe by hand      = {by_hand:.4f}   (mean/std * sqrt(252))")
print(f"  Sharpe from harness = {from_harness:.4f}   (mft.validation.metrics.sharpe)")
assert abs(by_hand - from_harness) < 1e-9, "they must match"
print("  -> they MATCH. You just verified the harness from scratch.")

print("""
Sharpe = return per unit of risk, annualized. ~1.0 is good; post-cost mid-freq
edges live at 0.5-1.5 (see docs/RESEARCH_PROCESS.md). If you ever see 3+, suspect a bug or leakage,
NOT genius. Never judge on Sharpe alone — always look at the vector:
""")
for k, v in full_metrics(ret).items():
    print(f"    {k:<14} {v:>10.4f}" if isinstance(v, float) else f"    {k:<14} {v:>10}")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 4 — Costs: where most "edges" die.
banner("LESSON 4 — the spread eats edges (gross -> net -> 2x -> 3x)")

g = sharpe(opening_range_break_returns(feat, cost_per_trade=0.0))            # GROSS
n1 = sharpe(opening_range_break_returns(feat, cost_per_trade=BASE_COST_PER_TRADE))      # 1x
n2 = sharpe(opening_range_break_returns(feat, cost_per_trade=BASE_COST_PER_TRADE * 2))  # 2x
n3 = sharpe(opening_range_break_returns(feat, cost_per_trade=BASE_COST_PER_TRADE * 3))  # 3x
print(f"  AAPL ORB Sharpe:  GROSS {g:.3f}  ->  NET(1x) {n1:.3f}  ->  2x {n2:.3f}  ->  3x {n3:.3f}")
print(f"  (1x = {2*BASE_COST_PER_TRADE*1e4:.0f} bp round-trip — realistic for a liquid mega-cap)")

print("""
GROSS ignores trading costs; NET pays the spread + slippage every round-trip. The
gap between them is the cost. A REAL edge survives 2x cost (we stress-test at 2x/3x
because we can't know the exact live cost — that's calibrated in Phase 6, not
guessed now). This single subtraction is why intraday reversal died NET -11.39 in
the ledger (T0050): the per-trade move was smaller than the spread.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 5 — Honest fills: the most important intraday lesson in this project.
banner("LESSON 5 — optimistic vs honest fills (reproducing the 0.84 -> 0.02 collapse)")

def book_sharpe(symbols: list[str], entry_slippage_bps: float, cost=BASE_COST_PER_TRADE):
    """Equal-weight book of ORB sleeves at a given entry-fill assumption."""
    streams = []
    for s in symbols:
        f = load_features(s)
        if len(f) < 250:
            continue
        streams.append(opening_range_break_returns(
            f, cost_per_trade=cost, entry_slippage_bps=entry_slippage_bps))
    book = pd.concat(streams, axis=1).mean(axis=1).dropna()
    return sharpe(book), max_drawdown(book), book

print("  High-beta book Sharpe as the ENTRY FILL gets more realistic:\n")
print(f"    {'entry slippage':<18}{'Sharpe':>9}   read")
notes = {0: "research's OPTIMISTIC fill-at-level (the lie)",
         2: "modestly realistic", 5: "~ honest next-bar taker fill",
         10: "adverse / fast market"}
for slip in (0, 2, 5, 10):
    sr, _, _ = book_sharpe(HIGH_BETA, entry_slippage_bps=slip)
    print(f"    +{slip:>2} bp{'':<11}{sr:>9.3f}   {notes[slip]}")

print("""
This IS the project's defining intraday finding (T0053 -> T0054). The research
engine assumed you fill exactly AT the breakout level (+0 bp) -> Sharpe ~0.84. But
a real taker fills on the NEXT bar, ~+5 bp worse -> the edge evaporates to ~0. The
ENTIRE 0.84 "edge" was the first 5 bp of the move. Lesson: an optimistic fill
assumption can manufacture an edge that doesn't survive contact with a real
order book. Always ask "what fill did this assume?" before believing a number.
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 6 — Multiple testing: why trying more makes the bar HIGHER.
banner("LESSON 6 — the Deflated Sharpe Ratio (why 56 trials raises the bar)")

_, _, book = book_sharpe(HIGH_BETA, entry_slippage_bps=0.0)
book_sr = sharpe(book)
print(f"  Optimistic high-beta book Sharpe = {book_sr:.3f}. Is it REAL or luck?\n")
print(f"    {'trials tried (N)':<18}{'hurdle E[max SR]':>18}{'DSR (prob real)':>18}")
for n in (1, 5, 10, 56):
    hurdle = expected_max_sharpe(n)                      # the bar, rises with N
    dsr = deflated_sharpe_ratio(book_sr, book.values, n_trials=n)
    print(f"    {n:<18}{hurdle:>18.3f}{dsr:>18.3f}")

print(f"""
If you try N strategies, the BEST one looks good by luck alone — so the more you
search, the higher the Sharpe you must clear. expected_max_sharpe(N) is that
rising bar; DSR is the probability the edge is real after paying for the search.
You need DSR >= 0.95 to advance (mft/validation/dsr.py). This is exactly why the
daily book FAILED Gate 4 and why mining more variants is self-defeating: every new
trial in trials/trials.csv (you're at 56) RAISES the bar for all of them.

(Calibration note, so you're not misled: the project's official Gate-4 DSR works in
daily-Sharpe units with an empirically-estimated benchmark — see scripts/run_cpcv.py
and RESEARCH_LOG.md sec.6. The numbers above teach the MECHANISM — bar up, DSR
down, as N grows — which is the durable intuition.)
""")


# %% ──────────────────────────────────────────────────────────────────────────
# LESSON 7 — What look-ahead bias actually LOOKS like.
banner("LESSON 7 — the smell of leakage (a deliberately cheating strategy)")

# CHEATING: decide today's position using today's close (which we wouldn't know
# until the day is over). This is look-ahead bias, on purpose, to see its signature.
leaky_pos = np.sign(feat["intraday_ret"])          # peeks at the close
leaky_ret = (leaky_pos * feat["intraday_ret"]).dropna()
print(f"  Honest AAPL ORB Sharpe          = {sharpe(ret):.3f}")
print(f"  CHEATING (peeks at the close)   = {sharpe(leaky_ret):.3f}   <- absurd")

print("""
The cheating "strategy" only knows the answer because it looked at the future. Its
Sharpe is impossibly high. THAT is the smell: a number too good to be true usually
means leakage, not alpha. When a backtest looks amazing, your first instinct should
be "where did I accidentally use the future?", not "I'm rich." This is why
tests/test_look_ahead.py is called the most important test in the repo — and why a
real spike bug (full-sample remove_outliers) was caught and fixed (RESEARCH_LOG sec.5).
""")


# %% ──────────────────────────────────────────────────────────────────────────
# CLOSING — the 6-question checklist you can now apply to ANY result.
banner("YOU CAN NOW VALIDATE ANY RESULT — ask these 6 questions")
print("""
  1. RETURN SERIES — can I get the daily returns? (everything reduces to this)
  2. BY HAND       — does mean/std*sqrt(252) match the reported Sharpe?
  3. COSTS         — is it NET, and does it survive 2x cost stress?
  4. FILLS         — what entry fill did it assume? optimistic or honest?
  5. MULTIPLE TESTING — how many trials? does DSR clear 0.95 at that N?
  6. LEAKAGE       — is it suspiciously good? could it be peeking at the future?

Run this file, change a number (try a different symbol, or HIGH_BETA universe),
and watch what moves. When these six are second nature, you are driving the
factory — not watching it. Next: open mft/alphas/base.py and read compute_signal;
everything in the system is plumbing around that one method.
""")
