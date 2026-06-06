# MFT — Master Research Log

> The complete chronological record of this project: every phase, every trial, every
> finding, and — just as important — every idea we *rejected* and why. This is the
> lab notebook. It is meant to be read top-to-bottom by anyone (including future-us)
> who needs to understand *how we got here* without re-deriving it.
>
> Companion docs: `docs/RESEARCH_PROCESS.md` (operating discipline + methodology),
> `docs/project_review_current_state.md` (sober checkpoint), `docs/RESEARCH_VERDICT.md`
> (the conclusion), `trials/trials.csv` (the append-only ledger — never edited).
>
> Last updated: 2026-06-06. Trials logged: 129. Tests passing: 216.

---

## 0. The discipline (the rules we never break)

These are the guardrails. Every finding below was produced under them.

- **One code path.** The same `AlphaBase.compute_signal()` runs in backtest, paper, and
  live. No strategy is ever rewritten "for production."
- **Count every trial.** Every backtest = one row in `trials/trials.csv`. Untracked
  search = guaranteed overfit. We are at 53 trials.
- **Lock-box, hardcoded.**
  - Daily research: `LOCKBOX_CUTOFF = 2022-07-01`. Never read a daily bar after it.
  - Intraday research: `INTRADAY_LOCKBOX = 2023-07-01`. Never read a minute bar after it.
  - Opened exactly once, at the Phase 4 final exam. Still sealed.
- **Economic rationale before statistics.** If we can't say *why* an edge exists in one
  sentence, it's noise.
- **Realistic Sharpe.** Post-cost mid-freq Sharpe is ~0.5–1.5. A 2+ means a bug or overfit.
- **A failed gate does not graduate.** No rationalizing a pass. No lowering the bar.

---

## 1. Phase 0 — Skeleton + parity (PASSED Gate 0)

**Goal:** prove the machine before trusting any number it produces.

- Built the repo skeleton, the `AlphaBase` interface (the spine everything calls), and
  two backtest engines: a fast vectorbt research harness and an event-driven harness.
- **Gate 0 = parity:** the same SMA-crossover strategy must produce the *same* returns
  through both engines. It did. This is what lets us trust that research numbers and
  live numbers come from the same logic.
- IB "hello world" + EODHD ingestion smoke test.

**Trials:** T0001–T0004 (SMACrossover, SPY). The first three were short-window parity
checks (negative Sharpe — expected, they're correctness tests not alphas). T0004 over
2010–2026 gave 0.61 — a sane trend number, confirming the engine works.

**Finding:** infrastructure is sound. Numbers can be trusted to *reflect the logic*
(whether the logic has edge is a separate question).

---

## 2. Phase 1 — Data layer (EODHD)

**Goal:** survivorship-bias-free, point-in-time daily data.

- Built the EODHD ingestion → 41,492 PIT files. US stocks + multi-asset ETFs/FX from 2000+.
- **Rules baked in:** all timestamps UTC; delisted instruments *included* (no survivorship
  bias); fundamentals stamped when *knowable* not when describing; `data/raw/` = vendor
  dumps, `data/pit/` = cleaned PIT-aligned Parquet.
- Constants live in `mft/data_layer/eodhd_ingest.py`: `RESEARCH_FROM = 2000-01-01`,
  `LOCKBOX_CUTOFF = 2022-07-01`.

**Decision recorded:** research window is the whole 2000-01-01 → 2022-07-01 span
(~23 yrs, ~5,870 trading days), *not* broken into small intervals. Rationale: more
continuous history = larger honest trial budget (MinBTL math, see §9).

---

## 3. Phase 2 — NautilusTrader parity + crash recovery (PASSED Gate 2)

**Goal:** a validation engine independent of the research engine, plus the ability to
survive a live crash without losing position state.

- `mft/backtest/nautilus_harness.py` — NautilusTrader parity engine (single-asset at
  this stage).
- `mft/execution/state.py` — crash-recovery state persistence.
- **Gate 2 passed:** parity held across the third engine.

**Caveat recorded (honest):** the Nautilus harness was Phase-2 single-asset only. It does
*not* yet support the fractional/multi-asset/hold semantics the Phase 3 research harnesses
use, so it could not validate the later cross-sectional sleeves. Flagged as an open item.

---

## 4. Phase 3 — Candidate alpha suite + correlation screen (PASSED Gate 3)

**Goal:** assemble a stable of economically-motivated candidate sleeves, screen them for
*low mutual correlation* (the only free lunch — uncorrelated edges stack).

### First pass (T0005–T0024): broad casting, including post-lockbox windows
We ran a wide set to see what had any pulse:
- **TSMomentum (time-series trend)** — the workhorse. SPY 0.68, plus multi-asset scan:
  GLD 0.65, IEF 0.37, EEM 0.29, EFA 0.22, TLT 0.06, USO 0.02, EURUSD −0.14.
- **XSMomentum (cross-sectional)** — 0.73 on SPY/QQQ/IWM.
- **ShortReversion (daily mean-reversion)** — negative nearly everywhere (SPY −0.71;
  per-name AAPL/MSFT/AMZN/JPM/XOM all weak-to-bad). First hint reversal doesn't work on
  daily bars.
- **PairsMeanReversion** — −0.59. Deferred (needs cointegration/hedge-ratio stability work).
- **LowVolAnomaly** — 0.78 headline. *Held for scrutiny* (see Phase 3.5 — it was a trap).
- **LongShortMomentum (WML, dollar-neutral)** — the survivorship story starts here.

*These early rows count against the trial budget but are NOT clean evidence* (some used
post-lockbox windows / pre-fix code). Logged honestly.

### Clean redo (T0025–T0045): lock-box enforced, data from 2000, bugs fixed
A code-hardening pass fixed real simulator bugs (target-percent orders not binary
entry/exit; `{}` = hold; next-bar-open execution; one-bar-short lookback denominators).
Re-ran everything **pre-lockbox only**:
- TSMOM_SPY **0.556**, TSMOM_GLD **0.512**, TSMOM_IEF 0.42, TSMOM_EFA 0.33, TSMOM_EEM 0.30,
  TSMOM_TLT 0.19, TSMOM_USO −0.02, TSMOM_EURUSD −0.02.
- XSMomentum 19-stock: 0.92–0.93 headline **but −0.98 to −0.60 max drawdown** → long-only
  XS momentum is destroyed in crashes (a tell, not an alpha).
- LongShortMomentum 19-stock: 0.20 (down from earlier — the exact-lookback fix made it
  honest and *weaker*).
- LowVolAnomaly 19-stock: 0.36 after fixes (down from 0.78 — another tell).

### Gate 3 — correlation screen PASSED
All candidate pairwise |ρ| < 0.3 on clean pre-lockbox data:
- SPY×GLD +0.069, SPY×TLT −0.286, SPY×LongShort +0.130, GLD×TLT +0.102,
  GLD×LongShort +0.086, TLT×LongShort +0.131.

The sleeves are genuinely diversifying relative to each other. Gate 3 is about
*independence*, not strength — and it passed.

---

## 5. Phase 3.5 — Research quality (the honesty pass, pre-Phase-4)

Acted on a research-quality review *before* the honest validation. This is where
several headline numbers died — which is the point.

### Causal cleaning bug fixed
`remove_outliers()` was full-sample (look-ahead — a late spike could rewrite earlier
bars). Rewrote it causal: trailing rolling z-score shifted by 1. Tests now prove a late
spike cannot change earlier bars.

### LowVolAnomaly — REJECTED (disguised market beta)
Regressed its returns on SPY buy-and-hold (2000–2022): **beta 0.65, R² 0.66, annualized
alpha −0.26%, beta-hedged Sharpe −0.03.** It was defensive *equity beta* wearing a
low-vol costume, not a low-vol premium. Dropped from the suite. (This is why we held it
for scrutiny in Phase 3.)

### Survivorship-free cross-sectional study (the big one)
Built `mft/backtest/survivorship_harness.py` — a PIT survivorship-free engine with
delisting liquidation, tracking **signed dollar position values** (no division — two
earlier share-accounting / weight-renormalization builds blew up with −142,000%
drawdowns shorting low-priced delisting names).

The liquid candidate pool (≥$20M ADV, data by 2010) is **1,929 names — 35% (673) delisted
before 2022.** Exactly the bias the 19-name screen had hidden.

- **LongShortMomentum: 0.11 (19 survivors) → 0.41 (avg 598 names, survivorship-free,
  2010–2022).** Breadth + de-biasing ~4×'d it (Grinold's law: breadth × IC). Survives
  2× and 3× cost stress (0.41 / 0.39 / 0.36) and liquidity-tiered costs (0.409). A real
  market-neutral diversifier candidate (raw MaxDD −50%, needs vol-targeting in Phase 5).
- **XSMomentum long-only: non-viable** on a broad universe (−97% MaxDD). Dollar-neutral
  form only.

### Single-source cost model
`mft/execution/costs.py` became the *single source of truth*: `DEFAULT_COMMISSION_PCT` /
`DEFAULT_SLIPPAGE_PCT` (10 bps/leg) + `liquidity_tiered_cost(adv)` (10–80 bps by ADV
tier). All engines import these — no scattered literals. **Cost philosophy:** costs are
*not* equal per instrument; the flat default is conservative for ETFs, optimistic for
illiquid names. We handle the uncertainty with cost *stress* (2×/3×), not false
per-name precision. Real per-fill impact gets calibrated live in Phase 6.

### ShortTermReversal — REJECTED (T0046, distinct-family attempt)
We wanted a non-momentum family for breadth. Short-term reversal on the broad
survivorship-free universe: **gross 0.39, but 76× turnover.** Net after tiered costs:
**−0.03 / −0.44 / −0.85 at 1×/2×/3×.** ρ≈0 with everything (a *genuine* diversifier by
correlation) but **non-viable net.** This is the finding that planted the intraday seed:
*reversal is real but needs intraday data to capture.*

### Phase 3.5 outcome — the suite that entered Phase 4
| Sleeve | IS Sharpe | Universe | Note |
|---|---:|---|---|
| TSMomentum(SPY) | 0.55 ✓ | single | equity trend |
| TSMomentum(GLD) | 0.51 ✓ | single | commodity trend, ρ=+0.07 vs equity |
| TSMomentum(TLT) | 0.19 | single | bond diversifier, ρ=−0.29 (weak standalone) |
| LongShortMomentum | 0.41 ✓ | ~598 survivorship-free | dollar-neutral WML, survives 3× cost |
| ~~LowVolAnomaly~~ | rejected | — | disguised 0.65 market beta |
| ~~XSMomentum long-only~~ | non-viable | — | −97% DD; dollar-neutral only |
| ~~ShortTermReversal~~ | rejected | — | gross 0.39 but 76× turnover; net negative; needs intraday |

---

## 6. Phase 4 — Honest validation (Gate 4) → **SUITE FAILS DSR**

**Goal:** the kill test. Does *any* version of this book clear a multiple-testing-corrected
significance bar (Deflated Sharpe Ratio ≥ 0.95)?

Tools (both lock-box-safe, ≤ 2022-07-01):
- `scripts/run_validation.py` — per-sleeve DSR.
- `scripts/run_cpcv.py` — portfolio DSR + CPCV distribution.

The "one canonical engine" question dissolved: the sleeves are two *types* (single-asset
trend via event harness; cross-sectional dollar-neutral via the survivorship harness), so
the common currency is the **daily return series**. DSR/CPCV operate on returns.

### A bug we caught (and why the number is trustworthy)
First DSR run passed *annualized* Sharpe with default `sr_std=1.0` → benchmark 2.24, DSR
0.00 for everything. Fixed: work in *daily* Sharpe units, estimate the benchmark from the
*empirical* cross-trial Sharpe dispersion (~0.45 annualized), filter non-finite trial
Sharpes. **Deflated bar settled at ≈ 1.02 annualized Sharpe** (N=46 trials).

### The verdict
Per-sleeve DSR (max windows): SPY **0.016**, GLD **0.019**, TLT **0.000**, LongShort
**0.001**. No single sleeve is close.

Combined book (common 2010–2022 window):
| Portfolio | Sharpe | MaxDD | DSR | CPCV median | frac+ |
|---|---:|---:|---:|---:|---:|
| equal-weight | 0.61 | −14% | 0.010 | 0.67 | 100% |
| **inverse-vol** | **0.73** | **−11%** | **0.134** | 0.72 | 96% |

**Two things are both true:**
1. The book **passes robustness** — CPCV stability (96% of sub-period paths positive),
   2× cost stress, survivable −11% drawdown, mostly-positive regimes. In absolute terms
   it's an attractive, stable, diversified book in the realistic 0.5–1.5 band.
   *Diversification worked:* higher Sharpe and half the drawdown of any single sleeve.
2. It **fails the significance gate** — after honestly accounting for 46 trials, 0.73
   cannot be distinguished from the best of 46 lucky draws at 95% (DSR 0.134 ≪ 0.95).

**Per the playbook: a failed gate does not graduate. Do not risk real capital.** The gate
did its job — it stopped us deploying an unproven edge.

### Root cause — DATA, not infrastructure
Daily, price-only data yields real-but-modest momentum (0.5–0.7) that can't clear an
honest bar after a realistic search. The uncorrelated families that *would* lift the book
need data we lacked:
- short-horizon reversal → needs **intraday** data;
- value / quality / carry → needs **fundamental** data.

**Decision:** pursue better data, then re-run *this exact pipeline* unchanged.

---

## 7. The "better data" detour — three attempts to clear Gate 4

We tried to add genuinely uncorrelated breadth with free data sources. Result: all real,
all uncorrelated, none enough — and one instructive backfire.

### 7a. EDGAR fundamentals — Value (T0047) and Quality (T0048)
Built a free, PIT-correct SEC EDGAR fundamentals layer (`mft/data_layer/edgar_ingest.py`)
+ `CrossSectionalFactor` sleeve. Two factors:
- **Value (book-to-market):** standalone **0.278**. ρ ≈ 0/negative with all momentum
  sleeves (textbook Asness value-momentum negative correlation). Adding to the book:
  Sharpe 0.73 → 0.85, MaxDD −11% → −7.7%, DSR 0.134 → **0.251.**
- **Quality (Novy-Marx gross profitability, GP/A):** standalone **0.159**. ρ = −0.33 vs
  Value (excellent diversifier), ~0 vs momentum. Stacking: book DSR 0.27 → **0.35**,
  Sharpe → 0.91, MaxDD → −4.1%.

Better, stable, still **fails Gate 4** (0.35 ≪ 0.95).

### 7b. The EDGAR survivorship "fix" that BACKFIRED (rejected, recorded)
Free EDGAR gives clean fundamentals for *survivors* but the value/quality universe was
survivorship-biased (CIK map = current tickers, ~0% delisted vs 35% in the momentum pool).
We tried to de-bias it by matching delisted tickers to historical SEC CIKs
(`cik-lookup-data.txt`, ~1M historical names) and ingesting their fundamentals (385
recovered).

Result *looked* spectacular — value 0.29 → 0.77, portfolio DSR 0.35 → **0.82** — and we
**REJECTED it as a selection-bias artifact:**
- **67% of recovered delisted names ended within 30% of their 1-yr peak (median
  last/peak 0.94)** → the matchable delistings are overwhelmingly orderly *acquisitions*
  (end high, target cheap/value names), **not bankruptcies.** The name-match recovers
  clean acquisition filings but misses the messy/absent bankruptcy filings → value
  *spuriously inflated.*
- Two tells: (1) adding "failed" firms *raised* value — wrong direction for a survivorship
  fix; (2) 2010–2022 was a value *winter*, so 0.77 is implausible.

**Lesson:** free EDGAR cannot produce a clean survivorship-free cross-section. That needs
paid data (CRSP/Compustat). We do **not** count the 0.82. Trustworthy state stands at
value ~0.29, quality ~0.16, portfolio DSR ~0.35. (Full detail in §7 above.)

### 7c. Cross-asset FX momentum (T0049)
A dimension with *no* survivorship or fundamental-bias problem: cross-sectional momentum
across 15 USD currency pairs. Standalone **0.274**, ρ ≈ 0 with the equity book. Survives
2× but not 3× cost. Modest, clean, real — but doesn't move the book past the gate either.

**Net of the detour:** momentum-only DSR 0.14 → +value+quality 0.35 → +FX still ~0.18–0.35
band. Every honest addition helps a little; none clears 0.95. The daily/fundamental free-data
ceiling is real.

---

## 8. Stage 5 + Stage 6 — Building the live machine (frequency-agnostic)

Even though no sleeve graduated, we built the portfolio/risk/execution machinery — because
it's frequency-agnostic and transfers directly to whatever edge *does* eventually clear.

### Stage 5 — Portfolio + risk
- `mft/portfolio/book.py` — `inverse_vol_alloc`, `net_book` (sleeve netting),
  `book_exposure`.
- `mft/portfolio/portfolio.py` — `Portfolio`: composes `inverse_vol_alloc → net_book →
  RiskManager` into one pure, deterministic callable. `construct() → BookResult`,
  `mark_to_market()`, `halted` property.
- Tail metrics added to `mft/validation/metrics.py`: `value_at_risk`, `conditional_var`.
- Tested RiskManager + tail metrics.

### Stage 6 — Paper-trading harness (the book through the live code path)
- `mft/execution/paper_trader.py` — `PaperTrader.run_cycle(as_of, prices) → CycleReport`.
  `CallableSleeve` wraps any function as a sleeve. Uses an `ExecutionAdapter`
  (`SimulatedAdapter` now, IB later) and the `StateStore` for crash recovery.
- This is the same machine that will run live — proving the one-code-path discipline holds
  from research to paper to live.

---

## 9. The overfitting budget (why we can't just keep searching)

- **MinBTL(N)** = minimum backtest length to justify N trials
  (`mft.validation.dsr.min_backtest_length`).
- At 45 trials, MinBTL ≈ 5.0 yrs; we have 23 yrs of daily history → budget ~120+ configs.
  Daily search still has headroom — but each new momentum variant *raises* N and *worsens*
  the DSR bar, so mining more momentum is self-defeating.
- **Intraday is the opposite problem:** only ~3 years of free history → a *tiny* trial
  budget. This is why the intraday work below is deliberately minimal, not a parameter hunt.

---

## 10. THE PIVOT — "I want to build a minute and second mid-freq trading model"

The decisive correction. Everything above was a daily/monthly *factor book*. The
actual goal is a **minute/second-frequency** model that updates and decides continuously.
The daily factor work isn't wasted — the infrastructure (one code path, DSR/CPCV, cost
model, lock-box, portfolio/risk/paper machine) all transfers — but the *frequency* changes
everything about which edges exist.

Decisions made at the pivot (via explicit choice):
- **Start at minute-level** (not second) — prototype the mechanics first.
- **Free data first (Alpaca)** — prove there's something worth paying for *before* paying.

### Security note (still in effect)
Alpaca paper credentials were exposed during development. Action items:
- **API keys go in `.env` only, never in code or notebooks.**
- **If a key was exposed, regenerate it from the dashboard immediately.**
- `.env` is gitignored; do not commit it or any local credentials.

---

## 11. Intraday research (minute frequency, Alpaca free / IEX)

New lock-box for this regime: `INTRADAY_LOCKBOX = 2023-07-01`. Minute annualization:
`BARS_PER_YEAR = 252 × 390 = 98,280`. Microstructure cost: `BASE_COST_PER_TRADE =
0.00015` (1.5 bp/trade, ~3 bp round-trip for liquid names).

### Data layer
- `mft/data_layer/alpaca_ingest.py` — paginated minute-bar fetch (UTC OHLCV + trade
  count + vwap), `save/load_intraday`, keys from env.
- `scripts/ingest_alpaca.py` — 15 liquid mega-caps; **14 succeeded** (QQQ failed). Data
  span **2020-07-27 → 2024-12-31** (research uses ≤ 2023-07-01).

### 11a. Intraday reversal (T0050) — real gross edge, **taker-unprofitable**
`IntradayReversal` (z-score vs rolling mean, fade extremes) — the minute cousin of the
rejected daily ShortTermReversal.
- **GROSS Sharpe 1.46** on AAPL — the signal is *real* at minute frequency.
- **NET −11.39** — 47,600 fills (~one per 6 min); per-trade edge 0.4–2 bp vs 3 bp
  round-trip cost. The spread eats it.
- Raising the threshold to trade less *kills the gross too* (the big moves that clear the
  threshold are momentum, not reversal).

**Conclusion:** minute reversal is unprofitable as a liquidity **taker**; spread > edge
per trade. It's a **maker-only** edge (you'd have to *provide* liquidity, not cross it).

### 11b. Market-making (Path A) — **blocked by free-data quality**
If the reversal edge is maker-only, can we model market-making? Probed free IEX quotes:
**184 bp median AAPL spread** — an artifact of IEX-only BBO (≈2–3% of volume) ≠ the true
NBBO. You cannot model a liquidity-provision edge on quotes that wrong. Real NBBO needs
paid data (Alpaca SIP ~$99/mo or Polygon). **Path A parked** until paid quotes exist.

### 11c. Longer-hold intraday (Path B) — the taker-viable band
The fix for the taker problem: trade *less often* so the per-trade move can exceed the
spread. ~one round-trip/day, decided from pre-holding-period info, held open→close.
Built `mft/features/intraday.py` (`daily_session_features`: gap, opening range, breakout
level — all no-look-ahead, tested) and `mft/backtest/intraday_session.py`.

- **Gap-fade (T0051)** — fade the overnight gap, hold open→close. Equal-weight book NET
  **−0.07.** **Rejected** as a book.
- **Opening-range breakout (T0052)** — trade the first post-opening-range breakout in its
  direction, hold to close (intraday *trend*, not reversal). Equal-weight book NET
  **0.431**, survives 1× cost. **Concentrated in high-beta/tech** (AMZN 1.13, TSLA 1.25,
  AAPL 0.94; defensives negative). **The first taker-viable intraday edge in the project.**
  Caveat: 3 yr only → t ≈ 0.74, not yet significant; IEX data.

### 11d. Opening-range breakout — refinement / pressure-test (T0053)
The candidate was tightened as a pressure-test. `scripts/refine_intraday_breakout.py` — deliberately
minimal (tiny trial budget), economically motivated, **not** a parameter hunt:
1. **Ex-ante high-beta universe** — rank by realized intraday vol (a *characteristic*, not
   the strategy outcome) and take the top half. Intraday trend should live in high-vol names.
2. **Parameter plateau** — opening range 15/30/60 min: is the edge stable across neighbours
   or a lucky single setting?
3. **CPCV** — Sharpe distribution across purged sub-period combinations.
4. **Cost stress.**

Results — the refinement **strengthened** the edge:
| Check | Result | Read |
|---|---|---|
| Ex-ante high-beta vs low-beta | HIGH **0.843** vs LOW **−0.520** | Economically coherent: high-vol names trend, defensives don't |
| Parameter plateau (OR 15/30/60) | **0.46 / 0.84 / 1.48** — all positive | Robust across the range, not a spike |
| CPCV (15 purged paths) | median **0.844**, **80% positive**, min −1.10 max 2.08 | Not carried by one period |
| Significance | t ≈ **1.44** over 2.9 yr | **NOT significant** |
| Cost stress | 0.84 @1× → 0.35 @2× → −0.14 @3× | Real at 1×, marginal at 2×, gone at 3× |

High-beta book = TSLA, NVDA, META, AMZN, XOM, AAPL, GOOGL. CAGR 12.12%, MaxDD −13.89%.

**Verdict (research path):** the best taker-viable signal the project had produced — real,
economically-coherent intraday momentum in high-beta names that got *cleaner* under
disciplined pressure-testing. **But** (1) not statistically significant (t ≈ 1.44 over
2.9 yr), and (2) cost-sensitive. *Caveat that turned out to be decisive: this number used
the research engine's optimistic fill (enter at the breakout LEVEL).* See §11e.

### 11e. Production-path realism check (T0054) — the edge collapses under honest execution
Per the continuation instructions, the frozen candidate was refactored into a production
`AlphaBase` sleeve (`mft/alphas/intraday_orb.py`) and run through a realistic taker
execution simulator (`mft/execution/intraday_sim.py`: next-bar fills, spread/slippage,
forced EOD flat) on the **same** free data. The research entered at the breakout *level*;
honest taker execution enters on the *next bar*.

| Execution | Book Sharpe @1× | @2× | @3× | t |
|---|---:|---:|---:|---:|
| Research (optimistic fill-at-level) | 0.843 | 0.35 | −0.14 | 1.44 |
| **Realistic next-bar taker** | **0.025** | −0.43 | −0.88 | **0.04** |

**Entry-slippage sensitivity** (the cause): at-level Sharpe is 0.843 at +0 bp, 0.515 at
+2 bp, **0.022 at +5 bp (breakeven)**, −0.80 at +10 bp. Two independent methods agree:
next-bar taker ≈ +5 bp adverse entry. **The entire 0.84 "edge" is the first ~5 bp of the
breakout move — the alpha budget is ~5 bp on the entry fill, razor-thin.**

**Conclusion:** ORB-as-taker is **not viable** (fails spec gates G2 + G6 under honest
execution) — the same lesson intraday reversal taught (T0050), now confirmed for the
breakout. Crucially, **longer 1-minute bars cannot resolve this** (minute bars can't model
the breakout-entry fill); only tick/quote data could, and the economics (adverse selection
at a resting stop) argue the realistic fill is *worse* than +5 bp, not better. Full write-up:
`reports/intraday_orb_realism_check_20260604.md`.

### 11f. Afternoon continuation (T0055) — the method works, the signal is too weak
The disciplined response to §11e: build an intraday-momentum sleeve whose edge sits over a
**multi-hour hold** and enters at a **scheduled time** (`mft/alphas/intraday_continuation.py`,
`IntradayAfternoonContinuation`) — sign the morning return at 12:00 ET, hold to close —
so it can't be entry-fragile. Run through the realistic sim from the start.

**The execution-robustness hypothesis was confirmed:** optimistic fill 0.479 → realistic
next-bar 0.382 — a *tiny* haircut, versus ORB's 0.84 → 0.025. Moving the edge off the entry
instant worked exactly as predicted. **But the edge itself is weak:**
- High-beta book **0.382**, **t ≈ 0.65** over 2.9 yr — **not significant**.
- **Dies at 2× cost** (−0.21); cost-sensitive.
- **Name-concentrated:** TSLA 1.13 and AAPL 0.92 carry it; META/AMZN/XOM/GOOGL negative.
- **All-14 book ≈ 0** (−0.06) — not a broad phenomenon on this universe.
- Decision-time is **not a clean plateau** (11:00 = 0.51, 12:00 = 0.38, 13:00 = −0.14).

**Verdict:** the *first* intraday edge to survive honest execution **positive** — a real
methodological win (we now know the structural template that avoids the taker wall) — but
this specific signal is **not deployable and not significant.** It reads as "TSLA + AAPL
trend in the afternoon," not a robust cross-sectional effect. Useful, not graduating.

### 11g. Continuation breadth test (T0056) — the edge is NOT broad; the breadth bet is dead
To decide whether paid broad-universe data was worth buying (the §12 option-1 bet), we ran
the question on FREE data first: ingested 19 more liquid names (Alpaca IEX) → **33 names**,
ranked ex-ante by realized vol, and swept the top-K continuation book.

**Result — a decisive NO.** The book Sharpe *declines monotonically with breadth*:
K=5 → 0.51, K=10 → 0.17, K=20 → −0.12, K=33 → **−0.49**. Only **8 of 33 names are
individually positive** (median −0.33). The edge is **TSLA (1.13), AMD (0.71), NVDA (0.29)**
— 2–3 of the most retail-momentum-chased single names of the **2020–2023 COVID/AI era**,
not a broad cross-sectional anomaly. All three pre-registered breadth criteria fail (Sharpe
should *rise* with K — it falls; t should clear ~1.5 — peaks at 0.87; majority positive —
only 24%).

**Conclusion:** breadth makes it *worse*, not better → **paid broad-universe data is NOT
justified**; option-1 is dead. Adding names dilutes a 2–3-name, regime-specific quirk. This
is the **third** time a free pre-test prevented a bad spend (after ORB minute-bars, §11e).
The "high-beta afternoon continuation" was really "TSLA/AMD/NVDA trended in 2020–23."

---

## 12. Where we are now — the decision, reframed by §11e–g

**Status update (2026-06-04):** we built the full production intraday stack and ran the
frozen ORB candidate through *honest* execution. The earlier "buy longer minute bars to
validate" recommendation is **superseded** — the realism check (§11e) answered the binding
question more cheaply than buying data would have:

- **ORB-as-taker is not viable.** Realistic next-bar execution collapses 0.84 → 0.025
  (t=0.04); it fails its own pre-registered gates (G2, G6). The alpha budget is ~5 bp on
  the entry fill.
- **Buying minute bars would not help** — minute bars can't model the breakout-entry fill,
  which is the whole game. That would have been wasted money. The check just saved it.
- The *only* surviving thread is a **maker / tight-stop** version filling within ~2–5 bp,
  which needs **tick/quote data** (not minute aggregates) and must beat adverse selection.

**The honest engineering, though, is now permanent and reusable:** production `AlphaBase`
ORB sleeve, vendor-agnostic data provider, realistic intraday execution simulator, and
no-leakage + parity + execution tests (19 new tests, 161 total). Whatever intraday idea
comes next plugs straight in. See `docs/INTRADAY_ORB_EXPERIMENT_SPEC.md` for the frozen
gates and `reports/intraday_orb_realism_check_20260604.md` for the verdict.

**Update (2026-06-05):** chose to test an execution-robust structure on free data first —
afternoon continuation (§11f). Result: the **method works** (survives honest execution) but
the **signal is too weak** (0.38, t=0.65, dies at 2×, TSLA+AAPL-concentrated). Two natural
execution-robust intraday-momentum forms now tested: breakout (dies on execution),
continuation (survives execution, too weak). **The free-data / 14-mega-cap / taker-momentum
well is essentially dry** — the per-name edge that exists is too small and too concentrated
to clear an honest bar with only 14 names and 2.9 yr.

**The standing pattern across the whole project:** every signal we find is real but small
relative to honest costs — daily momentum fails DSR; intraday reversal & ORB die at the
spread; continuation survives execution but is sub-significant on a thin universe.

**Update (2026-06-05, cont'd):** ran option-1's free pre-test (§11g). **Breadth bet is
dead** — the continuation edge is 2–3 era-specific names, not broad; paid broad-universe
data is NOT justified. That closes the free-data intraday-taker program: reversal
(taker-dead), breakout (execution-dead), continuation (2–3-name quirk, breadth-dead).

**What is now established with high confidence:**
- Every taker edge on retail-accessible data (daily price, free intraday minute) is too
  small, too concentrated, or too execution-fragile to clear an honest bar. This is a
  *robust, repeatedly-confirmed* finding, not a gap in effort.
- The durable asset is the **honest research factory** that produced these clean kills.

**Remaining honest options (option 1 removed):**
1. **Maker / liquidity-provision thread** — the one genuinely unexplored frontier, and the
   one with prior evidence: intraday reversal (T0050) was real GROSS, dead only as a taker —
   i.e. a *maker* edge (earn the spread, don't pay it). Studying it needs **quote/order-book
   data** (NBBO at least) and a passive-fill execution model. Honest caveat: retail maker is
   a steep climb (queue position, adverse selection, no rebates). A real multi-month program,
   not a backtest.
2. **Pause / bank** — accept that retail-accessible systematic alpha at this frequency is
   genuinely hard; the reusable stack + the documented negative results are the deliverable.
   Resume if/when a new data edge or thesis appears.

No real capital, no data purchase, and no new strategy is committed until you choose.

---

## 13. Intraday IC-first research engine + first two stock families (T0057–T0075)

**Built `mft/research/` — a 5-min cross-sectional research engine, now working end-to-end** on
the same one-code-path discipline. Pipeline: `panel` (1min→5min resample, RTH, aligned) →
`features` (past-only) → `targets` (entry-lagged forward returns, flat overnight) → `signal_lab`
(**IC-first**: rank-IC, IC t-stat, IC-by-month, **bucket returns** + monotonicity, **alpha-decay**)
→ `xs_backtest` (**non-overlapping dollar-neutral long-short** top/bottom-N, gross+net, **cost
sweeps**) → `report` (results/leaderboard/figures) + trial logging. Reuses `costs`/`metrics`/
`dsr`/`trial_log`. Split: train 2020-07-27→2022-12-31, validation →2023-06-30, **lock-box
2023-07-01 sealed and NOT touched** (research on train+val only). 188 tests pass; no-lookahead is
proven by new tests that caught and fixed two real leakage/aggregation bugs before any result.

**Family 1 — short-term residual reversal (T0057–T0064).** Fade recent residual-vs-SPY move;
long losers / short winners. Train rank-IC positive but **tiny** (~0.005–0.01; t inflated by
overlapping windows — directional only), bucket monotonicity mixed. Gross Sharpe real (val 1–4)
but **NET taker-dead**: ≈ −3 to −4 bps/trade at 2 bps/side. **A real but tiny gross mean-reversion
effect, taker-dead after realistic costs.** Independently reproduces T0046/T0050 via a new
cross-sectional path.

**Family 2 — abnormal-volume continuation (T0065–T0075) — REJECTED.** The continuation hypothesis
is **wrong-signed**: train IC is *negative* across all 11 configs → volume-confirmed moves
**reverse, not continue**, on this universe/feed (volume does not flip the reversal sign). Gross
edge per trade (**0.06 bps**) is *smaller* than reversal (0.66 bps); dead by 1 bp/side in the cost
sweep. The only positive configs (strict z>1.5/2 filters) are **rejected as small-sample
artifacts** — ~1 trade/day, no parameter plateau, and they contradict the systematic (negative)
IC. A naive leaderboard would have selected them; the discipline kills them.

**Joint conclusion.** Two families, two independent angles, one structural finding: **current
33-name IEX intraday equity data shows small *gross* reversal effects but no robust taker-tradable
alpha yet.** Costs dominate the per-trade edge. Consistent with the standing project verdict — the
only surviving frame is a possible **maker** (liquidity-provision) edge, not a taker edge.

**Explicitly NOT claimed:** no production alpha; no live-trading readiness.

**Limitations (reaffirmed):** survivorship-biased 33-name (all-survivor, mega-cap) universe; IEX
**partial-volume tape** (~2–3% of consolidated volume — especially damaging to a *volume*-based
signal); **no quote/spread/order-book data**; **costs dominate the per-trade edge**; any real
effect is plausibly a **maker edge, not a taker edge**.

---

## 14. Opening-range breakout — third stock family (T0076–T0084) — REJECTED

Tested ORB through the IC-first engine as a deliberately **directional-continuation** hypothesis
(families 1–2 were both mean-reversion). Built a cross-sectional signed breakout-strength signal
(raw + SPY-adjusted, buffer + volume-gate variants); **9 pre-registered configs (T0076–T0084),
top/bottom-3, lock-box untouched.** This is the CROSS-SECTIONAL reading — single-name *event* ORB
was already taker-dead (T0052–T0054).

**Verdict: REJECTED — a textbook artifact caught by the guardrails.** Several validation backtests
*looked* attractive (config 4 gross Sharpe 2.36; config 8 +3.54 gross bps/trade; headline gross
**+1.27 bps/trade ≈ 2× reversal**). All artifacts:
- **Not IC-supported:** train rank-IC is **negative for every config** (−0.006 to −0.016, t up to
  **−8.4**) with **negative bucket monotonicity** ⇒ stronger breakouts **mildly revert, not continue.**
- **Train/validation sign flip:** headline OR30/hold-60 has **train gross −0.48** vs **validation
  +1.27** bps/trade — the edge flips sign across the split (regime noise, not a stable effect).
- **Both legs positive in validation** (long +4.69, short +2.16 bps) ⇒ **market drift / beta
  contamination**, not clean dollar-neutral alpha (a real one wants long+ / short−).
- **Dies after costs:** net positive only at 0 bps; **−0.73 at 1 bp/side, −2.73 at 2 bps**; gross
  1.27 < 4 bps round-trip. **0 of 9 configs are net-positive with enough trades.**
- **Volume confirmation HURTS** (OR30 gross 1.27→0.33) and collapses trade count (→61) — consistent
  with the IEX **partial-volume tape** being unreliable for volume signals.
- Configs with **too few trades** (cfg 6: 11, cfg 9: 41) or **no OR-window plateau** rejected as artifacts.

### Joint conclusion after three families
Residual reversal (T0057–64), abnormal-volume continuation (T0065–75), and opening-range breakout
(T0076–84) **all point to the same structural finding:** **current 33-name IEX intraday equity data
shows small *gross* mean-reversion effects but no robust taker-tradable alpha.** Directional
continuation has now been **explicitly tested and rejected**. **No production alpha; no live-trading
readiness.** Next recommended direction — one of:
1. **Crypto data ingestion** — complete venue-level tape plus funding / OI / basis features (the
   inputs the IEX equity tape lacks); or
2. **Maker / quote-level equity modeling** — the only surviving equity frame is a possible *maker*
   (liquidity-provision) edge, which needs quote/order-book data.

---

## 15. Crypto Phase 2 — 24/7 backtest fix + BTC→alt lead-lag (T0085–T0094)

**24/7 backtest mode fixed and tested.** Added `continuous=True` to `xs_backtest`: a continuous **UTC
rebalance grid** (no ET-day reset), forward returns no longer nulled at ET-midnight, `tod_windows`
rejected in 24/7 mode. Tests (`tests/test_crypto_backtest.py`) prove the grid is uniform, **weekend
bars are preserved**, and **holds span midnight** correctly. Equity path unchanged.

**BTC→alt lead-lag (perp 5-min, 30 days REAL validated Binance data, T0085–T0094) — SMOKE TEST.**
Hypothesis: BTC leads alts; laggards catch up. Three arms (catch-up via `lag_gap`, continuation,
reversal controls), 10 configs, IC vs **BTC-relative** forward return, continuous 24/7 dollar-neutral
top/bottom-3, crypto taker cost sweep.

- **Sign plausible and IC-supported — notably stronger than equities.** All catch-up configs have
  **positive BTC-relative IC ≈ +0.05, t ≈ 11–12** (≈3–10× the equity-family ICs of ~0.005–0.016);
  continuation control negative, reversal control positive. Gross spread across **22/31 days** (top
  day 10.5%) — not a one-day fluke.
- **KEY INSIGHT — the cross-sectional lead-lag REDUCES TO alt short-term reversal.** BTC's return is
  the same constant for every alt at a timestamp, so `lag_gap = BTC_ret − alt_ret` has the *identical
  cross-sectional ranking* as `−alt_ret` (verified: cfg5 ≡ cfg10 ranks, backtests match exactly). In a
  dollar-neutral cross-section the BTC term **cancels**; the only BTC-dependent element is the
  strong-BTC *gate*. So this is the **same short-horizon reversion** the equity families showed.
- **Gross edge ≈ 0.8–1.7 bps/trade. Taker costs kill it:** crypto taker ~5 bps/side → net ≈ −9
  bps/trade; **dead by 2 bps/side; 0/10 configs net-positive.** (Sharpe magnitudes are the
  near-constant-cost-drag artifact — read bps/trade.)
- **No alpha claimed — 30 days only (one regime).**

**Conclusion:** short-horizon reversion appears **real** (and IC-stronger in crypto), but **taker
execution is the wall** — identical to the standing equity finding. The untested frontier is **maker**
execution (earn the spread instead of paying it), now being prototyped (Maker Execution Phase 1).

---

## 16. Maker Execution Phase 1 — passive-fill sim on the reversal signal (T0095–T0099) — BRANCH REJECTED

**Built and tested a conservative maker-fill simulator** (`mft/research/maker_sim.py`,
`tests/test_maker_sim.py`; **210 tests pass**). Fill models: **touch (optimistic)**, **trade-through
(conservative)**, and **probabilistic** — it **never assumes a fill without price touching / trading
through the limit**. Exit modes: 1 taker, 2 maker-assume-fill, 3 maker-with-taker-fallback; adverse
selection as an explicit per-fill penalty. Tested signal: the crypto short-horizon cross-sectional
reversal (= the BTC lead-lag equivalent).

**Result — maker does NOT rescue the signal; adverse selection is the killer.**
- Maker net ≈ **−12.1 bps/trade** vs taker ≈ **−14.1** — maker is marginally better (saves part of the
  spread) but **both are deeply negative**.
- **Filled-trade gross is NEGATIVE (≈ −4 bps):** passive bids fill when losers *keep falling* (you buy
  the continued-losers). **Unfilled opportunity is +22 bps:** the fast bounces — exactly the trades the
  signal wants — run away from the limit and stay unfilled. **The passive order captures the wrong half
  of the distribution.**
- **Even optimistic assumptions don't produce a believable positive:** the only positive *gross* is
  exit-mode 2 (assume the passive exit always fills) at +0.9, still **−5 net** after fees. Conservative
  fill models are clearly negative (−9.5 to −12).
- **Wider spread assumptions make it WORSE** (gross −2 at 1 bp → −7.3 at 10 bp): a deeper passive order
  only fills on *larger* adverse moves — the opposite of "capture more spread".

**Conclusion:** short-horizon reversal is **predictive but not tradable** under taker OR naive maker
execution. **No alpha claimed; no live-trading readiness. This branch is REJECTED** unless future work
uses richer **order-book / trade-level data** and a proper **queue-position + adverse-selection** model.
**The short-horizon reversal branch is now closed.**

**Next direction (NOT yet run):** shift to **larger-edge / lower-turnover crypto derivative** signals —
(1) funding-rate reversal, (2) open-interest + price positioning, (3) spot-perp basis, and (4) longer
holds (1h / 4h / 8h / 24h).

---

## 17. Crypto Derivatives Phase 1 — funding-rate reversal / crowded positioning (T0100–T0109)

**The first genuinely promising branch in the project** — heavily caveated. Built a funding-aware
backtest (`mft/research/funding_backtest.py`) with **PRICE vs FUNDING-CARRY attribution**, and funding
helpers (`crypto_panel.py`: **known-only 8h→5m mapping**, z-score, cumulative-funding carry, OI change).
Tests prove **no future funding is used** (`tests/test_funding.py`); **216 tests pass**. Three signal
families (A funding mean-reversion, B funding + 4h price confirmation, C funding + OI rising) → **10
configs**, continuous 24/7, dollar-neutral, entry next bar, crypto taker cost sweep. **No alpha claimed.**

- **Funding is a genuinely NEW signal** (positioning, independent of price — does NOT collapse to reversal).
  The **longer-hold / lower-turnover thesis works**: total gross grows with hold — 4h **1.8** → 8h **3.7**
  → 24h **24 bps/trade**, vs the 5-min reversal's ~1.2. Bigger moves clear the fixed cost.
- **First net-positive-after-cost configs in the whole project.** **Config 3** (funding z-score top/bottom
  20%, **24h** hold): **net +14 bps/trade @5 bps/side**, still positive at 10 bps. Some other 24h configs
  survive 5–10 bps too; **4–8h holds still die** at cost.
- **IC is positive and INCREASES with horizon** (Family A 0.006→0.014→**0.031**; **Family B price-confirmed
  0.092, t≈14 at 24h**; Family-A IC-positive 5/5). (t inflated by overlap — honest count is ~25–90 trades.)
- **Funding carry is NEGLIGIBLE (~0.1–0.5 bps/trade, <2% of total); the edge is almost entirely PRICE
  reversion.** So **funding is a positioning SIGNAL, not an income/carry source** — the carry half of the
  hypothesis is debunked.
- **NOT yet alpha — 30-day sample.** **Major caveat:** the 30-day window was a **−25.4% BTC crash**; both
  legs are negative and the book wins only because crowded longs/majors fell *more* (liquidation). The
  result may be **crash/liquidation beta, not general alpha** — indistinguishable on one down-month.
  Per-symbol contribution is spread (LINK/BTC/SOL/BNB +, AVAX/XRP −), so it's not one coin; but 24h configs
  have only ~25 trades.

**Decision:** unlike the reversal/maker branch (closed), this thread **is worth a pre-registered 6–12 month
backfill + regime validation** — kill/confirm test: does the funding-reversal edge persist in **up and
sideways** regimes, or is it only crowded-long liquidation in crashes?

---

## 18. Crypto funding reversal — 365-day backfill / OOS regime validation (T0110–T0119) — REJECTED

**The pre-registered backfill killed it.** The promising 30-day result (§17) was a crash artifact.

### Data backfill (succeeded, clean)
- Backfilled **365 days** of real Binance data: all 10 majors, **clean perp AND spot bars 2025-06-06 →
  2026-06-06**, **525,601 perp + 525,601 spot bars** and **1,095 funding settlements** per symbol.
- Validation **passed**: 100% coverage, 0 gaps, 0 OHLC errors, `clean=True` for all.
- **OI still limited to ~30 days** by the Binance API → OI-gated configs (Family C) cannot be properly
  tested over the full year (no OI before 2026-05 → no signal in train/val).

### Method (no tuning)
Reran the **exact same 10 pre-registered configs** from T0100–T0109 via a shared builder
(`mft/research/funding_signals.py`), on a **pre-registered split: train 60% / val 20% / lock-box 20%**
(lock-box reported separately, never used to tune). **No new configs, no tuning.** Runner:
`scripts/run_crypto_funding_backfill.py`. 216 tests pass.

### Main result — the edge does NOT persist outside the 30-day crash
- **Headline IC collapsed out-of-sample:** config 3 **0.031 → 0.016**; Family-B price-confirmed (cfg7)
  **0.092 (t≈14) → ~0.000**. The strong IC *was* the crash.
- **Gross collapsed:** config 3 **~24 bps/trade (crash) → ~7 bps/trade (full year)** — *below* the ~10 bps
  round-trip taker cost.
- **Net @5 bps/side is NEGATIVE across every split, including the lock-box** (cfg3 lock-box ≈ −10 bps/trade).
- **All 10 configs REJECTED.**

### Regime finding — crash/liquidation short-beta, not alpha
Config 3 net by BTC **trend regime**: **UP negative | DOWN strongly positive | SIDEWAYS strongly negative**
(−670 / +1140 / −1420 bps). It makes money **only in down/crash months** and loses in up and sideways;
positive mainly when BTC subsequently falls. **⇒ crash/liquidation short-beta, not general alpha.**

### Attribution
Edge is still **almost entirely price movement; funding carry remains negligible** (~**+0.23** bps/trade
vs ~**+7.07** bps/trade price PnL). The **funding-carry thesis stays debunked** — funding is at best a
crowded-positioning *signal*, not an income source.

### Conclusion
**Crypto funding reversal is REJECTED as a standalone general MFT alpha.** It may be a crash/liquidation
regime signal, but not a robust alpha. **No alpha claimed; no live-trading readiness. This branch is closed.**
The OOS/regime discipline did its job — it cheaply caught a crash artifact that looked like the project's
first edge.

---

## 19. Crypto cross-sectional MOMENTUM (T0120–T0129) — REJECTED → alpha hunting PAUSED

**The final cheap retail-data alpha test** — the "go slower with momentum" escape hatch. Tested
cross-sectional trend/momentum (long winners, short losers) at 4h/8h/24h horizons — a genuinely
different hypothesis from everything prior (all reversal/breakout/volume/lead-lag/funding). **10
pre-registered configs** (raw / BTC-relative / vol-adjusted / volume-confirmed) on the **365-day clean
Binance dataset**, **same train 60% / val 20% / lock-box 20% split**, funding as context only.
**No new configs, no tuning.** 216 tests pass. Outputs: CSV + `research_logs/crypto_cross_sectional_momentum.md`
+ figures (equity, drawdown, monthly/regime/symbol PnL, IC-by-month, cost sensitivity). Shared eval helpers
extracted to `mft/research/crypto_eval.py`.

### Main result — rejected, and WRONG-SIGNED
- **All 10 configs REJECTED.** No config survives validation **and** lock-box; **net @5 bps/side is negative
  in val and lock-box for every config** (headline cfg3 raw-24h: train −3.6 / val −25.8 / lock −12.0).
- **Gross edge ≈ 1 bps/trade** (cfg3 price +1.3) — far below the ~10 bps round-trip cost.
- **The momentum hypothesis is WRONG-SIGNED: IC is NEGATIVE across all configs (~−0.024 to −0.051).** A
  negative momentum IC means the crypto cross-section **mean-reverts rather than trends at 4h–24h** — the
  same reversal direction found at minutes. Slowing down did NOT reveal momentum.
- **BTC-relative ≡ raw** (BTC is a constant offset at each timestamp, so it drops out of the
  cross-sectional ranking — identical configs/results).
- **Not alpha, not beta:** cfg3 loses across up / down / sideways regimes (−742 / −1125 / −1377 bps). Funding
  a small drag. Per-symbol dispersed and net-negligible.

### Conclusion — alpha hunting PAUSED
The **"go slower with crypto momentum" escape hatch is closed.** Across equities and crypto the project has
now tested **reversal, mean-reversion, breakouts, volume, lead-lag, maker execution, funding/positioning,
and momentum** — **no production-ready alpha found.** Per the pre-registered rule ("if all configs fail,
stop alpha hunting and return to strategy review"), **alpha hunting with current free/retail data is now
paused.** The durable deliverable is the **research factory, the validated data stack, the testing
discipline (pre-registration + lock-box + regime validation), and the documented negative results.** See
`docs/project_review_current_state.md` (Final retail-data MFT verdict).

---

## Appendix A — Complete trial ledger (T0001–T0129)

Source of truth is `trials/trials.csv` (never edited). Summary:

| ID | Strategy | Universe | IS Sharpe | Verdict / note |
|---|---|---|---:|---|
| T0001–03 | SMACrossover | SPY | −0.54 to −0.56 | Gate 0 parity (correctness, not alpha) |
| T0004 | SMACrossover | SPY | 0.61 | Gate 0 parity on real 2010–2026 |
| T0005 | TSMomentum | SPY | 0.68 | first trend pulse (post-lockbox window) |
| T0006 | ShortReversion | SPY | −0.71 | daily reversal fails |
| T0007 | XSMomentum | SPY/QQQ/IWM | 0.73 | (post-lockbox window) |
| T0008 | PairsMeanReversion | SPY/QQQ | −0.59 | deferred |
| T0009 | LowVolAnomaly | 10 names | 0.78 | held for scrutiny → later REJECTED |
| T0010 | LongShortMomentum | 20 names | nan | engine issue, superseded |
| T0011 | LongShortMomentum | 10 names | 0.20 | early WML |
| T0012–18 | TSMomentum | multi-asset | 0.02–0.65 | GLD/IEF best; USO/EURUSD dead |
| T0019–24 | ShortReversion | per-name | −0.95 to 0.34 | mostly fails |
| T0025 | TSMomentum | SPY | **0.556** | CLEAN pre-lockbox ✓ core |
| T0026 | TSMomentum | TLT | 0.19 | clean, weak diversifier |
| T0027 | TSMomentum | GLD | **0.512** | CLEAN pre-lockbox ✓ core |
| T0028–32 | TSMomentum | IEF/EFA/EEM/USO/EURUSD | −0.02 to 0.42 | clean multi-asset scan |
| T0033–35 | XS/LongShort/LowVol | 16 names | nan | engine, superseded by fixed runs |
| T0036–38 | ShortReversion | clean per-name | −0.19 to 0.07 | daily reversal confirmed weak |
| T0039 | XSMomentum | 19 stock | 0.93 | but −0.98 DD → tell, not alpha |
| T0040 | LongShortMomentum | 19 stock | 0.20 | clean |
| T0041 | LowVolAnomaly | 19 stock | nan | superseded |
| T0042 | XSMomentum | 19 stock EW | 0.92 | −0.60 DD |
| T0043 | LongShortMomentum | 19 stock EW | 0.198 | clean |
| T0044 | LowVolAnomaly | 19 stock EW | nan | superseded |
| T0045 | LowVolAnomaly | 19 stock EW | 0.362 | → REJECTED (beta 0.65, hedged −0.03) |
| T0046 | ShortTermReversal | ~600 survivorship-free | −0.028 | REJECTED: gross 0.39, 76× turnover, net neg → needs intraday |
| T0047 | ValueFactor_BM | EDGAR ~745 | 0.278 | real, ρ≈0 vs mom; book DSR 0.13→0.25; fails Gate 4 |
| T0048 | QualityFactor_GPA | EDGAR ~745 | 0.159 | ρ=−0.33 vs value; book DSR →0.35; fails Gate 4 |
| T0049 | FX_XS_Momentum | 15 ccy pairs | 0.274 | clean, ρ≈0 vs equity; survives 2× not 3×; fails Gate 4 |
| T0050 | IntradayReversal | AAPL 1min IEX | −11.39 net | gross 1.46 real; taker-unprofitable; maker-only |
| T0051 | IntradayGapFade | 14 liquid 1min | −0.071 | book ~0; REJECTED |
| T0052 | IntradayORBreakout | 14 liquid 1min | 0.431 | FIRST taker-viable intraday edge; high-beta concentrated |
| T0053 | IntradayORBreakout_HighBeta | 7 ex-ante high-vol | 0.843 | refined (research/optimistic fill); plateau + CPCV stable; t≈1.44 not significant |
| T0054 | IntradayORB_RealisticExec | 7 ex-ante high-vol | 0.025 | PRODUCTION path, honest next-bar taker exec: 0.84→0.025 (2x −0.43); ~5bp entry budget; FAILS gates G2/G6; not viable as taker |
| T0055 | IntradayAfternoonContinuation | 7 ex-ante high-vol | 0.382 | sign(morning ret)@12:00, hold to close. EXECUTION-ROBUST (opt 0.48→real 0.38 — method validated) but WEAK: t=0.65 not significant, dies 2x (−0.21), TSLA+AAPL carry it, all-14 ~0. First intraday edge positive under honest execution; not deployable |
| T0056 | IntradayContinuation_BreadthTest | 33 liquid names | −0.493 | Free breadth diagnostic: book Sharpe DECLINES with breadth (K=5:0.51→K=33:−0.49), 8/33 positive. Edge = TSLA/AMD/NVDA (2020-23 momentum darlings), not broad. Paid breadth NOT justified; option-1 dead |
| T0057–64 | ShortTermReversalXS (5min) | 31 IEX intraday | gross+ / net− | NEW IC-first engine, family 1. Real but TINY gross residual reversal (IC~0.005–0.01, mixed monotonicity); NET taker-dead ≈ −3 to −4 bps/trade @2bps/side. Reproduces T0046/T0050 cross-sectionally. Lock-box sealed |
| T0065–75 | AbnVolContinuationXS (5min) | 31 IEX intraday | REJECTED | Family 2. Continuation WRONG-SIGNED (IC negative across all 11 → volume-confirmed moves revert). Gross 0.06 < reversal 0.66 bps/trade; dead by 1bp/side. "Positive" z>1.5/2 configs rejected as small-sample artifacts (~1 trade/day, no plateau, contradict IC). Lock-box sealed |
| T0076–84 | OpeningRangeBreakoutXS (5min) | 31 IEX intraday | REJECTED | Family 3 (directional continuation). Train rank-IC NEGATIVE all 9 (t to −8.4), neg monotonicity → breakouts mildly REVERT. "Attractive" val gross (cfg4 Sharpe 2.36) is artifact: train/val sign flip (−0.48→+1.27), both legs positive (drift/beta), dead by 1bp/side, 0/9 net+ with enough trades. Volume gate hurts. Lock-box sealed |
| T0085–94 | CryptoBTCLeadLagXS (perp 5min) | 9 USDT alts (Binance) | smoke / taker-dead | Crypto Family 1, 30d REAL data, 24/7. Catch-up IC +0.05 t~11–12 (>equity), 8/8 positive, controls confirm sign. BUT lag_gap ≡ alt short-term reversal (BTC constant cancels x-sec; cfg5≡cfg10). Gross 0.8–1.7 bps/trade; crypto taker ~5bps/side → net ~−9; 0/10 net+. No alpha (30d). Maker frontier next |
| T0095–99 | CryptoMakerReversal (sim) | 9 USDT alts (Binance) | BRANCH REJECTED | Maker-fill PROTOTYPE on crypto reversal. Touch/trade-through/prob fills; never fills w/o touch. Maker net ~−12.1 vs taker ~−14.1 bps/trade; filled gross −4 (ADVERSE SELECTION: fill continued-losers), unfilled opportunity +22 (miss the bounces). Optimistic mode-2 +0.9 gross/−5 net; conservative −9.5 to −12; wider spread worse. Reversal predictive but untradable taker OR naive maker. No alpha. Closed unless order-book/queue data |
| T0100–09 | CryptoFundingReversal | 10 USDT perp majors | PROMISING (caveated) | Crypto Derivatives Phase 1, 30d. NEW positioning signal (not reversal). 3 families/10 configs, price+carry attribution. FIRST net-positive-after-cost: cfg3 (fz 20%, 24h) net +14 bps/trade @5bps. Gross grows w/ hold (4h 1.8→24h 24 vs 5m ~1.2). IC positive, rises w/ horizon (A 0.03, B 0.092 t14 @24h). Carry NEGLIGIBLE (<2%) → funding is SIGNAL not income; edge is price reversion. CAVEAT: 30d = −25% BTC crash → maybe crash/liquidation beta; ~25 trades. Backfill+regime test next. No alpha |
| T0110–19 | CryptoFundingReversalBackfill | 10 USDT perp majors | REJECTED (crash beta) | 365d OOS/regime validation, EXACT same 10 configs, pre-registered train60/val20/lockbox20, no tuning. Data clean (525,601 perp+spot bars, 1,095 funding/sym; OI still ~30d). KILLED IT: IC collapsed (cfg3 0.031→0.016; B cfg7 0.092→0.000); gross 24→7 bps/trade < 10bps cost; net@5bps NEGATIVE all splits incl lockbox (cfg3 −10). Trend: positive ONLY in down months (+1140), neg up (−670)/sideways (−1420) → crash/liquidation short-beta NOT alpha. Carry negligible (price +7 vs fund +0.2). All 10 rejected. Branch closed. No alpha |
| T0120–29 | CryptoXSMomentum | 10 USDT perp majors | REJECTED → hunt PAUSED | FINAL cheap retail-data test. 10 pre-registered momentum configs (raw/btcrel/voladj/volconf, 4h/8h/24h), 365d, same train60/val20/lock20, no tuning. WRONG-SIGNED: IC NEGATIVE all configs (~−0.04) → cross-section MEAN-REVERTS at 4h–24h, not momentum. Gross ~1bp << 10bp cost; net@5bps neg in val AND lockbox all 10; loses up/down/sideways (not alpha, not beta). BTC-rel ≡ raw. All rejected → "go slower" hatch closed → ALPHA HUNTING PAUSED. No alpha |

> **Rejected-ideas index** (so we never re-litigate them blind): daily ShortReversion
> (T0006/19–24/36–38), LowVolAnomaly (disguised beta, T0009/45), XSMomentum long-only
> (−97% DD), PairsMeanReversion (deferred), daily ShortTermReversal (turnover, T0046),
> EDGAR survivorship "fix" (acquisition selection bias — numbers NOT counted), intraday
> reversal as taker (T0050), gap-fade (T0051), ORB-as-taker under honest next-bar
> execution (T0054 — real only with a ~5bp entry fill the free data can't model), afternoon
> continuation breadth (T0056 — a 2–3-name 2020–23 quirk, not broad; breadth makes it worse),
> abnormal-volume continuation (T0065–75 — wrong-signed: IC negative, moves revert not continue;
> "positive" strict-filter configs are small-sample artifacts), opening-range breakout
> cross-sectional continuation (T0076–84 — negative IC, train/val sign flip, both-legs-positive
> drift/beta, dead at 1bp/side, 0/9 net-positive), crypto BTC lead-lag AS A TAKER (T0085–94 —
> reduces to alt short-term reversal; gross 0.8–1.7 bps/trade < crypto taker cost; the reversion
> signal itself is real), crypto short-horizon reversal AS A NAIVE MAKER (T0095–99 — adverse selection
> makes filled gross negative −4 bps while +22 bps of bounces stay unfilled; rejected unless future
> order-book/trade-level data + a queue/adverse-selection model). The short-horizon reversal branch is closed.
> Crypto funding reversal AS A GENERAL ALPHA (T0100–19 — 30d looked promising but the 365-day pre-registered
> backfill showed IC collapse, gross < cost, net-negative OOS incl lock-box, and PnL only in down/crash
> months ⇒ crash/liquidation short-beta, not alpha; funding-carry thesis debunked). Branch closed.
> Crypto cross-sectional MOMENTUM at 4h–24h (T0120–29 — WRONG-SIGNED: IC negative ~−0.04, the cross-section
> mean-reverts not trends even at slower horizons; gross ~1bp << cost; net-negative OOS incl lock-box across
> all regimes; not alpha/not beta). The "go slower with momentum" escape hatch is closed; **free/retail-data
> MFT alpha hunting is paused** — every signal family across equities + crypto has now been rejected.

## Appendix B — Gates status

| Gate | What it proves | Status |
|---|---|---|
| Gate 0 | Engine parity (research == event) | ✅ PASSED |
| Gate 2 | Independent validation engine + crash recovery | ✅ PASSED |
| Gate 3 | Candidate sleeves mutually uncorrelated (\|ρ\|<0.3) | ✅ PASSED |
| Gate 4 | Multiple-testing-corrected significance (DSR ≥ 0.95) | ❌ FAILED (best 0.35; daily/free-data ceiling) |

## Appendix C — Key files (the durable assets)

- `mft/alphas/base.py` — `AlphaBase`, the one interface everything calls.
- `mft/backtest/{vectorbt_harness,event_harness,nautilus_harness}.py` — three parity engines.
- `mft/backtest/survivorship_harness.py` — PIT survivorship-free XS engine (dollar-value).
- `mft/backtest/intraday_session.py` + `mft/features/intraday.py` — Path B intraday machinery.
- `mft/data_layer/{eodhd_ingest,edgar_ingest,alpaca_ingest}.py` — daily / fundamentals / minute.
- `mft/execution/costs.py` — single source of truth for transaction costs.
- `mft/execution/{state,paper_trader}.py` — crash recovery + paper harness.
- `mft/portfolio/{book,portfolio}.py` — allocation + netting + risk in one pure callable.
- `mft/validation/{dsr,cpcv,metrics,diagnostics}.py` — the honesty toolkit.
- `trials/trials.csv` — the ledger. `tests/test_look_ahead.py` — the most important test.
