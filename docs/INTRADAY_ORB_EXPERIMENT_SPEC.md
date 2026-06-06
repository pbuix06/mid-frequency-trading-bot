# Intraday Opening-Range Breakout (ORB) — Experiment Spec

> **This spec is frozen before new-data validation.** The hypothesis, universe rule,
> splits, execution assumptions, cost model, metrics, and pass/fail gates below are
> fixed *now*. They may not be loosened after seeing results. If a design choice must
> change, that is a NEW experiment with a NEW spec version and its own trials — it does
> not retroactively rescue this one.
>
> Spec version: **v1** · Author: Phong Bui · Date: 2026-06-04 · Status: **frozen candidate, awaiting better-data validation**
> Governing rules: `docs/RESEARCH_PROCESS.md`. Lineage: `RESEARCH_LOG.md` §11.
>
> **UPDATE 2026-06-04 (free-data realism check, T0054):** before any paid data, the frozen
> candidate was run through the production path with honest taker execution (next-bar
> fills). It **FAILS gates G2 + G6** — Sharpe 0.84 → 0.025 — because the alpha budget is
> only ~5 bp on the breakout entry fill. Minute bars cannot resolve that fill; only
> tick/quote data could. Step 7 (paid minute-bar validation) is therefore **not the right
> next spend**. See `reports/intraday_orb_realism_check_20260604.md` and §9 below — the
> gates stand; the candidate did not clear them under honest execution.

---

## 0. One-sentence purpose

Decide — on long, clean intraday data with realistic execution — whether the high-beta
intraday opening-range breakout is a **repeatable, cost-survivable, regime-robust edge
worth paper trading**, or a **3-year free-data artifact to archive**.

This spec does **not** try to improve the strategy. It tries to *kill or confirm* the
frozen candidate. Improvement only happens after a confirm.

---

## 1. Hypothesis (economic rationale — frozen)

> **High-volatility, high-attention U.S. equities that break their opening range tend to
> continue in the breakout direction during the same session**, because early order
> imbalance, attention/flow, stop-triggering, and volatility-chasing create short-horizon
> intraday trend that persists into the close.

Falsifiable predictions this implies (used as sanity checks, not extra knobs):
- The edge should be **stronger in higher-realized-vol names** (where flow/attention
  effects are larger) and **weak or absent in low-vol defensives**.
  *Free-data evidence: HIGH-beta book 0.84 vs LOW-beta −0.52. Consistent.*
- The edge should **not** depend on a single opening-range definition (it is a behavior,
  not a parameter). *Free-data evidence: positive across OR 15/30/60. Consistent.*
- The edge should **not** be carried by one stock or one year.
  *Untested on free data — a primary check for this experiment (Gate G5/G6).*

If the better-data result contradicts the first two predictions, the hypothesis is
**rejected** regardless of headline Sharpe — a number without its mechanism is noise.

---

## 2. Strategy definition (frozen)

| Element | Definition |
|---|---|
| **Frequency** | 1-minute bars (regular session only). |
| **Decision** | Per symbol, per session: flat until the opening range completes; on the first post-OR breakout, take a position in the breakout direction; hold to close. |
| **Direction** | +1 (long) if price first breaks **above** the opening-range high; −1 (short) if it first breaks **below** the opening-range low; 0 if neither. |
| **Holding** | Intraday only. **Forced flat by session close. No overnight exposure, ever.** |
| **Opening-range lengths** | 15 / 30 / 60 minutes. **30 is the primary; 15 and 60 are the plateau check only** (not free parameters to pick the best of). The reported headline uses OR=30. |
| **Sizing** | Equal-weight across the book (per-name target ±1, book = mean). Vol-targeting is a *post-confirmation* improvement, out of scope here. |
| **Session** | 09:30–16:00 America/New_York. Pre/post-market excluded. |

**One code path:** the canonical logic lives in `mft/alphas/intraday_orb.py`
(`AlphaBase`-compatible, per-minute `compute_signal`). The research session backtest
(`mft/backtest/intraday_session.py`) is a fast approximation and must be shown to agree
with the alpha sleeve (parity test). No strategy logic may exist only in a script.

---

## 3. Universe rule (ex-ante — frozen)

The universe must be selected using **only information knowable before the trading
period**. No hand-picking by outcome.

**Rule:** at each rebalance (monthly), from liquid U.S. equities passing:
- minimum average daily dollar volume (ADV) — large/liquid only,
- minimum price (avoid sub-$5 microstructure),
then **rank by prior-window realized intraday volatility and take the top half (or top
quantile)**. That ranked-high-vol set is the tradable book for the next period.

**Freezing note on the current 7 names** (TSLA, NVDA, META, AMZN, XOM, AAPL, GOOGL):
these are the *output* of applying "top-half by realized intraday vol" to the 14 free-data
names — they are an **illustration of the rule, not a hardcoded universe.** On better
data the rule is re-applied; the resulting names may differ and that is expected. **Do not
hardcode this list as the permanent universe.** A test must assert the selection uses only
past data (Gate G-leak).

---

## 4. Data period & splits (frozen — define before looking)

Free Alpaca/IEX data (2020-07-27 → 2023-06-30, ≤ `INTRADAY_LOCKBOX = 2023-07-01`) was
enough to *discover* the candidate. It is **not** enough to *prove* it (2.9 yr, one
regime, IEX-only ≈ 2–3% of volume).

**Target data for this experiment:** ≥ 8–10 years of corporate-action-adjusted 1-minute
OHLCV with reliable volume and, ideally, spread/quote estimates (vendor TBD — Polygon /
Databento / AlgoSeek / etc., see continuation instructions §8).

**Splits (fixed before any new-data run):**
| Split | Window | Use |
|---|---|---|
| **Development** | earliest available → cutoff D (e.g. first ~60% of history) | re-run frozen candidate; all diagnostics; any (rare) re-spec lives here |
| **Validation** | cutoff D → cutoff V (next ~25%) | confirm the development read holds out-of-sample, unseen |
| **Final lockbox** | cutoff V → most recent (~last 15%) | opened **once**, at the very end, for the final exam |

The exact dates are set in the validation report header *before* the first run, once the
vendor/history length is known. The final lockbox is never inspected more than once.
`INTRADAY_LOCKBOX = 2023-07-01` remains untouched as the free-data holdout.

---

## 5. Execution assumptions (frozen — costs are part of the model)

For minute-level trading the edge and the fill are inseparable. Every simulated trade
models:

- **Spread crossing:** enter buy@ask / sell@bid; exit sell@bid / buy@ask. Pay the spread.
- **Commission + slippage** on top of spread.
- **Next-bar execution.** Signal is computed on a **closed** bar; the order fills on the
  **next** bar (open, or a participation-capped VWAP). **No same-bar fill at the exact
  breakout level** — the free-data research used an optimistic "fill at OR level" which
  this experiment explicitly replaces with next-bar realism.
- **Ambiguous-bar safety:** if a bar both breaks out and reverses, assume **unfavorable**
  ordering. Never assume favorable intrabar sequencing without tick data.
- **Participation cap:** order size capped at a fraction of the bar's volume; excess is a
  **skipped/partial** trade, logged.
- **Skip rules:** skip if spread too wide or bar volume too low (logged with reason).
- **EOD liquidation:** forced flat at a defined pre-close time; liquidation cost modeled.

**Baseline cost:** `BASE_COST_PER_TRADE = 0.00015` (1.5 bp/trade ≈ 3 bp round-trip for
liquid names), the existing single-source value in `mft/execution/costs.py`. On better
data with spread estimates, cost becomes **per-symbol/per-time**, not a flat constant.

---

## 6. Cost stress (frozen)

Report every headline at **1× / 2× / 3×** base cost.
- Dies at **1×** → reject outright.
- Dies at **2×** → fragile; record but do not advance to paper.
- Survives **2×**, dies only at **3×** → may advance, flagged cost-sensitive.

*Free-data baseline to beat: 0.84 @1× → 0.35 @2× → −0.14 @3× (survives 2×, dies 3×).*

---

## 7. Risk limits (frozen — enforced in any paper/live path)

- Max position per symbol; max gross exposure; max daily loss; max-drawdown halt.
- Max order size as % of minute volume.
- No trading outside the allowed session; **forced flat by close**; kill switch.

(These bind the execution/paper/live machine, not the backtest signal, but are part of
the candidate's definition — a strategy that needs more than these limits is a different
strategy.)

---

## 8. Metrics to report (frozen — never just the headline Sharpe)

Full-period **and** per-split: Sharpe, ann. return, ann. vol, MaxDD, turnover, avg
trades/day, avg holding time, win rate, avg win/avg loss. Decomposition: **PnL by symbol,
by year, by market regime, by opening-range length.** Robustness: **cost stress 1×/2×/3×,
CPCV (distribution + median + frac positive), DSR, sensitivity to entry/exit timing.**

A report that shows only the headline Sharpe is **non-compliant** with this spec.

---

## 9. Pass/fail gates (FROZEN — set before results are seen)

The candidate may advance **to paper trading** only if **all hard gates pass** and **≥ 4
of 5 soft gates pass**. These thresholds may not be lowered after seeing results.

### Hard gates (all required)
| ID | Gate | Threshold |
|---|---|---|
| **G1** | No-leakage tests | All pass (universe, OR, breakout, execution, EOD all use only past data) |
| **G2** | Long-data Sharpe | Meaningfully positive over the full development+validation history (materially > 0, not marginal) |
| **G3** | Out-of-sample holds | Validation-split Sharpe positive and directionally consistent with development |
| **G4** | DSR improvement | DSR **materially improved** vs the current 0.13 free-data result, accounting for the *cumulative* trial count |
| **G5** | Not one regime | Positive in **> 1** distinct market regime; not solely a 2020–2023 artifact |
| **G6** | Cost survival | **≥ 0** at **2×** base cost (not destroyed); 3× not catastrophic |

### Soft gates (≥ 4 of 5)
| ID | Gate | Threshold |
|---|---|---|
| **S1** | CPCV median | Positive |
| **S2** | CPCV breadth | ≥ **70%** of paths positive |
| **S3** | Year concentration | No single year contributes an unacceptable majority of total PnL |
| **S4** | Name concentration | No single stock contributes the majority of total PnL |
| **S5** | Trade count | Large enough for inference (thousands of trades, not dozens) |

**Outcomes:**
- **Pass** → advance to paper trading (engineering + slippage measurement), then a micro-live
  diagnostic checklist. Still not "proven profitable" — paper is plumbing, not alpha proof.
- **Fail** → archive honestly in `RESEARCH_LOG.md` with the reason; preserve lessons; the
  candidate does **not** trade. Then pick **one** economically distinct intraday family.

---

## 10. Trial-counting rules (frozen)

- **Every** run on the new data — each universe variant, each OR length, each cost
  assumption, each filter — is **one row** in `trials/trials.csv`. No exceptions.
- The plateau check (OR 15/30/60) is **3 trials**, not a free "best of three." The headline
  is OR=30, fixed in advance.
- DSR/MinBTL accounting uses the **cumulative** trial count across the whole project (now
  53), because multiple testing does not reset when the data source changes.
- Re-running the **frozen, unchanged** candidate on new data is the cheap path; *searching*
  on the new data spends budget fast and raises the bar — minimize it (continuation
  instructions §10/§3.2).

---

## 11. Explicit non-goals (out of scope for this spec)

- Improving/tuning the strategy (only after a confirmed pass).
- New alpha families (only after this candidate is confirmed or rejected).
- Vol-targeting / leg-weighting (a post-confirmation improvement).
- Market-making / second-level / maker execution (needs quote/order-book data; separate spec).
- Treating paper-trading PnL as alpha proof.

---

## 12. Definition of done

This experiment is **done** when: the frozen candidate has been run unchanged on ≥8–10 yr
of realistic-execution data; the full metric set (§8) is reported in
`reports/intraday_orb_validation_<date>.md`; every gate in §9 is evaluated against its
pre-registered threshold; and the binary **advance-to-paper / archive** decision is recorded
in `RESEARCH_LOG.md` with its reasoning. No headline-only summaries; no moved gates.
