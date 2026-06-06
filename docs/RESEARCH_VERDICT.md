# Research verdict

**No production-ready alpha was found. No live trading is approved. No real capital is at risk.**

This document states the project's conclusion directly. It is a negative result, produced
deliberately and documented honestly.

## The verdict in one paragraph

Across ~130 logged trials spanning daily equities, intraday equities, and crypto, **every tested
strategy branch was rejected.** The signals are not all noise — several had genuinely positive
in-sample information coefficients — but none survived the combination of realistic transaction
costs, honest execution assumptions, and out-of-sample / regime validation. **Free and
retail-accessible mid-frequency data is insufficient to deploy a profitable MFT strategy.** The only
remaining serious frontier requires paid order-book / tick / trade-level data and an explicit
queue-position and adverse-selection model — a deliberate, capital-committing program, not a free
experiment.

## Branches tested and why each was rejected

| Branch | Trials | Why it was rejected |
|---|---|---|
| Daily equity factor book (momentum / value / quality / FX) | T0001–T0056 | Real but modest (Sharpe 0.5–0.7); **fails Deflated-Sharpe** after honest multiple-testing (best book 0.35). A significance failure, not execution. |
| Intraday equity residual reversal (5-min) | T0057–T0064 | Real but tiny gross edge; **taker-dead** net (≈ −3 to −4 bps/trade) after realistic costs. |
| Intraday equity abnormal-volume continuation | T0065–T0075 | **Wrong-signed** (IC negative → reverts); volume confirmation hurt (partial IEX tape). |
| Intraday equity opening-range breakout | T0076–T0084 | Negative IC, train/validation **sign flip**, both legs positive (drift/beta, not alpha). |
| Crypto BTC→alt lead-lag / reversal | T0085–T0094 | IC-supported but **mathematically equals alt cross-sectional reversal** (BTC cancels); **taker-dead**. |
| Crypto naive maker reversal (simulation) | T0095–T0099 | **Adverse selection**: passive fills capture the continued-losers (gross −4 bps) while the bounces that carry the edge (+22 bps) stay unfilled. |
| Crypto funding reversal — 30-day smoke | T0100–T0109 | *Looked* like the first cost-surviving edge — but the 30-day window was a crash. |
| Crypto funding reversal — 365-day backfill | T0110–T0119 | **Rejected: crash beta.** IC collapsed, gross fell below cost, net-negative out-of-sample including lock-box; PnL only in down months. |
| Crypto cross-sectional momentum (4h/8h/24h) | T0120–T0129 | **Rejected: wrong-signed.** Negative IC ⇒ the cross-section mean-reverts even at slower horizons; net-negative OOS across all regimes. |

## The three recurring killers

1. **Transaction costs.** At minute-to-hourly frequency the honest per-trade edge (~0.5–2 bps) is
   below the realistic round-trip taker cost (~3–10 bps). Edges only approached cost-viability at
   24-hour holds — i.e. by leaving mid-frequency entirely — and even those failed out-of-sample.
2. **Adverse selection.** The one way to beat the spread is to provide liquidity (maker), but naive
   passive orders are systematically filled on the wrong side of the move. Without a real
   queue/adverse-selection model on order-book data, the maker path is not investable.
3. **Regime confounding.** Short or single-regime samples manufacture edges. The funding signal's
   apparent profitability was a crash-liquidation artifact that vanished — and reversed sign — under
   a pre-registered 12-month split with regime attribution.

## What this means

- **Do not live trade.** There is nothing validated to deploy.
- **Do not buy 10–30 years of equity intraday data.** History length was never the binding
  constraint; cost economics, execution, and regime-robustness are.
- **Do not continue running more free-data MFT configs.** The space has been thoroughly explored;
  additional configs add multiple-testing risk for no expected gain.
- **Do not tune rejected branches.** Re-skinning a rejected idea (e.g. spot-perp basis ≈ funding) is
  re-litigating settled results.

## The next serious frontier (if continued)

**Paid order-book / tick / trade-level data with explicit queue-position and adverse-selection
modelling** — the proper version of the maker thread, and the only un-falsified path to *earning*
rather than *paying* the spread. This is a multi-month, capital- and time-committing program entered
with eyes open. Otherwise, the project should be **banked as a rigorous research-engine deliverable
with strong negative-results documentation** — which is itself a genuine, hard-won contribution.

> Honest negative results, produced under discipline, are worth more than an unvalidated backtest
> that "works." This project chose the former, on purpose.
