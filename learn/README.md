# learn/ — driving your own factory

Hands-on lessons that turn "I can't assess what the system produced" into "I can
validate any result myself." Each lesson runs on **your real data** (`data/intraday/`,
`data/pit/`, `data/crypto/`) and **your real code** (`mft/...`) — nothing is invented
for teaching, and nothing writes to `trials/trials.csv` (re-examining logged results
is free; only new configs spend overfitting budget).

## Run

```bash
source .venv/bin/activate
python learn/01_validation_walkthrough.py    # then 02, then 03
```

Or step through it cell-by-cell in VS Code (click **Run Cell** above any `# %%`).

## Lessons

| File | Teaches | The "aha" |
|---|---|---|
| `01_validation_walkthrough.py` | The full validation stack on one real edge (intraday ORB) | You re-derive a Sharpe by hand and it matches the harness; you watch costs and honest fills kill the edge; you see why 56 trials raises the bar; you see what leakage smells like. |
| `02_gate4_rederivation.py` | The Gate 4 verdict, re-derived from raw sleeves + the ledger | You rebuild the 4-sleeve book (Sharpe 0.73, half the drawdown — diversification works), reconstruct the luck bar (1.02) from the first 46 ledger rows, and compute DSR = 0.134 yourself — the exact recorded number that kept the gate shut. Plus: why re-running today fails even harder (N=129, polluted σ). |
| `03_funding_autopsy.py` | How the discipline unmasked the project's best "edge" as crash beta | The +14 bps/trade funding signal: you show its 25-trade sample flips sign with the backtest clock's start hour, reproduce the pre-registered split that killed it (val +1.9 / lock −10.0), and prove via regime attribution that ALL profit sat in crash months. The checklist grows from 6 questions to 8. |

## The 8-question checklist (the whole point)

After the lessons you can interrogate any backtest — mine, a paper's, your own:

1. **Return series** — can I get the daily returns? (everything reduces to this)
2. **By hand** — does `mean/std*sqrt(252)` match the reported Sharpe?
3. **Costs** — is it NET, and does it survive 2× cost stress?
4. **Fills** — what entry fill did it assume? optimistic or honest?
5. **Multiple testing** — how many trials? does DSR clear 0.95 at that N?
6. **Leakage** — is it suspiciously good? could it be peeking at the future?
7. **Regime** — which months/regimes carry the P&L? One regime = a bet, not an edge.
8. **Sample robustness** — how many trades? do harmless choices (grid phase, window
   start) flip the sign?

## Where to go next

- Read `mft/alphas/base.py` (`compute_signal`) — everything is plumbing around it.
- Read `RESEARCH_LOG.md` top-to-bottom — now you can verify each finding it claims.
- Change a number in a lesson (a different symbol, universe, split, or grid phase) and
  watch what moves. That's how the intuition sticks.
