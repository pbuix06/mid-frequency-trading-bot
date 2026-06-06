# learn/ — driving your own factory

Hands-on lessons that turn "I can't assess what the system produced" into "I can
validate any result myself." Each lesson runs on **your real data** (`data/intraday/`)
and **your real code** (`mft/...`) — nothing is invented for teaching.

## Run

```bash
source .venv/bin/activate
python learn/01_validation_walkthrough.py
```

Or step through it cell-by-cell in VS Code (click **Run Cell** above any `# %%`).

## Lessons

| File | Teaches | The "aha" |
|---|---|---|
| `01_validation_walkthrough.py` | The full validation stack on one real edge (intraday ORB) | You re-derive a Sharpe by hand and it matches the harness; you watch costs and honest fills kill the edge; you see why 56 trials raises the bar; you see what leakage smells like. |

## The 6-question checklist (the whole point)

After Lesson 1 you can interrogate any backtest — mine, a paper's, your own:

1. **Return series** — can I get the daily returns? (everything reduces to this)
2. **By hand** — does `mean/std*sqrt(252)` match the reported Sharpe?
3. **Costs** — is it NET, and does it survive 2× cost stress?
4. **Fills** — what entry fill did it assume? optimistic or honest?
5. **Multiple testing** — how many trials? does DSR clear 0.95 at that N?
6. **Leakage** — is it suspiciously good? could it be peeking at the future?

## Where to go next

- Read `mft/alphas/base.py` (`compute_signal`) — everything is plumbing around it.
- Read `RESEARCH_LOG.md` top-to-bottom — now you can verify each finding it claims.
- Change a number in the walkthrough (a different symbol, a different universe) and
  watch what moves. That's how the intuition sticks.
