"""
Intraday cross-sectional alpha research engine (5-minute bars).

Pipeline:  panel -> features -> targets -> signal_lab (IC-first) -> xs_backtest -> report

Discipline (enforced here, tested in tests/test_research_no_lookahead.py):
  - All features are PAST-ONLY (computable at a bar's close from data <= that close).
  - A signal at the close of bar t is tradeable only at bar t+1 (entry lag = 1 bar).
  - Lock-box (intraday 2023-07-01) is never read during research; see splits.py.
  - Universe is 33 survivor mega-caps (survivorship-biased BY CONSTRUCTION) — this
    engine MEASURES signals honestly; the UNIVERSE does not generalize. Never overclaim.
"""
