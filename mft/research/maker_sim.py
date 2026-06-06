"""
Conservative MAKER-fill simulator — can a short-horizon reversal be monetized with passive
limit orders instead of crossing the spread (taker)?

PROTOTYPE / SIMULATION ONLY. We do NOT have real bid/ask quotes, so the spread is a
configurable assumption (`spread_bps`) around the bar close treated as mid. A passive order
only fills if the next bars' low/high actually reach (or trade through) the limit — we never
assume a fill without price touching it. Adverse selection is modelled as an explicit penalty
because passive orders fill preferentially when the move continues against you.

Entry limits (mid = bar close):
    passive BUY  limit = mid * (1 - spread_bps/20000)   (post at the bid, below mid)
    passive SELL limit = mid * (1 + spread_bps/20000)   (post at the ask, above mid)

Fill models:
    A touch       : buy fills if next-bar low <= limit (sell if high >= limit).      OPTIMISTIC.
    B trade-through: buy fills only if low < limit*(1-buffer) (sell if high > limit*(1+buffer)). CONSERVATIVE.
    C probabilistic: fill prob rises with penetration depth past the limit (stylized).  seeded.

Exit modes:
    1 taker exit  : cross the spread at horizon (pay half-spread + taker fee).        conservative.
    2 maker exit  : post passive, ASSUME it fills at the favourable price (maker fee). optimistic.
    3 maker w/ taker fallback: try passive exit for `timeout` bars, else taker.        realistic.

Every realized return is built from ACTUAL fill prices; the `adverse_selection_bps` penalty is
layered on top (per passive fill) to stress the microstructure selection the bars cannot show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MakerConfig:
    spread_bps: float = 5.0
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 5.0
    adverse_selection_bps: float = 2.0
    timeout_bars: int = 2
    model: str = "B"          # 'A' touch | 'B' trade-through | 'C' probabilistic
    buffer_bps: float = 1.0   # model B trade-through buffer
    exit_mode: int = 1        # 1 taker | 2 maker(assume fill) | 3 maker w/ taker fallback
    hold_bars: int = 6        # horizon


def limit_price(mid: float, side: str, spread_bps: float) -> float:
    half = spread_bps / 20000.0
    return mid * (1.0 - half) if side == "buy" else mid * (1.0 + half)


def _passive_fill(high: np.ndarray, low: np.ndarray, start: int, n: int,
                  limit: float, side: str, cfg: MakerConfig, rng) -> int:
    """First bar in (start, start+timeout] where a passive order at `limit` fills; -1 if none."""
    end = min(start + cfg.timeout_bars, n - 1)
    for b in range(start + 1, end + 1):
        if side == "buy":
            lo = low[b]
            if cfg.model == "A":
                if lo <= limit:
                    return b
            elif cfg.model == "B":
                if lo < limit * (1.0 - cfg.buffer_bps * 1e-4):
                    return b
            else:  # C — penetration-scaled probability
                pen = max(0.0, (limit - lo) / limit)
                if rng.random() < min(1.0, pen / max(cfg.spread_bps * 1e-4, 1e-9)):
                    return b
        else:  # sell
            hi = high[b]
            if cfg.model == "A":
                if hi >= limit:
                    return b
            elif cfg.model == "B":
                if hi > limit * (1.0 + cfg.buffer_bps * 1e-4):
                    return b
            else:
                pen = max(0.0, (hi - limit) / limit)
                if rng.random() < min(1.0, pen / max(cfg.spread_bps * 1e-4, 1e-9)):
                    return b
    return -1


def _exit_price(close, high, low, n, fill_bar, side, cfg, rng) -> tuple[float, float, bool]:
    """Return (exit_price, exit_fee_bps, was_maker_exit) at the horizon from the fill bar."""
    half = cfg.spread_bps / 20000.0
    x = min(fill_bar + cfg.hold_bars, n - 1)
    mid_x = close[x]
    exit_side = "sell" if side == "buy" else "buy"   # long sells to exit; short buys

    if cfg.exit_mode == 2:  # assume passive exit fills at the favourable price
        px = mid_x * (1 + half) if side == "buy" else mid_x * (1 - half)
        return px, cfg.maker_fee_bps, True

    if cfg.exit_mode == 3:  # passive exit, taker fallback
        elim = limit_price(mid_x, exit_side, cfg.spread_bps)
        fb = _passive_fill(high, low, x, n, elim, exit_side, cfg, rng)
        if fb != -1:
            return elim, cfg.maker_fee_bps, True
        xb = min(x + cfg.timeout_bars, n - 1)
        mid_fb = close[xb]
        px = mid_fb * (1 - half) if side == "buy" else mid_fb * (1 + half)
        return px, cfg.taker_fee_bps, False

    # exit_mode 1: taker cross at horizon
    px = mid_x * (1 - half) if side == "buy" else mid_x * (1 + half)
    return px, cfg.taker_fee_bps, False


def simulate(events: list[tuple[str, int, str]], bars: dict[str, dict],
             cfg: MakerConfig, seed: int = 0) -> pd.DataFrame:
    """
    Simulate passive fills for entry events.

    events: list of (symbol, bar_index, side) with side in {'buy','sell'}.
    bars:   {symbol: {'close','high','low': np.ndarray, 'ts': DatetimeIndex}}.
    Returns one row per event: filled?, delay, gross/net returns, and the mid-to-mid
    signal return (used for unfilled opportunity cost).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sym, t, side in events:
        d = bars[sym]
        close, high, low, n = d["close"], d["high"], d["low"], len(d["close"])
        if t + 1 >= n:
            continue
        mid_t = close[t]
        lim = limit_price(mid_t, side, cfg.spread_bps)
        x0 = min(t + cfg.hold_bars, n - 1)
        signal_ret = (close[x0] / mid_t - 1) if side == "buy" else (mid_t / close[x0] - 1)

        f = _passive_fill(high, low, t, n, lim, side, cfg, rng)
        if f == -1:
            rows.append(dict(sym=sym, ts=d["ts"][t], side=side, filled=False, delay=np.nan,
                             gross=np.nan, net_fees=np.nan, net_adv=np.nan, signal_ret=signal_ret))
            continue

        exit_price, exit_fee, maker_exit = _exit_price(close, high, low, n, f, side, cfg, rng)
        gross = (exit_price / lim - 1) if side == "buy" else (lim / exit_price - 1)
        fees = (cfg.maker_fee_bps + exit_fee) * 1e-4
        adv = cfg.adverse_selection_bps * 1e-4 * (2 if maker_exit else 1)  # each passive fill is selected
        rows.append(dict(sym=sym, ts=d["ts"][f], side=side, filled=True, delay=int(f - t),
                         gross=gross, net_fees=gross - fees, net_adv=gross - fees - adv,
                         signal_ret=signal_ret))
    return pd.DataFrame(rows)


def aggregate(trades: pd.DataFrame, years: float) -> dict:
    """Fill/economics metrics. `years` = data span in years (for annualizing the filled Sharpe)."""
    n = len(trades)
    f = trades[trades["filled"]]
    nf = trades[~trades["filled"]]
    nfill = len(f)
    out = {
        "n_events": n, "n_filled": nfill,
        "fill_rate": round(nfill / n, 4) if n else float("nan"),
        "avg_fill_delay_bars": round(float(f["delay"].mean()), 3) if nfill else float("nan"),
        "gross_bps": round(float(f["gross"].mean()) * 1e4, 3) if nfill else float("nan"),
        "net_fees_bps": round(float(f["net_fees"].mean()) * 1e4, 3) if nfill else float("nan"),
        "net_adv_bps": round(float(f["net_adv"].mean()) * 1e4, 3) if nfill else float("nan"),
        "unfilled_opp_bps": round(float(nf["signal_ret"].mean()) * 1e4, 3) if len(nf) else 0.0,
        "win_rate": round(float((f["net_adv"] > 0).mean()), 4) if nfill else float("nan"),
        "avg_win_bps": round(float(f.loc[f["net_adv"] > 0, "net_adv"].mean()) * 1e4, 3) if (f["net_adv"] > 0).any() else 0.0,
        "avg_loss_bps": round(float(f.loc[f["net_adv"] < 0, "net_adv"].mean()) * 1e4, 3) if (f["net_adv"] < 0).any() else 0.0,
    }
    if nfill > 2 and f["net_adv"].std() > 0:
        ppy = nfill / max(years, 1e-9)
        out["sharpe_filled"] = round(float(f["net_adv"].mean() / f["net_adv"].std() * np.sqrt(ppy)), 2)
    else:
        out["sharpe_filled"] = float("nan")
    return out
