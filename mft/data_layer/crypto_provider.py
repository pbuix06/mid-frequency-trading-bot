"""
Crypto data interface — DESIGN ONLY. No crypto data exists yet (verified: data/ has none).

This mirrors mft.data_layer.intraday_provider.IntradayProvider so the research engine
(panel -> features -> targets -> signal_lab -> xs_backtest) can consume crypto bars through
the SAME code path as equities, once a real vendor is wired. Until then every fetch raises
NotImplementedError — we will NOT fabricate crypto results.

Normalized crypto bar schema (UTC index), a superset of the equity schema:

    required:  open, high, low, close, volume
    optional:  vwap, trades, quote_volume
    perp-only: funding_rate     (8h funding, forward-filled to bar; sign = longs pay shorts>0)
               open_interest    (contracts or USD notional, vendor-stated)
               mark_price, index_price
    derived:   basis = perp_close / spot_close - 1   (needs spot+perp aligned)
    stamped:   df.attrs["symbol"], df.attrs["source"], df.attrs["market"]  # 'spot'|'perp'
               df.attrs["exchange"], df.attrs["quote"]  # e.g. 'USDT'

Why this shape: the crypto alpha families you specified need exactly these fields —
  - momentum / reversal / lead-lag : OHLCV only (works today once OHLCV lands)
  - funding-rate reversal          : funding_rate
  - OI + price                     : open_interest
  - spot-perp basis                : spot close + perp close (two providers, aligned)

See docs/crypto_data_requirements.md for vendors, exact columns, and acquisition steps.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["vwap", "trades", "quote_volume", "taker_buy_base", "taker_buy_quote"]
PERP_COLUMNS = ["funding_rate", "open_interest", "mark_price", "index_price"]
NORMALIZED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + PERP_COLUMNS

# Binance endpoints (public, no key needed for historical market data).
SPOT_BASE = "https://api.binance.com/api/v3"
FAPI_BASE = "https://fapi.binance.com"
_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}


def normalize_crypto_bars(
    df: pd.DataFrame, symbol: str, source: str, market: str = "spot",
    exchange: str = "", quote: str = "USDT",
) -> pd.DataFrame:
    """
    Coerce a vendor crypto frame into the normalized schema. Network-free, deterministic.

    Enforces: UTC tz-aware DatetimeIndex (crypto is 24/7 — NO session masking), sorted,
    de-duplicated, required columns present, optional/perp columns created as NaN when
    absent, close > 0. Stamps symbol/source/market/exchange/quote into df.attrs.

    This is intentionally implemented (it is pure data hygiene) so a future ingester only
    has to fetch + map columns, not re-derive the contract.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}:{symbol} missing required columns {missing}")

    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()

    for col in OPTIONAL_COLUMNS + PERP_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[NORMALIZED_COLUMNS].astype({c: float for c in REQUIRED_COLUMNS})
    out = out[out["close"] > 0]
    out.attrs.update(symbol=symbol, source=source, market=market,
                     exchange=exchange, quote=quote)
    return out


class CryptoProvider(ABC):
    """A source of normalized crypto bars. Subclass per vendor (Binance, Bybit, ...)."""

    name: str = "abstract"
    market: str = "spot"  # 'spot' or 'perp'

    @abstractmethod
    def get_bars(self, symbol: str, start: str, end: str, freq: str = "5min") -> pd.DataFrame:
        """Return normalized bars for [start, end] UTC. See NORMALIZED_COLUMNS."""
        ...

    @abstractmethod
    def get_funding(self, symbol: str, start: str, end: str) -> pd.Series:
        """Perp funding-rate series (UTC). Spot providers may raise NotImplementedError."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.name!r}, market={self.market!r})"


class _UnimplementedCryptoProvider(CryptoProvider):
    """
    Placeholder so the interface imports and tests can assert the contract — but it
    refuses to return data. Replace with a real BinanceProvider / BybitProvider that
    fetches + maps columns + calls normalize_crypto_bars(...). DO NOT fake bars here.
    """

    name = "UNIMPLEMENTED"

    def get_bars(self, symbol: str, start: str, end: str, freq: str = "5min") -> pd.DataFrame:
        raise NotImplementedError(
            "No crypto vendor wired yet. See docs/crypto_data_requirements.md — acquire "
            "real OHLCV (+funding/OI for perps) and implement a CryptoProvider subclass."
        )

    def get_funding(self, symbol: str, start: str, end: str) -> pd.Series:
        raise NotImplementedError("Funding requires a perp vendor (Binance/Bybit USDT-perp).")


# ── Concrete Binance providers (real public data; no API key required) ────────

def _get_json(session, url: str, params: dict, max_retries: int = 6):
    """GET with Binance rate-limit (418/429) backoff. Returns parsed JSON."""
    import requests
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=25)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):  # rate-limited / IP-banned: back off
            wait = int(r.headers.get("Retry-After", 0)) or (2 ** attempt)
            time.sleep(min(wait, 60))
            continue
        raise requests.HTTPError(f"{r.status_code} {url} {params}: {r.text[:200]}")
    raise RuntimeError(f"giving up after {max_retries} retries: {url} {params}")


def _klines_to_frame(rows: list) -> pd.DataFrame:
    """Binance 12-field kline rows -> normalized OHLCV frame (UTC open-time index)."""
    if not rows:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    idx = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    num = ["open", "high", "low", "close", "volume", "quote_volume",
           "taker_buy_base", "taker_buy_quote"]
    out = df[num].apply(pd.to_numeric, errors="coerce")
    out["trades"] = pd.to_numeric(df["trades"], errors="coerce")
    out.index = idx
    out.index.name = "timestamp"
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_klines(session, base: str, path: str, symbol: str, interval: str,
                 start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    """Paginate klines [start_ms, end_ms) for one symbol. base = SPOT_BASE or FAPI_BASE."""
    step = _INTERVAL_MS[interval]
    out, cur = [], start_ms
    while cur < end_ms:
        rows = _get_json(session, f"{base}{path}",
                         {"symbol": symbol, "interval": interval, "startTime": cur,
                          "endTime": end_ms, "limit": limit})
        if not rows:
            break
        out.extend(rows)
        nxt = int(rows[-1][0]) + step
        if nxt <= cur or len(rows) < limit:
            cur = nxt
            if len(rows) < limit:
                break
        else:
            cur = nxt
        time.sleep(0.06)  # polite pacing
    return _klines_to_frame(out)


class BinanceSpotProvider(CryptoProvider):
    """Spot 1m/5m klines from api.binance.com (public)."""

    name = "binance"
    market = "spot"

    def __init__(self):
        import requests
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "mft-research/1.0"})

    def get_bars(self, symbol: str, start: str, end: str, freq: str = "1m") -> pd.DataFrame:
        s = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        e = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        raw = fetch_klines(self._s, SPOT_BASE, "/klines", symbol, freq, s, e, 1000)
        return normalize_crypto_bars(raw, symbol, "binance", market="spot",
                                     exchange="binance", quote="USDT")

    def get_funding(self, symbol: str, start: str, end: str) -> pd.Series:
        raise NotImplementedError("funding is a perp concept — use BinancePerpProvider.")


class BinancePerpProvider(CryptoProvider):
    """USDT-margined perpetual: klines + funding + open interest from fapi.binance.com."""

    name = "binance"
    market = "perp"

    def __init__(self):
        import requests
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "mft-research/1.0"})

    def get_bars(self, symbol: str, start: str, end: str, freq: str = "1m") -> pd.DataFrame:
        s = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        e = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        raw = fetch_klines(self._s, FAPI_BASE, "/fapi/v1/klines", symbol, freq, s, e, 1500)
        return normalize_crypto_bars(raw, symbol, "binance", market="perp",
                                     exchange="binance", quote="USDT")

    def get_funding(self, symbol: str, start: str, end: str) -> pd.Series:
        """8-hourly funding rate (UTC index). markPrice carried in .attrs is dropped here."""
        s = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        e = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        out, cur = [], s
        while cur < e:
            rows = _get_json(self._s, f"{FAPI_BASE}/fapi/v1/fundingRate",
                             {"symbol": symbol, "startTime": cur, "endTime": e, "limit": 1000})
            if not rows:
                break
            out.extend(rows)
            cur = int(rows[-1]["fundingTime"]) + 1
            if len(rows) < 1000:
                break
            time.sleep(0.06)
        if not out:
            return pd.Series(dtype=float, name="funding_rate")
        df = pd.DataFrame(out)
        idx = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
        ser = pd.Series(pd.to_numeric(df["fundingRate"], errors="coerce").values,
                        index=idx, name="funding_rate")
        ser.index.name = "timestamp"
        return ser[~ser.index.duplicated(keep="last")].sort_index()

    def get_open_interest(self, symbol: str, start: str, end: str, period: str = "5m") -> pd.DataFrame:
        """Open interest history (Binance retains ~30 days). Columns: open_interest, oi_value."""
        s = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        e = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        step = _INTERVAL_MS[period] * 500
        out, cur = [], s
        while cur < e:
            rows = _get_json(self._s, f"{FAPI_BASE}/futures/data/openInterestHist",
                             {"symbol": symbol, "period": period, "startTime": cur,
                              "endTime": min(cur + step, e), "limit": 500})
            if not rows:
                cur += step
                continue
            out.extend(rows)
            cur = int(rows[-1]["timestamp"]) + _INTERVAL_MS[period]
            time.sleep(0.06)
        if not out:
            return pd.DataFrame(columns=["open_interest", "oi_value"])
        df = pd.DataFrame(out)
        idx = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        res = pd.DataFrame({
            "open_interest": pd.to_numeric(df["sumOpenInterest"], errors="coerce").values,
            "oi_value": pd.to_numeric(df["sumOpenInterestValue"], errors="coerce").values,
        }, index=idx)
        res.index.name = "timestamp"
        return res[~res.index.duplicated(keep="last")].sort_index()
