"""
Train / validation / lock-box splits for intraday research.

Discipline:
  - TRAIN      : hypothesis generation, feature/signal construction.
  - VALIDATION : the ONLY out-of-sample number you may rank/select on.
  - LOCK-BOX   : sealed final exam. Read EXACTLY ONCE, at the very end, never to tune.

The intraday lock-box (2023-07-01) is the same constant the rest of the project uses
(mft.data_layer.alpaca_ingest.INTRADAY_LOCKBOX). `load_lockbox()` refuses to hand back
lock-box data unless you pass `i_am_done_tuning=True`, so it can't be touched by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mft.data_layer.alpaca_ingest import INTRADAY_LOCKBOX

# Pre-registered intraday split (UTC). Validation ends the day before the lock-box.
TRAIN_START = pd.Timestamp("2020-07-27", tz="UTC")
TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
VAL_END = pd.Timestamp("2023-06-30", tz="UTC")
LOCKBOX_START = INTRADAY_LOCKBOX                      # 2023-07-01
LOCKBOX_END = pd.Timestamp("2024-12-31", tz="UTC")


@dataclass(frozen=True)
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive (whole day)

    def mask(self, index: pd.DatetimeIndex) -> pd.Series:
        end_excl = self.end + pd.Timedelta(days=1)
        return (index >= self.start) & (index < end_excl)

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.mask(df.index)]


TRAIN = Split("train", TRAIN_START, TRAIN_END)
VALIDATION = Split("validation", VAL_START, VAL_END)
LOCKBOX = Split("lockbox", LOCKBOX_START, LOCKBOX_END)

# What research is allowed to load. Lock-box is deliberately excluded.
RESEARCH_END = VAL_END  # research data window is [TRAIN_START, VAL_END]


def research_window() -> tuple[str, str]:
    """ISO (start, end) covering train+validation only — never the lock-box."""
    return TRAIN_START.date().isoformat(), VAL_END.date().isoformat()


def split_train_val(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {'train': ..., 'validation': ...}. Lock-box intentionally absent."""
    return {"train": TRAIN.slice(df), "validation": VALIDATION.slice(df)}


def assert_no_lockbox(index: pd.DatetimeIndex) -> None:
    """Raise if any timestamp is on/after the lock-box — call before research compute."""
    if (index >= LOCKBOX_START).any():
        raise AssertionError(
            f"Lock-box leak: {int((index >= LOCKBOX_START).sum())} bars on/after "
            f"{LOCKBOX_START.date()} present in a research frame."
        )


def load_lockbox(*, i_am_done_tuning: bool = False) -> Split:
    """
    Hand back the LOCKBOX split — only with explicit acknowledgement. This exists so
    the final exam is a deliberate, one-time act, never an accident inside a loop.
    """
    if not i_am_done_tuning:
        raise PermissionError(
            "Lock-box is sealed. Pass i_am_done_tuning=True ONLY for the final, "
            "one-time out-of-sample exam — never to select or tune parameters."
        )
    return LOCKBOX
