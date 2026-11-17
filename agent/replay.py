"""Replay: step the agent day by day over a past window.

The window is the judge's: ``start..end`` are the days whose returns count.
The book is flat at $10,000 on the close of start - 1 day, the first decision
day. The agent then steps every close from start - 1 to end, each time seeing
only data up to that close. Trades decided on end's close are in the trade
log; their cost settles the day after the window.

Data comes from the judge's own loader (``judge.judge._load_data``), from
start - LOOKBACK_DAYS to end, as ``run()`` loads it.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agent.account import Account
from agent.judge_link import J, load_prereg
from agent.step import daily_step
from judge.metrics import max_drawdown

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


def replay_frames(prereg: dict, close, quote_volume, funding, start: date, end: date) -> Account:
    """Step a fresh account over start..end on these frames (cut to each day as it goes)."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    account = Account.new(prereg["candidate_id"], start=start)
    first = pd.Timestamp(account.data_start)
    frames = [f.loc[first:] if f is not None else None for f in (close, quote_volume, funding)]
    day = account.first_decision_day
    while day <= end:
        ts = pd.Timestamp(day)
        cut = [f.loc[:ts] if f is not None else None for f in frames]
        account, _ = daily_step(prereg, *cut, account, day)
        day += timedelta(days=1)
    return account


def daily_returns(account: Account) -> pd.Series:
    """Day-over-day returns of the equity marks (the judge's daily net returns)."""
    days = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in account.equity_history])
    equity = pd.Series([v for _, v in account.equity_history], index=days, dtype=float)
    return (equity / equity.shift(1) - 1.0).iloc[1:]


def summary(account: Account) -> dict:
    return {
        "candidate_id": account.candidate_id,
        "start": account.start.isoformat(),
        "end": account.last_day.isoformat() if account.last_day else None,
        "final_equity": account.equity,
        "cash": account.cash,
        "trades": len(account.trade_log),
        "max_drawdown": max_drawdown(daily_returns(account)),
    }


def replay(prereg_path: Path, start: date, end: date, cache_dir: Path | None = None, loader=None) -> Account:
    """Load the judge's data for the window and replay it. ``loader`` defaults to the judge's."""
    prereg = load_prereg(Path(prereg_path))  # judge validation + judge_tree_hash startup check
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    load_start = start - timedelta(days=J.LOOKBACK_DAYS)
    loader = loader or J._load_data
    close, qv, funding = loader(J.MARKET, load_start, end, cache_dir)
    last = J._data_last_date(close)
    if last < end:
        raise J.DataNotReady(f"data ends {last}, before the window end {end}")
    return replay_frames(prereg, close, qv, funding, start, end)
