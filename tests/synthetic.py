"""A deterministic synthetic market for the agent tests (no network).

25 coins of trending random walks with funding over 900 days, laid out like
the judge's data panel: dates x symbols, NaN before a listing and after a
delisting. The judge loads from DATA_START, LOOKBACK_DAYS before the window
start; the panel starts earlier so replays must cut it there. The window
includes the events the rules react to: universe changes at month ends, a
member delisting, a member flagged illiquid and a member flagged stale (both
dropped by the judge's daily exit check), and late listings.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from agent.judge_link import J

N_COINS = 25
PANEL_DAYS = 900
WINDOW_START = date(2025, 3, 26)
WINDOW_DAYS = 45
WINDOW_END = WINDOW_START + timedelta(days=WINDOW_DAYS - 1)  # 2025-05-09
DATA_START = WINDOW_START - timedelta(days=J.LOOKBACK_DAYS)  # where the judge starts loading
PANEL_START = WINDOW_END - timedelta(days=PANEL_DAYS - 1)

DELISTED = "SYN05USDT"  # last close WINDOW_START + 24
ILLIQUID = "SYN03USDT"  # volume collapses WINDOW_START + 3 .. + 38
STALE = "SYN08USDT"  # price barely moves WINDOW_START + 8 .. + 43
LATE = ("SYN22USDT", "SYN23USDT")  # listed 40 and 70 days after DATA_START


@lru_cache(maxsize=None)
def _panel():
    rng = np.random.default_rng(20251117)
    idx = pd.date_range(pd.Timestamp(PANEL_START), pd.Timestamp(WINDOW_END), freq="D", name="date")
    n, k = len(idx), N_COINS
    cols = [f"SYN{i:02d}USDT" for i in range(k)]
    w0 = (pd.Timestamp(WINDOW_START) - idx[0]).days
    d0 = (pd.Timestamp(DATA_START) - idx[0]).days

    # Trending regimes: a drift that changes every 30-120 days, plus noise.
    drift = np.zeros((n, k))
    for j in range(k):
        t = 0
        while t < n:
            span = int(rng.integers(30, 121))
            drift[t:t + span, j] = rng.normal(0.0, 0.008)
            t += span
    vol = rng.uniform(0.02, 0.06, k)
    rets = drift + rng.normal(0.0, 1.0, (n, k)) * vol
    s = cols.index(STALE)
    rets[w0 + 8:w0 + 44, s] = rng.normal(0.0, 0.002, 36)
    close = 50.0 * rng.uniform(0.2, 5.0, k) * np.exp(np.cumsum(rets, axis=0))

    base = np.exp(rng.uniform(np.log(4e6), np.log(8e7), k))
    for coin in (ILLIQUID, DELISTED, STALE):
        base[cols.index(coin)] = 2e8  # keep them in the top 20
    qv = base * np.exp(rng.normal(0.0, 0.4, (n, k)))
    qv[w0 + 3:w0 + 39, cols.index(ILLIQUID)] *= 0.002

    funding = rng.normal(1e-4, 3e-4, (n, k))

    close = pd.DataFrame(close, index=idx, columns=cols)
    qv = pd.DataFrame(qv, index=idx, columns=cols)
    funding = pd.DataFrame(funding, index=idx, columns=cols)
    close.loc[idx[w0 + 25]:, DELISTED] = np.nan
    close.loc[:idx[d0 + 39], LATE[0]] = np.nan
    close.loc[:idx[d0 + 69], LATE[1]] = np.nan
    qv = qv.where(close.notna())
    funding = funding.where(close.notna())
    return close, qv, funding


def panel():
    """(close, quote_volume, funding) copies, PANEL_START .. WINDOW_END."""
    close, qv, funding = _panel()
    return close.copy(), qv.copy(), funding.copy()


def cut(frames, day: date):
    """The frames up to and including ``day``'s close."""
    ts = pd.Timestamp(day)
    return tuple(f.loc[:ts] for f in frames)


def judge_load():
    """The panel as the judge's run() loads it: DATA_START .. WINDOW_END."""
    first = pd.Timestamp(DATA_START)
    return tuple(f.loc[first:] for f in panel())
