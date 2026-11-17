"""One daily step: mark the book at ``day``'s close, then trade to the weights
the judge's locked rules decide on that close.

Every formula is the judge's own code, called on the history up to ``day``:

* universe: ``judge.judge._monthly_universe`` (with the JUDGING universe size),
  then only coins ever in it are kept, as ``run()`` does;
* daily exit check: ``judge.judge._daily_exclusions`` for ``day`` and the day
  before (the only days that reach ``day``'s weights and change flags);
* target weights and change flags: ``judge.strategies.donchian.portfolio_weights``,
  last row;
* executed weights: ``judge.costs._executed_weights`` (the rebalance threshold)
  on two rows, the book's current executed weights then today's targets. The
  first row has every flag set, so it loads the current weights exactly; the
  second row is the judge's own loop step;
* the mark: ``judge.costs.net_returns`` on the book's last two executed weight
  rows, which gives the judge's net return for ``day`` (price change, the cost
  of the last close's trades, funding).

All the judge's functions are causal (each row reads only earlier rows), so
their row for ``day`` on history cut at ``day`` equals the row ``run()`` gets
on the full window. tests/test_replay_matches_judge.py checks this end to end.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from agent.account import Account, Trade
from agent.judge_link import J, check_judge_tree
from judge.costs import _executed_weights, net_returns
from judge.strategies.donchian import portfolio_weights


class StepError(ValueError):
    """The step cannot run on these inputs."""


class LookAheadError(StepError):
    """A frame holds data after the decision day's close."""


def _rules(prereg: dict) -> dict:
    """The prereg's rules, validated by the judge, with the judge's fixed JUDGING values."""
    missing = [k for k in J.REQUIRED_KEYS if k not in prereg]
    if missing:
        raise J.PreregError(f"prereg missing required key(s): {', '.join(missing)}")
    J._validate(prereg, prereg["candidate_id"])
    return {
        "long_short": prereg["long_short"],
        "lookbacks": tuple(prereg["lookbacks"]),
        "vol_target": float(prereg["vol_target"]),
        "vol_window": int(prereg["vol_window"]),
        "per_coin_cap": float(prereg["per_coin_cap_repro"]),
        "gross_cap": float(J.JUDGING["gross_cap"]),
        "cost": float(J.JUDGING["cost_per_side"]),
        "threshold": float(J.JUDGING["rebalance_threshold"]),
        "universe_size": J.JUDGING["universe_size"],
    }


def _check_frames(close, quote_volume, funding, account: Account, day: date) -> None:
    ts = pd.Timestamp(day)
    first = pd.Timestamp(account.data_start)
    for name, frame in (("close", close), ("quote_volume", quote_volume), ("funding", funding)):
        if frame is None and name == "funding":
            continue
        idx = frame.index
        if len(idx) == 0 or not idx.is_monotonic_increasing or idx[-1] != ts:
            last = idx[-1].date() if len(idx) else None
            raise LookAheadError(f"{name} must end at {day}'s close (no look-ahead); it ends at {last}")
        if idx[0] != first:
            raise StepError(
                f"{name} must start at the account's data_start {account.data_start} "
                f"(where the judge starts loading); it starts at {idx[0].date()}"
            )


def _check_day(account: Account, day: date) -> None:
    if account.last_day is None:
        if day != account.first_decision_day:
            raise StepError(f"the first step must be the first decision day {account.first_decision_day}, got {day}")
    elif day != account.last_day + timedelta(days=1):
        raise StepError(f"the next step must be the next day {account.last_day + timedelta(days=1)}, got {day}")


def _mark(account: Account, close, funding, day: date, cost: float) -> float:
    """The judge's net return for ``day`` on the book's executed weights."""
    coins = [c for c in close.columns if c in account.weights or c in account.prev_weights]
    if not coins:
        return 0.0
    t = pd.Timestamp(account.last_day)
    idx = pd.DatetimeIndex([t - pd.Timedelta(days=1), t, pd.Timestamp(day)])
    prev = pd.Series(account.prev_weights, dtype=float).reindex(coins, fill_value=0.0)
    cur = pd.Series(account.weights, dtype=float).reindex(coins, fill_value=0.0)
    weights = pd.DataFrame([prev.to_numpy(), cur.to_numpy(), cur.to_numpy()], index=idx, columns=coins)
    flags = pd.DataFrame(True, index=idx, columns=coins)
    net = net_returns(weights, flags, close, funding, cost, rebalance_threshold=None)
    return float(net.loc[pd.Timestamp(day)])


def _decide(rules: dict, close, quote_volume, account: Account, day: date) -> pd.Series:
    """Executed weights for ``day``'s close (index: coins ever in the universe so far)."""
    universe = J._monthly_universe(close, quote_volume, rules["universe_size"])
    ever = set().union(*universe.values()) if universe else set()
    members = [c for c in close.columns if c in ever]  # other coins never get weight
    outside = sorted(set(account.weights) - set(members))
    if outside:
        raise StepError(f"held coins outside every universe so far: {outside}")
    if not members:
        return pd.Series(dtype=float)
    close, quote_volume = close[members], quote_volume[members]
    excluded = J._daily_exclusions(close, quote_volume, (max(day, account.start), day))
    weights, changed = portfolio_weights(
        close, universe, rules["long_short"], rules["per_coin_cap"], rules["gross_cap"], excluded,
        lookbacks=rules["lookbacks"], vol_target=rules["vol_target"], vol_window=rules["vol_window"],
    )
    ts = pd.Timestamp(day)
    rows = pd.DatetimeIndex([ts - pd.Timedelta(days=1), ts])
    current = pd.Series(account.weights, dtype=float).reindex(members, fill_value=0.0)
    two = pd.DataFrame([current.to_numpy(), weights.loc[ts].to_numpy()], index=rows, columns=members)
    flags = pd.DataFrame([[True] * len(members), changed.loc[ts].to_numpy(dtype=bool)], index=rows, columns=members)
    return _executed_weights(two, flags, rules["threshold"]).loc[ts]


def daily_step(prereg: dict, close, quote_volume, funding, account: Account, day: date) -> tuple[Account, list[Trade]]:
    """Mark the book at ``day``'s close and trade to that close's decision.

    close, quote_volume, funding: the judge's panel frames (dates x symbols),
    from ``account.data_start`` up to and including ``day``, nothing later.
    Returns a new Account and the day's trades; ``account`` is not changed.
    """
    if prereg.get("candidate_id") != account.candidate_id:
        raise StepError(f"prereg candidate {prereg.get('candidate_id')!r} is not this account's {account.candidate_id!r}")
    rules = _rules(prereg)
    check_judge_tree(prereg)
    _check_frames(close, quote_volume, funding, account, day)
    _check_day(account, day)
    missing = sorted(set(account.weights) - set(close.columns))
    if missing:
        raise StepError(f"held coins missing from the price data: {missing}")

    if account.last_day is None:
        mark = account.starting_equity
    else:
        mark = account.equity * (1.0 + _mark(account, close, funding, day, rules["cost"]))

    executed = _decide(rules, close, quote_volume, account, day)
    old = account.weights
    trades = []
    for coin in executed.index:
        before, after = float(old.get(coin, 0.0)), float(executed[coin])
        if after != before:
            notional = abs(after - before) * mark
            trades.append(Trade(day, coin, "buy" if after > before else "sell", before, after,
                                notional, rules["cost"] * notional))
    new = replace(
        account,
        cash=mark - sum(t.cost for t in trades),
        weights={c: float(w) for c, w in executed.items() if w != 0.0},
        prev_weights=dict(old),
        last_day=day,
        equity_history=account.equity_history + [(day, mark)],
        trade_log=account.trade_log + trades,
    )
    return new, trades
