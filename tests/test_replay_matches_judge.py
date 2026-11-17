"""The proof: stepping the agent day by day reproduces the judge.

On the synthetic panel, the judge's daily net returns are computed with the
same calls ``judge.judge.run()`` makes (universe, members, exclusions,
portfolio_weights, start_flat, net_returns, window_slice). The agent steps
from the first decision day (window start - 1) to the window end, one close
at a time, seeing only data up to each close. Its day-over-day equity returns
must equal the judge's net returns on every day, and its trade log's costs
must equal the judge's cost component, for both registered preregs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import synthetic as syn
from agent.judge_link import J, load_prereg
from agent.replay import replay_frames
from judge.costs import _executed_weights, net_returns
from judge.strategies.donchian import portfolio_weights

PREREG_DIR = Path(__file__).resolve().parent.parent / "vendor" / "tektradoor-judge" / "prereg"
TOL = 1e-10


def judge_view(prereg: dict, close, qv, funding, win):
    """The judge's daily net returns and cost component, exactly as run() computes them."""
    universe = J._monthly_universe(close, qv, J.JUDGING["universe_size"])
    ever = set().union(*universe.values()) if universe else set()
    members = [c for c in close.columns if c in ever]
    close, qv = close[members], qv[members]
    funding = funding.reindex(columns=members)
    excluded = J._daily_exclusions(close, qv, win)
    cost = float(J.JUDGING["cost_per_side"])
    threshold = float(J.JUDGING["rebalance_threshold"])
    weights, changed = portfolio_weights(
        close, universe, prereg["long_short"], float(prereg["per_coin_cap_repro"]),
        float(J.JUDGING["gross_cap"]), excluded, lookbacks=tuple(prereg["lookbacks"]),
        vol_target=float(prereg["vol_target"]), vol_window=int(prereg["vol_window"]),
    )
    flat = J.start_flat(weights, win)
    net = J.window_slice(net_returns(flat, changed, close, funding, cost, threshold), win)

    # Cost component of net (net_returns' own turnover): the cost of the
    # trades decided on day t's close, a fraction of the equity at t, is part
    # of day t+1's return. In USD on the judge's $10,000 equity curve:
    w_exec = _executed_weights(flat, changed, threshold)
    turnover = (w_exec - w_exec.shift(1).fillna(0.0)).abs().sum(axis=1)
    decision_days = net.index - pd.Timedelta(days=1)
    equity_before = 10_000.0 * np.concatenate([[1.0], np.cumprod(1.0 + net.to_numpy())[:-1]])
    cost_usd = float(np.sum(cost * turnover.loc[decision_days].to_numpy() * equity_before))
    return net, cost_usd, universe, excluded


@pytest.fixture(scope="module", params=["01-zarattini-long.json", "02-zarattini-long-short.json"])
def both(request):
    path = PREREG_DIR / request.param
    prereg = load_prereg(path).rules
    win = (syn.WINDOW_START, syn.WINDOW_END)
    judge = judge_view(prereg, *syn.judge_load(), win)
    # The agent gets the whole 900-day panel and must cut it where the judge loads.
    account = replay_frames(path, *syn.panel(), syn.WINDOW_START, syn.WINDOW_END)
    return prereg, judge, account


def _agent_returns(account) -> pd.Series:
    days = pd.DatetimeIndex([pd.Timestamp(m.day) for m in account.equity_history])
    equity = pd.Series([m.equity for m in account.equity_history], index=days)
    return (equity / equity.shift(1) - 1.0).iloc[1:]


def test_daily_returns_match_the_judge(both):
    prereg, (net, _, _, _), account = both
    ours = _agent_returns(account)
    assert list(ours.index) == list(net.index)  # every window day, start..end
    assert len(net) == syn.WINDOW_DAYS
    diff = (ours - net).abs()
    print(f"\n{prereg['candidate_id']}: max |agent - judge| daily return = {diff.max():.3e}")
    assert diff.max() <= TOL, diff.idxmax()
    assert net.abs().max() > 1e-3  # the book really traded


def test_trade_costs_match_the_judge(both):
    _, (_, cost_usd, _, _), account = both
    # Trades decided on the window end's close settle the day after the window.
    ours = sum(t.cost for t in account.trade_log if t.day < syn.WINDOW_END)
    assert cost_usd > 0
    assert abs(ours - cost_usd) <= TOL * 10_000.0, (ours, cost_usd)


def test_daily_components_add_up_and_costs_match_the_judge(both):
    _, (_, cost_usd, _, _), account = both
    marks = account.equity_history
    for prev, m in zip(marks, marks[1:]):
        assert abs((m.equity - prev.equity) - (m.price_pnl + m.funding_pnl - m.costs)) <= 1e-9, m.day
    assert sum(m.funding_pnl for m in marks) != 0.0
    # The marks' costs are the judge's cost component, day by day.
    assert abs(sum(m.costs for m in marks) - cost_usd) <= TOL * 10_000.0


def test_final_equity_is_the_judges_total_return(both):
    _, (net, _, _, _), account = both
    assert account.equity == pytest.approx(10_000.0 * (1.0 + J.total_return(net)), rel=1e-12)


def test_the_window_exercises_the_rules(both):
    prereg, (_, _, universe, excluded), account = both
    window_months = [d for d in universe if syn.WINDOW_START.replace(day=1) <= d.replace(day=1) <= syn.WINDOW_END]
    assert len({tuple(universe[d]) for d in sorted(universe)[-4:]}) > 1  # the universe changes
    assert len(window_months) >= 2
    flagged = set().union(*excluded.values())
    assert {syn.ILLIQUID, syn.STALE} <= flagged
    assert any(t.coin == syn.DELISTED and t.new_weight == 0.0 for t in account.trade_log)
    assert any(t.coin == syn.ILLIQUID for t in account.trade_log)
    if prereg["long_short"]:
        assert any(t.new_weight < 0 for t in account.trade_log)
