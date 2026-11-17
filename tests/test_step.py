import copy
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

import synthetic as syn
from agent.account import Account
from agent.judge_link import JudgeTreeMismatch, load_prereg
from agent.step import LookAheadError, StepError, daily_step

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "tektradoor-judge"
LONG = load_prereg(VENDOR / "prereg" / "01-zarattini-long.json")
LONG_SHORT = load_prereg(VENDOR / "prereg" / "02-zarattini-long-short.json")


def _account(prereg=LONG):
    return Account.new(prereg["candidate_id"], start=syn.WINDOW_START)


def _first_day():
    return syn.WINDOW_START - timedelta(days=1)


def test_first_step_trades_from_flat():
    acct = _account(LONG_SHORT)
    day = _first_day()
    close, qv, funding = syn.cut(syn.judge_load(), day)
    acct, trades = daily_step(LONG_SHORT, close, qv, funding, acct, day)
    assert acct.last_day == day
    assert acct.equity_history == [(day, 10_000.0)]
    assert trades and all(t.day == day and t.old_weight == 0.0 for t in trades)
    assert {t.coin: t.new_weight for t in trades} == acct.weights
    assert sum(abs(w) for w in acct.weights.values()) <= 1.0 + 1e-12  # the judge's gross cap
    cost = sum(t.cost for t in trades)
    assert cost > 0
    assert acct.cash == pytest.approx(10_000.0 - cost, abs=1e-9)
    for t in trades:
        assert t.notional == pytest.approx(abs(t.new_weight - t.old_weight) * 10_000.0)
        assert t.cost == pytest.approx(0.0011 * t.notional)
        assert t.side == ("buy" if t.new_weight > t.old_weight else "sell")


def test_frames_running_past_the_day_are_refused():
    acct = _account()
    day = _first_day()
    close, qv, funding = syn.cut(syn.judge_load(), day + timedelta(days=1))
    with pytest.raises(LookAheadError, match="close"):
        daily_step(LONG, close, qv, funding, acct, day)
    c, q, f = syn.cut(syn.judge_load(), day)
    with pytest.raises(LookAheadError, match="quote_volume"):
        daily_step(LONG, c, qv, f, acct, day)
    with pytest.raises(LookAheadError, match="funding"):
        daily_step(LONG, c, q, funding, acct, day)
    assert acct.last_day is None and acct.trade_log == []  # nothing changed


def test_frames_must_start_where_the_judge_starts_loading():
    acct = _account()
    day = _first_day()
    close, qv, funding = syn.cut(syn.judge_load(), day)
    with pytest.raises(StepError, match="data_start"):
        daily_step(LONG, close.iloc[1:], qv.iloc[1:], funding.iloc[1:], acct, day)


def test_days_must_follow_on():
    acct = _account()
    day = _first_day()
    with pytest.raises(StepError, match="first decision day"):
        daily_step(LONG, *syn.cut(syn.judge_load(), day + timedelta(days=1)), acct, day + timedelta(days=1))
    acct, _ = daily_step(LONG, *syn.cut(syn.judge_load(), day), acct, day)
    skip = day + timedelta(days=2)
    with pytest.raises(StepError, match="next day"):
        daily_step(LONG, *syn.cut(syn.judge_load(), skip), acct, skip)


def test_prereg_must_belong_to_the_account():
    acct = _account(LONG)
    day = _first_day()
    with pytest.raises(StepError, match="candidate"):
        daily_step(LONG_SHORT, *syn.cut(syn.judge_load(), day), acct, day)


def test_prereg_on_other_judge_code_is_refused():
    bad = copy.deepcopy(LONG)
    bad["judge_tree_hash"] = "f" * 40
    day = _first_day()
    with pytest.raises(JudgeTreeMismatch):
        daily_step(bad, *syn.cut(syn.judge_load(), day), _account(), day)


def test_second_step_marks_the_book():
    acct = _account(LONG_SHORT)
    d0 = _first_day()
    acct, _ = daily_step(LONG_SHORT, *syn.cut(syn.judge_load(), d0), acct, d0)
    d1 = d0 + timedelta(days=1)
    acct, _ = daily_step(LONG_SHORT, *syn.cut(syn.judge_load(), d1), acct, d1)
    assert [d for d, _ in acct.equity_history] == [d0, d1]
    assert acct.equity_history[1][1] != 10_000.0
    assert isinstance(acct.equity_history[1][1], float)
    assert not isinstance(acct.equity_history[1][1], pd.Series)
