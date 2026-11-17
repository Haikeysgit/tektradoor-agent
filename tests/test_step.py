import subprocess
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

import synthetic as syn
from agent import judge_link
from agent.account import Account
from agent.judge_link import JudgeTreeMismatch, PreregNotRegistered, load_prereg
from agent.step import LookAheadError, StepError, daily_step

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "tektradoor-judge"
LONG = VENDOR / "prereg" / "01-zarattini-long.json"
LONG_SHORT = VENDOR / "prereg" / "02-zarattini-long-short.json"


def _account(path=LONG):
    return Account.open(load_prereg(path), syn.WINDOW_START)


def _first_day():
    return syn.WINDOW_START - timedelta(days=1)


def _frames(day):
    return syn.cut(syn.judge_load(), day)


def test_first_step_trades_from_flat():
    acct = _account(LONG_SHORT)
    day = _first_day()
    acct, trades = daily_step(LONG_SHORT, *_frames(day), acct, day)
    assert acct.last_day == day
    assert [(m.day, m.equity, m.price_pnl, m.funding_pnl, m.costs) for m in acct.equity_history] == [
        (day, 10_000.0, 0.0, 0.0, 0.0)
    ]
    assert trades and all(t.day == day and t.old_weight == 0.0 for t in trades)
    assert {t.coin: t.new_weight for t in trades} == acct.weights
    assert sum(abs(w) for w in acct.weights.values()) <= 1.0 + 1e-12  # the judge's gross cap
    cost = sum(t.cost for t in trades)
    assert cost > 0
    assert acct.equity_after_costs == pytest.approx(10_000.0 - cost, abs=1e-9)
    for t in trades:
        assert t.notional == pytest.approx(abs(t.new_weight - t.old_weight) * 10_000.0)
        assert t.cost == pytest.approx(0.0011 * t.notional)
        assert t.side == ("buy" if t.new_weight > t.old_weight else "sell")


def test_second_step_marks_the_book_with_its_components():
    acct = _account(LONG_SHORT)
    d0 = _first_day()
    acct, trades0 = daily_step(LONG_SHORT, *_frames(d0), acct, d0)
    d1 = d0 + timedelta(days=1)
    acct, _ = daily_step(LONG_SHORT, *_frames(d1), acct, d1)
    m0, m1 = acct.equity_history
    assert m1.day == d1 and isinstance(m1.equity, float) and m1.equity != 10_000.0
    assert m1.costs == pytest.approx(sum(t.cost for t in trades0), abs=1e-9)  # yesterday's trades
    assert m1.price_pnl != 0.0 and m1.funding_pnl != 0.0
    assert m1.equity - m0.equity == pytest.approx(m1.price_pnl + m1.funding_pnl - m1.costs, abs=1e-9)


def test_frames_running_past_the_day_are_refused():
    acct = _account()
    day = _first_day()
    close, qv, funding = _frames(day + timedelta(days=1))
    with pytest.raises(LookAheadError, match="close"):
        daily_step(LONG, close, qv, funding, acct, day)
    c, q, f = _frames(day)
    with pytest.raises(LookAheadError, match="quote_volume"):
        daily_step(LONG, c, qv, f, acct, day)
    with pytest.raises(LookAheadError, match="funding"):
        daily_step(LONG, c, q, funding, acct, day)
    assert acct.last_day is None and acct.trade_log == []  # nothing changed


def test_frames_must_start_where_the_judge_starts_loading():
    acct = _account()
    day = _first_day()
    close, qv, funding = _frames(day)
    with pytest.raises(StepError, match="data_start"):
        daily_step(LONG, close.iloc[1:], qv.iloc[1:], funding.iloc[1:], acct, day)


def test_a_missing_middle_day_is_refused():
    acct = _account()
    day = _first_day()
    frames = _frames(day)
    gap = [f.drop(f.index[200]) for f in frames]
    with pytest.raises(StepError, match="every day"):
        daily_step(LONG, *gap, acct, day)
    # Missing from one frame only: the indexes differ.
    close, qv, funding = frames
    with pytest.raises(StepError, match="quote_volume"):
        daily_step(LONG, close, qv.drop(qv.index[200]), funding, acct, day)
    dup = qv.iloc[list(range(len(qv) - 1)) + [len(qv) - 2, len(qv) - 1]]
    with pytest.raises(StepError, match="quote_volume"):
        daily_step(LONG, close, dup, funding, acct, day)


def test_held_coins_must_be_in_close_and_funding():
    acct = _account()
    d0 = _first_day()
    acct, _ = daily_step(LONG, *_frames(d0), acct, d0)
    d1 = d0 + timedelta(days=1)
    close, qv, funding = _frames(d1)
    held = sorted(acct.weights)[0]
    with pytest.raises(StepError, match="funding"):
        daily_step(LONG, close, qv, funding.drop(columns=[held]), acct, d1)
    sold = replace(acct, weights={}, prev_weights=dict(acct.weights))  # sold on the last close
    with pytest.raises(StepError, match="close"):
        daily_step(LONG, close.drop(columns=[held]), qv, funding, sold, d1)


def test_days_must_follow_on():
    acct = _account()
    day = _first_day()
    with pytest.raises(StepError, match="first decision day"):
        daily_step(LONG, *_frames(day + timedelta(days=1)), acct, day + timedelta(days=1))
    acct, _ = daily_step(LONG, *_frames(day), acct, day)
    skip = day + timedelta(days=2)
    with pytest.raises(StepError, match="next day"):
        daily_step(LONG, *_frames(skip), acct, skip)


def test_another_prereg_is_refused():
    acct = _account(LONG)
    day = _first_day()
    with pytest.raises(PreregNotRegistered, match="sha256"):
        daily_step(LONG_SHORT, *_frames(day), acct, day)


def test_an_account_on_other_judge_code_is_refused():
    acct = replace(_account(), judge_tree_hash="f" * 40)
    day = _first_day()
    with pytest.raises(JudgeTreeMismatch):
        daily_step(LONG, *_frames(day), acct, day)


def test_an_edited_copy_elsewhere_is_refused(tmp_path):
    copy = tmp_path / LONG.name
    copy.write_bytes(LONG.read_bytes().replace(b'"vol_target": 0.25', b'"vol_target": 0.9'))
    day = _first_day()
    with pytest.raises(PreregNotRegistered):
        daily_step(copy, *_frames(day), _account(), day)


def test_edits_to_vendored_files_are_refused_mid_run(tmp_path, monkeypatch):
    clone = tmp_path / "judge-clone"
    git = lambda *a: subprocess.run(["git", "-C", str(a[0]), *a[1:]], check=True, capture_output=True)  # noqa: E731
    git(tmp_path, "clone", "-q", "--no-checkout", str(VENDOR), str(clone))
    git(clone, "config", "core.autocrlf", "false")
    git(clone, "checkout", "-q", "143012e")
    monkeypatch.setattr(judge_link, "VENDOR", clone)
    path = clone / "prereg" / LONG.name
    acct = _account(path)
    d0 = _first_day()
    acct, _ = daily_step(path, *_frames(d0), acct, d0)  # clean: runs
    d1 = d0 + timedelta(days=1)

    costs = clone / "judge" / "costs.py"
    original = costs.read_bytes()
    costs.write_bytes(original + b"\n# edited\n")
    with pytest.raises(JudgeTreeMismatch, match="uncommitted"):
        daily_step(path, *_frames(d1), acct, d1)
    costs.write_bytes(original)

    path.write_bytes(path.read_bytes().replace(b'"vol_target": 0.25', b'"vol_target": 0.9'))
    with pytest.raises(PreregNotRegistered, match="sha256"):
        daily_step(path, *_frames(d1), acct, d1)
