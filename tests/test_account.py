import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from agent.account import Account, Mark, Trade
from agent.judge_link import load_prereg

PREREG = Path(__file__).resolve().parent.parent / "vendor" / "tektradoor-judge" / "prereg" / "01-zarattini-long.json"
SHA = "a" * 64
TREE = "c0398cd440cea365009cedee9fc97c609eea2106"


def _account() -> Account:
    acct = Account.new("01-zarattini-long", date(2025, 3, 26), SHA, TREE)
    acct.equity_after_costs = 9_987.123456789012
    acct.weights = {"AAAUSDT": 0.1234567890123457, "BBBUSDT": -0.05}
    acct.prev_weights = {"AAAUSDT": 0.1}
    acct.last_day = date(2025, 3, 27)
    acct.equity_history = [
        Mark(date(2025, 3, 25), 10_000.0, 0.0, 0.0, 0.0),
        Mark(date(2025, 3, 26), 10_001.000000000002, 4.25, -0.125, 3.124999999999998),
    ]
    acct.trade_log = [
        Trade(date(2025, 3, 26), "AAAUSDT", "buy", 0.1, 0.1234567890123457, 234.5678, 0.2580),
        Trade(date(2025, 3, 26), "BBBUSDT", "sell", 0.0, -0.05, 500.0, 0.55),
    ]
    return acct


def test_new_account_defaults():
    acct = Account.new("01-zarattini-long", date(2025, 3, 26), SHA, TREE)
    assert acct.starting_equity == 10_000.0
    assert acct.equity_after_costs == 10_000.0
    assert acct.prereg_sha256 == SHA and acct.judge_tree_hash == TREE
    assert acct.weights == {} and acct.trade_log == [] and acct.equity_history == []
    assert acct.first_decision_day == date(2025, 3, 25)
    assert acct.data_start == date(2025, 3, 26) - timedelta(days=450)  # the judge's LOOKBACK_DAYS
    assert acct.last_day is None


def test_open_pins_the_registered_prereg():
    reg = load_prereg(PREREG)
    acct = Account.open(reg, date(2025, 3, 26))
    assert acct.candidate_id == "01-zarattini-long"
    assert acct.prereg_sha256 == reg.sha256
    assert acct.judge_tree_hash == reg.judge_tree_hash == TREE


def test_json_round_trip_is_exact():
    acct = _account()
    text = acct.to_json()
    raw = json.loads(text)  # plain JSON
    assert "cash" not in raw
    assert raw["equity_history"][1] == {
        "day": "2025-03-26", "equity": 10_001.000000000002, "price_pnl": 4.25,
        "funding_pnl": -0.125, "costs": 3.124999999999998,
    }
    back = Account.from_json(text)
    assert back == acct
    assert back.to_json() == text


def test_from_json_rejects_unknown_version():
    raw = json.loads(_account().to_json())
    raw["version"] = 1
    with pytest.raises(ValueError, match="version"):
        Account.from_json(json.dumps(raw))


def test_equity_is_the_last_mark():
    acct = _account()
    assert acct.equity == 10_001.000000000002
    assert Account.new("x", date(2025, 1, 2), SHA, TREE).equity == 10_000.0
