import json
from datetime import date, timedelta

import pytest

from agent.account import Account, Trade


def _account() -> Account:
    acct = Account.new("01-zarattini-long", start=date(2025, 3, 26))
    acct.cash = 9_987.123456789012
    acct.weights = {"AAAUSDT": 0.1234567890123457, "BBBUSDT": -0.05}
    acct.prev_weights = {"AAAUSDT": 0.1}
    acct.last_day = date(2025, 3, 27)
    acct.equity_history = [(date(2025, 3, 25), 10_000.0), (date(2025, 3, 26), 10_001.000000000002)]
    acct.trade_log = [
        Trade(date(2025, 3, 26), "AAAUSDT", "buy", 0.1, 0.1234567890123457, 234.5678, 0.2580),
        Trade(date(2025, 3, 26), "BBBUSDT", "sell", 0.0, -0.05, 500.0, 0.55),
    ]
    return acct


def test_new_account_defaults():
    acct = Account.new("01-zarattini-long", start=date(2025, 3, 26))
    assert acct.starting_equity == 10_000.0
    assert acct.cash == 10_000.0
    assert acct.weights == {} and acct.trade_log == [] and acct.equity_history == []
    assert acct.first_decision_day == date(2025, 3, 25)
    assert acct.data_start == date(2025, 3, 26) - timedelta(days=450)  # the judge's LOOKBACK_DAYS
    assert acct.last_day is None


def test_json_round_trip_is_exact():
    acct = _account()
    text = acct.to_json()
    json.loads(text)  # plain JSON
    back = Account.from_json(text)
    assert back == acct
    assert back.to_json() == text


def test_from_json_rejects_unknown_version():
    raw = json.loads(_account().to_json())
    raw["version"] = 99
    with pytest.raises(ValueError, match="version"):
        Account.from_json(json.dumps(raw))


def test_equity_is_the_last_mark():
    acct = _account()
    assert acct.equity == 10_001.000000000002
    assert Account.new("x", start=date(2025, 1, 2)).equity == 10_000.0
