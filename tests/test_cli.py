import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

import synthetic as syn
from agent import replay as replay_mod
from agent.__main__ import main
from agent.account import Account

PREREG = Path(__file__).resolve().parent.parent / "vendor" / "tektradoor-judge" / "prereg" / "01-zarattini-long.json"
END = syn.WINDOW_START + timedelta(days=3)


def fake_loader(calls):
    def load(market, start, end, cache_dir):
        calls.append((market, start, end, Path(cache_dir)))
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        return tuple(f.loc[lo:hi] for f in syn.panel())
    return load


def test_replay_cli_prints_the_summary(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(replay_mod, "load_data", fake_loader(calls))
    save = tmp_path / "account.json"
    code = main(["replay", str(PREREG), "--start", syn.WINDOW_START.isoformat(), "--end", END.isoformat(),
                 "--cache-dir", str(tmp_path / "cache"), "--save", str(save)])
    out = capsys.readouterr().out
    assert code == 0
    # The judge's load: market um, LOOKBACK_DAYS of warmup, through the window end.
    assert calls == [("um", syn.DATA_START, END, tmp_path / "cache")]
    acct = Account.from_json(save.read_text(encoding="utf-8"))
    assert acct.last_day == END and len(acct.equity_history) == 5
    assert f"final equity: {acct.equity:,.2f}" in out
    assert f"trades: {len(acct.trade_log)}" in out
    assert "max drawdown: " in out


def test_replay_cli_refuses_data_that_ends_early(monkeypatch, capsys):
    def short(market, start, end, cache_dir):
        return tuple(f.loc[pd.Timestamp(start):pd.Timestamp(end) - pd.Timedelta(days=1)] for f in syn.panel())

    monkeypatch.setattr(replay_mod, "load_data", short)
    code = main(["replay", str(PREREG), "--start", syn.WINDOW_START.isoformat(), "--end", END.isoformat()])
    assert code == 3
    assert "before the window end" in capsys.readouterr().err


def test_replay_cli_refuses_a_bad_prereg(tmp_path, capsys):
    raw = json.loads(PREREG.read_text(encoding="utf-8"))
    raw["judge_tree_hash"] = "a" * 40
    bad = tmp_path / "01-other.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    code = main(["replay", str(bad), "--start", "2025-03-26", "--end", "2025-03-30"])
    assert code == 2
    assert "judge_tree_hash" in capsys.readouterr().err
