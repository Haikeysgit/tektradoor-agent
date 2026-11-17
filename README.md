# tektradoor-agent

An autonomous crypto trading agent that runs practice money with the exact
locked rules that [tektradoor-judge](https://github.com/Haikeysgit/tektradoor-judge)
tested. Once a day it decides trades by calling the judge's own code, so what
runs is what was judged.

Work in progress. Step 1 is the core: the practice account, the daily step
and a replay over past data. There is no live data feed, scheduler or
database yet.

## What runs is what was tested

- The judge is a git submodule at `vendor/tektradoor-judge`, pinned to the
  commit the rules were registered on. The agent imports it from there and
  does not reimplement any formula: universe, daily exit check, Donchian
  target weights, the rebalance threshold, costs and funding are the judge's
  functions, called on the price history up to each day's close.
- Before running a prereg, the agent checks that its `judge_tree_hash` equals
  `git -C vendor/tektradoor-judge rev-parse HEAD:judge` and that the judge
  code has no local changes. Otherwise it refuses.
- `tests/test_replay_matches_judge.py` steps the agent day by day over a
  synthetic market and checks that its daily equity returns equal the judge's
  daily net returns (within 1e-10, every day) and that its trade costs equal
  the judge's, for both registered preregs.

## Setup

```
git clone --recurse-submodules <repo url>
cd tektradoor-agent
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install --require-hashes -r requirements.txt
git config core.hooksPath .githooks
python -m pytest -q
```

## Replay

```
python -m agent replay vendor/tektradoor-judge/prereg/01-zarattini-long.json --start 2025-03-26 --end 2025-06-30
```

`start..end` are the days whose returns count, as in the judge's windows. The
book is flat at $10,000 on the close before `start`, the first close it
trades. Data comes from the judge's loader (Binance archives, cached in
`data_cache/`), from 450 days before `start`. It prints the final equity, the
trade count and the max drawdown; `--save account.json` keeps the account.

## License

Apache-2.0
