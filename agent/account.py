"""The practice account: one registered prereg's book of practice money.

Positions are USDT-margined perpetual futures held as weights of equity
(weight 0.1 = a long worth 10% of equity, -0.1 a short). Timing follows the
judge's cost model (judge/costs.py) and window rules (judge/judge.py):

* ``start`` is the first day whose return counts, like the judge's window
  start. The book is flat at ``starting_equity`` on the close of
  ``start - 1 day`` (``first_decision_day``), the first close the agent trades.
* Each daily step marks the book at that day's close, then trades to the
  weights decided on that close. ``equity_history`` holds one ``Mark`` per
  decision day; day-over-day changes of ``equity`` are the judge's daily net
  returns, split into price P&L, funding P&L and costs.
* The cost of the trades made on day t's close is charged in day t+1's mark,
  as in the judge (cost(t+1) = cost_per_side * turnover(t)). Positions are
  sized on the mark before that cost. ``equity_after_costs`` is the latest mark
  minus those pending costs: a derived figure for display only; nothing
  computes from it.
* The agent reads price history from ``data_start`` (start minus the judge's
  LOOKBACK_DAYS), the same first day the judge loads. Signals and the universe
  depend on where history starts, so it is fixed per account.
* The account is pinned to one registered prereg (``prereg_sha256``) and one
  judge code tree (``judge_tree_hash``); every step re-verifies both.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from agent.judge_link import J, Registered

STARTING_EQUITY = 10_000.0
JSON_VERSION = 2


@dataclass
class Trade:
    """One change in a coin's executed weight, decided on ``day``'s close."""

    day: date
    coin: str
    side: str  # "buy" (weight up) or "sell" (weight down)
    old_weight: float
    new_weight: float
    notional: float  # |new - old| * the day's mark, in USD
    cost: float  # cost_per_side * notional, in USD; charged in the next day's mark

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        return cls(**{**d, "day": date.fromisoformat(d["day"])})


@dataclass
class Mark:
    """The book at ``day``'s close, before that close's trades.

    equity - previous equity = price_pnl + funding_pnl - costs (up to float
    rounding), all in USD. costs are the previous close's trade costs.
    funding_pnl is negative when the book pays funding.
    """

    day: date
    equity: float
    price_pnl: float
    funding_pnl: float
    costs: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mark":
        return cls(date.fromisoformat(d["day"]), float(d["equity"]), float(d["price_pnl"]),
                   float(d["funding_pnl"]), float(d["costs"]))


@dataclass
class Account:
    candidate_id: str
    start: date
    data_start: date
    prereg_sha256: str
    judge_tree_hash: str
    starting_equity: float = STARTING_EQUITY
    equity_after_costs: float = STARTING_EQUITY  # derived, display only (see module doc)
    weights: dict[str, float] = field(default_factory=dict)  # executed, decided on last_day's close
    prev_weights: dict[str, float] = field(default_factory=dict)  # executed, before last_day's trades
    last_day: date | None = None  # last decision day stepped
    equity_history: list[Mark] = field(default_factory=list)
    trade_log: list[Trade] = field(default_factory=list)

    @classmethod
    def new(cls, candidate_id: str, start: date, prereg_sha256: str, judge_tree_hash: str,
            starting_equity: float = STARTING_EQUITY) -> "Account":
        return cls(
            candidate_id=candidate_id,
            start=start,
            data_start=start - timedelta(days=J.LOOKBACK_DAYS),
            prereg_sha256=prereg_sha256,
            judge_tree_hash=judge_tree_hash,
            starting_equity=float(starting_equity),
            equity_after_costs=float(starting_equity),
        )

    @classmethod
    def open(cls, reg: Registered, start: date, starting_equity: float = STARTING_EQUITY) -> "Account":
        """A new account pinned to a registered prereg (from judge_link.load_prereg)."""
        return cls.new(reg.rules["candidate_id"], start, reg.sha256, reg.judge_tree_hash, starting_equity)

    @property
    def first_decision_day(self) -> date:
        return self.start - timedelta(days=1)

    @property
    def equity(self) -> float:
        """The latest mark (starting equity before the first step)."""
        return self.equity_history[-1].equity if self.equity_history else self.starting_equity

    # ------------------------------------------------------------ JSON

    def to_dict(self) -> dict:
        return {
            "version": JSON_VERSION,
            "candidate_id": self.candidate_id,
            "start": self.start.isoformat(),
            "data_start": self.data_start.isoformat(),
            "prereg_sha256": self.prereg_sha256,
            "judge_tree_hash": self.judge_tree_hash,
            "starting_equity": self.starting_equity,
            "equity_after_costs": self.equity_after_costs,
            "weights": dict(self.weights),
            "prev_weights": dict(self.prev_weights),
            "last_day": self.last_day.isoformat() if self.last_day else None,
            "equity_history": [m.to_dict() for m in self.equity_history],
            "trade_log": [t.to_dict() for t in self.trade_log],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        if d.get("version") != JSON_VERSION:
            raise ValueError(f"unsupported account version {d.get('version')!r}, expected {JSON_VERSION}")
        return cls(
            candidate_id=d["candidate_id"],
            start=date.fromisoformat(d["start"]),
            data_start=date.fromisoformat(d["data_start"]),
            prereg_sha256=d["prereg_sha256"],
            judge_tree_hash=d["judge_tree_hash"],
            starting_equity=float(d["starting_equity"]),
            equity_after_costs=float(d["equity_after_costs"]),
            weights={k: float(v) for k, v in d["weights"].items()},
            prev_weights={k: float(v) for k, v in d["prev_weights"].items()},
            last_day=date.fromisoformat(d["last_day"]) if d["last_day"] else None,
            equity_history=[Mark.from_dict(m) for m in d["equity_history"]],
            trade_log=[Trade.from_dict(t) for t in d["trade_log"]],
        )

    def to_json(self) -> str:
        # json writes floats with repr, which round-trips exactly.
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Account":
        return cls.from_dict(json.loads(text))
