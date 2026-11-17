"""The practice account: one prereg's book of practice money.

Positions are USDT-margined perpetual futures held as weights of equity
(weight 0.1 = a long worth 10% of equity, -0.1 a short). Timing follows the
judge's cost model (judge/costs.py) and window rules (judge/judge.py):

* ``start`` is the first day whose return counts, like the judge's window
  start. The book is flat at ``starting_equity`` on the close of
  ``start - 1 day`` (``first_decision_day``), the first close the agent trades.
* Each daily step marks the book at that day's close, then trades to the
  weights decided on that close. ``equity_history`` holds the marks, one per
  decision day, so its day-over-day changes are the judge's daily net returns.
* A trade's cost is paid from cash when it is made. Positions are sized on the
  mark before that cost, as in the judge, where the cost of the trades decided
  on day t's close is part of day t+1's return. So right after a step
  ``cash = mark - that day's trade costs``, and the next mark includes them.
* The agent reads price history from ``data_start`` (start minus the judge's
  LOOKBACK_DAYS), the same first day the judge loads. Signals and the universe
  depend on where history starts, so it is fixed per account.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from agent.judge_link import J

STARTING_EQUITY = 10_000.0
JSON_VERSION = 1


@dataclass
class Trade:
    """One change in a coin's executed weight, decided on ``day``'s close."""

    day: date
    coin: str
    side: str  # "buy" (weight up) or "sell" (weight down)
    old_weight: float
    new_weight: float
    notional: float  # |new - old| * the day's mark, in USD
    cost: float  # cost_per_side * notional, in USD

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        return cls(**{**d, "day": date.fromisoformat(d["day"])})


@dataclass
class Account:
    candidate_id: str
    start: date
    data_start: date
    starting_equity: float = STARTING_EQUITY
    cash: float = STARTING_EQUITY
    weights: dict[str, float] = field(default_factory=dict)  # executed, decided on last_day's close
    prev_weights: dict[str, float] = field(default_factory=dict)  # executed, before last_day's trades
    last_day: date | None = None  # last decision day stepped
    equity_history: list[tuple[date, float]] = field(default_factory=list)  # (day, mark at its close)
    trade_log: list[Trade] = field(default_factory=list)

    @classmethod
    def new(cls, candidate_id: str, start: date, starting_equity: float = STARTING_EQUITY) -> "Account":
        return cls(
            candidate_id=candidate_id,
            start=start,
            data_start=start - timedelta(days=J.LOOKBACK_DAYS),
            starting_equity=float(starting_equity),
            cash=float(starting_equity),
        )

    @property
    def first_decision_day(self) -> date:
        return self.start - timedelta(days=1)

    @property
    def equity(self) -> float:
        """The latest mark (starting equity before the first step)."""
        return self.equity_history[-1][1] if self.equity_history else self.starting_equity

    # ------------------------------------------------------------ JSON

    def to_dict(self) -> dict:
        return {
            "version": JSON_VERSION,
            "candidate_id": self.candidate_id,
            "start": self.start.isoformat(),
            "data_start": self.data_start.isoformat(),
            "starting_equity": self.starting_equity,
            "cash": self.cash,
            "weights": dict(self.weights),
            "prev_weights": dict(self.prev_weights),
            "last_day": self.last_day.isoformat() if self.last_day else None,
            "equity_history": [[d.isoformat(), v] for d, v in self.equity_history],
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
            starting_equity=float(d["starting_equity"]),
            cash=float(d["cash"]),
            weights={k: float(v) for k, v in d["weights"].items()},
            prev_weights={k: float(v) for k, v in d["prev_weights"].items()},
            last_day=date.fromisoformat(d["last_day"]) if d["last_day"] else None,
            equity_history=[(date.fromisoformat(day), float(v)) for day, v in d["equity_history"]],
            trade_log=[Trade.from_dict(t) for t in d["trade_log"]],
        )

    def to_json(self) -> str:
        # json writes floats with repr, which round-trips exactly.
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Account":
        return cls.from_dict(json.loads(text))
