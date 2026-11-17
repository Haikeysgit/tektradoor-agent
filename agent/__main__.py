"""CLI.

    python -m agent replay prereg.json --start YYYY-MM-DD --end YYYY-MM-DD [--cache-dir DIR] [--save FILE]

replay steps the agent day by day over the judge's window start..end (the
book is flat at $10,000 on the close before start) and prints the final
equity, the trade count and the max drawdown. --save writes the account JSON.

Exit codes: 0 done, 2 refused (prereg invalid, judge code mismatch, bad
inputs), 3 a data error (download, integrity, data not ready).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from agent import replay as replay_mod
from agent.judge_link import J, JudgeTreeMismatch
from agent.step import StepError
from judge import data as judge_data


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {text!r}") from exc


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="step the agent over a past window")
    rp.add_argument("prereg", type=Path)
    rp.add_argument("--start", type=_parse_date, required=True, help="first day whose return counts")
    rp.add_argument("--end", type=_parse_date, required=True, help="last day whose return counts")
    rp.add_argument("--cache-dir", type=Path, default=None, help="data cache (default: data_cache/)")
    rp.add_argument("--save", type=Path, default=None, help="write the final account JSON here")
    return ap


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        account = replay_mod.replay(args.prereg, args.start, args.end, cache_dir=args.cache_dir)
    except J.DataNotReady as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 3
    except (J.JudgeRefusal, JudgeTreeMismatch, StepError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except (judge_data.DataIntegrityError, judge_data.ArchiveFetchError, judge_data.MissingArchiveFile,
            judge_data.ChecksumMismatch, OSError) as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 3

    s = replay_mod.summary(account)
    print(f"{s['candidate_id']}  {s['start']} .. {s['end']}")
    print(f"final equity: {s['final_equity']:,.2f}")
    print(f"trades: {s['trades']}")
    print(f"max drawdown: {s['max_drawdown']:.2%}")
    if args.save:
        args.save.write_text(account.to_json(), encoding="utf-8", newline="\n")
        print(f"account saved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
