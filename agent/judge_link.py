"""The link to the judge: import path and the startup check.

The judge (github.com/Haikeysgit/tektradoor-judge) is a git submodule at
vendor/tektradoor-judge. It is not installed; its directory goes first on
sys.path so ``import judge`` loads the vendored code and nothing else.

Startup check: a prereg's ``judge_tree_hash`` (the git tree of judge/ when the
rules were registered) must equal ``git -C vendor/tektradoor-judge rev-parse
HEAD:judge``, and judge/ must have no uncommitted changes. Otherwise the agent
refuses to run that prereg: the code that would trade is not the code that
was judged.
"""
from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "tektradoor-judge"

if not (VENDOR / "judge" / "__init__.py").exists():
    raise ImportError(
        f"the judge submodule is missing at {VENDOR}; run: git submodule update --init"
    )
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import judge  # noqa: E402

if Path(judge.__file__).resolve().parent != (VENDOR / "judge").resolve():
    raise ImportError(f"judge was imported from {judge.__file__}, not from {VENDOR / 'judge'}")

from judge import judge as J  # noqa: E402,F401  (the judge's run() module)


class JudgeTreeMismatch(RuntimeError):
    """The vendored judge code is not the code the prereg was registered with."""


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise JudgeTreeMismatch(f"git {' '.join(args)} failed in {repo}: {res.stderr.strip()}")
    return res.stdout.strip()


@lru_cache(maxsize=None)
def _tree_state(repo: str) -> tuple[str, str]:
    """(HEAD:judge tree hash, porcelain status of judge/). Read once per process."""
    path = Path(repo)
    tree = _git(path, "rev-parse", "HEAD:judge")
    dirty = _git(path, "status", "--porcelain", "--untracked-files=all", "--", "judge")
    return tree, dirty


def vendor_tree_hash(repo: Path = VENDOR) -> str:
    """``git -C <repo> rev-parse HEAD:judge``."""
    return _tree_state(str(Path(repo).resolve()))[0]


def check_judge_tree(prereg: dict, repo: Path = VENDOR) -> None:
    """Refuse (JudgeTreeMismatch) unless the prereg was registered against this judge code."""
    tree, dirty = _tree_state(str(Path(repo).resolve()))
    name = prereg.get("candidate_id", "prereg")
    want = prereg.get("judge_tree_hash")
    if want != tree:
        raise JudgeTreeMismatch(
            f"{name}: judge_tree_hash {want} does not match the vendored judge "
            f"(git -C {repo} rev-parse HEAD:judge = {tree}); refusing to run rules on other code"
        )
    if dirty:
        raise JudgeTreeMismatch(f"{name}: uncommitted changes under {repo}/judge: {dirty.splitlines()}")


def load_prereg(path: Path) -> dict:
    """Read a prereg with the judge's own validation, then run the startup check."""
    prereg = J.load_prereg(Path(path))
    check_judge_tree(prereg)
    return prereg
