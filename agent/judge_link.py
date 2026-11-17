"""The link to the judge: import path, registered preregs and the per-step check.

The judge (github.com/Haikeysgit/tektradoor-judge) is a git submodule at
vendor/tektradoor-judge. It is not installed; its directory goes first on
sys.path so ``import judge`` loads the vendored code and nothing else.

The agent runs registered rules only:

* ``load_prereg`` accepts a prereg only from vendor/tektradoor-judge/prereg/
  and only if the judge's own ``check_integrity`` passes (committed once,
  bytes equal the commit, sha256 equals the .sha256 registration, its
  judge_tree_hash equals HEAD:judge, nothing uncommitted under judge/ or
  prereg/). The account stores the prereg's sha256 and the judge tree hash.
* ``verify_step`` runs before every daily step: it re-hashes the prereg file,
  re-reads ``git rev-parse HEAD:judge`` and ``git status`` in the submodule
  (nothing is cached) and refuses unless both hashes equal the account's and
  judge/ and prereg/ are clean. The rules come from the bytes just hashed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
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

from judge import judge as J  # noqa: E402  (the judge's run() module)


class VerificationError(RuntimeError):
    """The agent cannot prove it would run registered rules on the judged code."""


class JudgeTreeMismatch(VerificationError):
    """The vendored judge code is not the code the prereg was registered with."""


class PreregNotRegistered(VerificationError):
    """The prereg is not a registered, unedited prereg of the vendored judge."""


@dataclass(frozen=True)
class Registered:
    """A prereg that passed the judge's integrity check."""

    path: Path
    rules: dict
    sha256: str  # of the committed (= working) file bytes, as in its .sha256 registration
    judge_tree_hash: str  # git rev-parse HEAD:judge in the submodule


def _vendor() -> Path:
    return Path(VENDOR).resolve()  # read at call time (tests point it at a clone)


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise JudgeTreeMismatch(f"git {' '.join(args)} failed in {repo}: {res.stderr.strip()}")
    return res.stdout.strip()


def vendor_tree_hash() -> str:
    """``git -C vendor/tektradoor-judge rev-parse HEAD:judge``, read fresh each call."""
    return _git(_vendor(), "rev-parse", "HEAD:judge")


def _registered_path(path: Path) -> Path:
    path = Path(path).resolve()
    home = _vendor() / "prereg"
    if path.parent != home:
        raise PreregNotRegistered(f"{path}: the agent runs only registered preregs, from {home}")
    return path


def load_prereg(path: Path) -> Registered:
    """Accept only a registered prereg of the vendored judge (the judge's check_integrity)."""
    path = _registered_path(path)
    try:
        J.check_integrity(path)
        rules = J.load_prereg(path)
        sha = J.committed_sha256(path)
    except J.JudgeRefusal as exc:
        raise PreregNotRegistered(f"{path.name} is not a registered, unedited prereg: {exc}") from exc
    return Registered(path=path, rules=rules, sha256=sha, judge_tree_hash=vendor_tree_hash())


def verify_step(path: Path, prereg_sha256: str, judge_tree_hash: str) -> dict:
    """Re-check before a step and return the rules read from the bytes just hashed.

    Refuses unless the prereg is the vendored file, its sha256 equals
    ``prereg_sha256``, HEAD:judge equals ``judge_tree_hash`` (and the prereg's
    own field), nothing is uncommitted under judge/ or prereg/, and the
    judge's ``check_integrity`` still passes.
    """
    path = _registered_path(path)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != prereg_sha256:
        raise PreregNotRegistered(f"{path.name}: sha256 is {sha}, the account was opened on {prereg_sha256}")
    tree = vendor_tree_hash()
    if tree != judge_tree_hash:
        raise JudgeTreeMismatch(
            f"vendored judge HEAD:judge is {tree}, the account's judge_tree_hash is {judge_tree_hash}; "
            "refusing to run rules on other code"
        )
    dirty = _git(_vendor(), "status", "--porcelain", "--untracked-files=all", "--", "judge", "prereg")
    if dirty:
        raise JudgeTreeMismatch(f"uncommitted changes in the vendored judge: {dirty.splitlines()}")
    # The judge's own guard as well: the file is still the registered one
    # (sha256 equals its .sha256, committed once), so a prereg edited and
    # committed in the submodule is refused even if the account's stored
    # sha256 was edited to match it.
    try:
        J.check_integrity(path)
    except J.JudgeRefusal as exc:
        raise PreregNotRegistered(f"{path.name} is not a registered, unedited prereg: {exc}") from exc
    if J.file_sha256(path) != sha:
        raise PreregNotRegistered(f"{path.name} changed while it was being verified")
    rules = json.loads(raw)
    missing = [k for k in J.REQUIRED_KEYS if k not in rules]
    if missing:
        raise PreregNotRegistered(f"{path.name}: missing required key(s): {', '.join(missing)}")
    J._validate(rules, path.name)
    if rules["judge_tree_hash"] != tree:
        raise JudgeTreeMismatch(f"{path.name}: judge_tree_hash {rules['judge_tree_hash']} is not HEAD:judge {tree}")
    return rules
