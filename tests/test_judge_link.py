import json
import re
import subprocess
from pathlib import Path

import pytest

from agent import judge_link
from agent.judge_link import (
    JudgeTreeMismatch,
    PreregNotRegistered,
    load_prereg,
    vendor_tree_hash,
    verify_step,
)

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "tektradoor-judge"
PREREGS = sorted((VENDOR / "prereg").glob("[0-9][0-9]-*.json"))


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def judge_clone(tmp_path, monkeypatch):
    """A full clone of the vendored judge, used as the agent's vendor dir."""
    clone = tmp_path / "judge-clone"
    _git(tmp_path, "clone", "-q", "--no-checkout", str(VENDOR), str(clone))
    _git(clone, "config", "core.autocrlf", "false")
    _git(clone, "checkout", "-q", _git(VENDOR, "rev-parse", "HEAD"))
    monkeypatch.setattr(judge_link, "VENDOR", clone)
    return clone


def test_judge_is_imported_from_the_submodule():
    import judge

    assert Path(judge.__file__).resolve().parent == (VENDOR / "judge").resolve()


def test_submodule_is_pinned_to_the_registered_commit():
    assert _git(VENDOR, "rev-parse", "HEAD").startswith("143012e")
    assert vendor_tree_hash() == "c0398cd440cea365009cedee9fc97c609eea2106"


def test_registered_preregs_load_with_their_registration():
    assert len(PREREGS) == 2
    for path in PREREGS:
        reg = load_prereg(path)
        assert reg.path == path.resolve()
        assert reg.sha256 == path.with_suffix(".sha256").read_text().split()[0].lower()
        assert reg.judge_tree_hash == vendor_tree_hash() == reg.rules["judge_tree_hash"]
        assert verify_step(path, reg.sha256, reg.judge_tree_hash) == reg.rules


def test_an_edited_copy_elsewhere_is_refused(tmp_path):
    raw = json.loads(PREREGS[0].read_text(encoding="utf-8"))
    raw["lookbacks"], raw["vol_target"] = [5, 10], 0.9  # same judge_tree_hash string
    copy = tmp_path / "prereg" / PREREGS[0].name
    copy.parent.mkdir()
    copy.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(PreregNotRegistered, match="registered"):
        load_prereg(copy)
    reg = load_prereg(PREREGS[0])
    with pytest.raises(PreregNotRegistered, match="registered"):
        verify_step(copy, reg.sha256, reg.judge_tree_hash)
    # An unedited copy elsewhere is refused too: only the vendored file counts.
    copy.write_bytes(PREREGS[0].read_bytes())
    with pytest.raises(PreregNotRegistered):
        load_prereg(copy)


def test_an_edited_vendored_prereg_is_refused(judge_clone):
    path = judge_clone / "prereg" / PREREGS[0].name
    reg = load_prereg(path)  # the clean clone passes the judge's integrity check
    path.write_bytes(path.read_bytes().replace(b'"vol_target": 0.25', b'"vol_target": 0.9'))
    with pytest.raises(PreregNotRegistered):
        load_prereg(path)
    with pytest.raises(PreregNotRegistered, match="sha256"):
        verify_step(path, reg.sha256, reg.judge_tree_hash)


def test_an_edited_vendored_judge_file_is_refused(judge_clone):
    path = judge_clone / "prereg" / PREREGS[0].name
    reg = load_prereg(path)
    costs = judge_clone / "judge" / "costs.py"
    costs.write_bytes(costs.read_bytes() + b"\n# edited\n")
    with pytest.raises(JudgeTreeMismatch, match="uncommitted"):
        verify_step(path, reg.sha256, reg.judge_tree_hash)
    with pytest.raises(PreregNotRegistered):
        load_prereg(path)


def test_tree_hash_is_reread_every_time(judge_clone):
    path = judge_clone / "prereg" / PREREGS[0].name
    reg = load_prereg(path)
    costs = judge_clone / "judge" / "costs.py"
    costs.write_bytes(costs.read_bytes() + b"\n# edited\n")
    _git(judge_clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "edit judge")
    assert vendor_tree_hash() != reg.judge_tree_hash  # no caching
    with pytest.raises(JudgeTreeMismatch, match="judge_tree_hash"):
        verify_step(path, reg.sha256, reg.judge_tree_hash)


def test_an_account_pinned_to_other_values_is_refused():
    reg = load_prereg(PREREGS[0])
    with pytest.raises(PreregNotRegistered, match="sha256"):
        verify_step(PREREGS[0], "0" * 64, reg.judge_tree_hash)
    with pytest.raises(JudgeTreeMismatch, match="judge_tree_hash"):
        verify_step(PREREGS[0], reg.sha256, "0" * 40)


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_library_pins_match_the_judge():
    assert _pins(ROOT / "requirements.txt") == _pins(VENDOR / "requirements.txt")
