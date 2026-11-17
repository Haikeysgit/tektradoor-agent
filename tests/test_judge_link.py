import copy
import re
import subprocess
from pathlib import Path

import pytest

from agent import judge_link
from agent.judge_link import JudgeTreeMismatch, check_judge_tree, load_prereg, vendor_tree_hash

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "tektradoor-judge"
PREREGS = sorted((VENDOR / "prereg").glob("[0-9][0-9]-*.json"))


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_judge_is_imported_from_the_submodule():
    import judge

    assert Path(judge.__file__).resolve().parent == (VENDOR / "judge").resolve()


def test_submodule_is_pinned_to_the_registered_commit():
    assert _git(VENDOR, "rev-parse", "HEAD").startswith("143012e")
    assert vendor_tree_hash() == "c0398cd440cea365009cedee9fc97c609eea2106"


def test_registered_preregs_match_the_vendored_judge():
    assert len(PREREGS) == 2
    for path in PREREGS:
        prereg = load_prereg(path)
        check_judge_tree(prereg)  # no error


def test_tree_hash_mismatch_is_refused():
    prereg = load_prereg(PREREGS[0])
    bad = copy.deepcopy(prereg)
    bad["judge_tree_hash"] = "0" * 40
    with pytest.raises(JudgeTreeMismatch, match="judge_tree_hash"):
        check_judge_tree(bad)


def test_dirty_judge_tree_is_refused(tmp_path):
    repo = tmp_path / "judge-copy"
    (repo / "judge").mkdir(parents=True)
    (repo / "judge" / "x.py").write_text("A = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    tree = _git(repo, "rev-parse", "HEAD:judge")
    prereg = {"candidate_id": "x", "judge_tree_hash": tree}
    check_judge_tree(prereg, repo=repo)  # clean: passes
    (repo / "judge" / "x.py").write_text("A = 2\n", encoding="utf-8")
    judge_link._tree_state.cache_clear()  # the state is read once per process, at startup
    with pytest.raises(JudgeTreeMismatch, match="uncommitted"):
        check_judge_tree(prereg, repo=repo)


def test_load_prereg_uses_the_judges_validation(tmp_path):
    raw = PREREGS[0].read_text(encoding="utf-8").replace('"cost_per_side": 0.0011', '"cost_per_side": 0.001')
    path = tmp_path / "01-bad.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(judge_link.J.PreregError, match="cost_per_side"):
        load_prereg(path)


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_library_pins_match_the_judge():
    assert _pins(ROOT / "requirements.txt") == _pins(VENDOR / "requirements.txt")
