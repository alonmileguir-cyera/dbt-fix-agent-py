"""Tests for `dbt_fixer.dbt_parse`: the best-effort dbt Parse Gate, fully offline.

Every test injects a fake `which` and/or a fake `DbtSubprocessRunner` --
never a real PATH lookup or a real subprocess -- matching the
`conftest.py`-enforced offline contract.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import List

import pytest

from dbt_fixer.dbt_parse import (
    DbtInvocationError,
    DbtParseTimeoutError,
    find_touched_project_dir,
    run_dbt_parse_gate,
)
from dbt_fixer.diffing import generate_unified_diff
from dbt_fixer.reaudit import ProcessOutcome
from dbt_fixer.scratch import ScratchCopyError


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "dbt_project.yml").write_text("name: my_project\n")
    (repo / "models" / "a.sql").write_text("select 1\nfrom x\n")
    return repo


def _candidate_diff(repo: Path, tmp_path: Path, new_content: str) -> str:
    after = tmp_path / "after"
    shutil.copytree(repo, after)
    (after / "models" / "a.sql").write_text(new_content)
    return generate_unified_diff(repo, after, ["models/a.sql"])


class _RecordingDbtRunner:
    def __init__(self, outcome: ProcessOutcome | None = None, *, error: Exception | None = None):
        self.calls: List[dict] = []
        self._outcome = outcome
        self._error = error

    def __call__(self, argv, cwd, timeout_seconds):
        cwd = Path(cwd)
        # Captured *during* the call, while the scratch copy is still alive
        # (it is torn down before `run_dbt_parse_gate` returns).
        self.calls.append(
            {
                "argv": list(argv),
                "cwd": cwd,
                "timeout": timeout_seconds,
                "cwd_name": cwd.name,
                "cwd_has_project_file": (cwd / "dbt_project.yml").is_file(),
            }
        )
        if self._error is not None:
            raise self._error
        return self._outcome


def _fake_which(path: str | None = "/usr/local/bin/dbt"):
    return lambda name: path if name == "dbt" else None


# ---------------------------------------------------------------------------
# dbt_parse_gate_runs_when_available
# ---------------------------------------------------------------------------


def test_gate_runs_dbt_parse_against_touched_project_dir_and_passes(tmp_path):
    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")
    runner = _RecordingDbtRunner(ProcessOutcome(returncode=0, stdout="", stderr=""))

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "passed"
    assert verdict.passed is True
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["argv"] == ["/usr/local/bin/dbt", "parse"]
    assert call["timeout"] == 10.0
    # The cwd handed to dbt must be the scratch copy's project root (the
    # directory containing dbt_project.yml), never the repo_root itself.
    assert call["cwd_has_project_file"] is True
    assert call["cwd"] != repo


def test_gate_kills_candidate_on_nonzero_exit_when_baseline_is_clean(tmp_path):
    """Differential contract: a nonzero candidate parse only kills the
    candidate when the unpatched baseline parses cleanly (the patch is
    what broke parse). The candidate scratch is probed first, the
    baseline second."""
    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")

    outcomes = [
        ProcessOutcome(returncode=1, stdout="", stderr="Compilation Error: bad ref()"),
        ProcessOutcome(returncode=0, stdout="", stderr=""),  # baseline: clean
    ]
    calls = []

    def runner(args, cwd, timeout_seconds):
        calls.append({"args": args, "cwd": cwd})
        return outcomes[len(calls) - 1]

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "failed"
    assert verdict.passed is False
    assert "bad ref" in verdict.reason
    assert len(calls) == 2  # candidate then baseline


def test_gate_skips_when_baseline_also_fails_to_parse(tmp_path):
    """Environmental failure (missing profile/deps, pre-existing project
    error): the unpatched baseline fails too, so the candidate's nonzero
    exit is not the patch's fault -> best-effort skip, never a kill.
    This is exactly the e2e run-13 case ('Could not find profile')."""
    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")
    runner = _RecordingDbtRunner(
        ProcessOutcome(returncode=2, stdout="", stderr="Could not find profile")
    )

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "skipped"
    assert verdict.passed is False
    assert "environmental" in verdict.reason


def test_gate_kills_candidate_on_timeout(tmp_path):
    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")
    runner = _RecordingDbtRunner(error=DbtParseTimeoutError("exceeded 5s"))

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=5.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "failed"
    assert verdict.passed is False
    assert "timed out" in verdict.reason.lower()


def test_gate_kills_candidate_when_runner_blocks_past_timeout(tmp_path):
    """`dbt_parse_gate_bounded_timeout`: the subprocess call is wrapped in a
    hard, interrupting wall-clock timeout -- a runner that itself never
    raises `DbtParseTimeoutError` but simply sleeps past the bound must
    still resolve to `"failed"`, promptly, never a hang."""

    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")

    def slow_runner(argv, cwd, timeout_seconds):
        time.sleep(0.5)
        return ProcessOutcome(returncode=0, stdout="", stderr="")

    start = time.monotonic()
    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=0.05,
        subprocess_runner=slow_runner,
        which=_fake_which(),
    )
    elapsed = time.monotonic() - start

    assert verdict.outcome == "failed"
    assert verdict.passed is False
    assert "timeout" in verdict.reason.lower() or "timed out" in verdict.reason.lower()
    # The gate must return promptly, not wait for the slow runner to finish.
    assert elapsed < 0.4


def test_gate_kills_candidate_when_runner_hangs_forever_without_hanging_the_test(tmp_path):
    """A runner that never returns at all (blocks on an event that's never
    set) must not hang the gate or the test -- the daemon-thread hard
    timeout must still resolve this within `timeout_seconds`."""

    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")

    def hangs_forever(argv, cwd, timeout_seconds):
        threading.Event().wait()  # never set; would block forever off a daemon thread
        return ProcessOutcome(returncode=0, stdout="", stderr="")  # unreachable

    start = time.monotonic()
    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=0.05,
        subprocess_runner=hangs_forever,
        which=_fake_which(),
    )
    elapsed = time.monotonic() - start

    assert verdict.outcome == "failed"
    assert verdict.passed is False
    assert elapsed < 0.4


def test_gate_finds_nested_project_dir(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub" / "models").mkdir(parents=True)
    (repo / "sub" / "dbt_project.yml").write_text("name: nested\n")
    (repo / "sub" / "models" / "a.sql").write_text("select 1\n")

    after = tmp_path / "after"
    shutil.copytree(repo, after)
    (after / "sub" / "models" / "a.sql").write_text("select 2\n")
    diff = generate_unified_diff(repo, after, ["sub/models/a.sql"])

    runner = _RecordingDbtRunner(ProcessOutcome(returncode=0, stdout="", stderr=""))
    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "passed"
    assert runner.calls[0]["cwd_name"] == "sub"


# ---------------------------------------------------------------------------
# dbt_parse_gate_honest_skip_when_unavailable
# ---------------------------------------------------------------------------


def test_gate_skips_when_dbt_not_on_path(tmp_path):
    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")
    runner = _RecordingDbtRunner(ProcessOutcome(returncode=0, stdout="", stderr=""))

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(None),
    )

    assert verdict.outcome == "skipped"
    assert verdict.passed is False
    assert "not" in verdict.reason.lower() or "no dbt" in verdict.reason.lower()
    assert runner.calls == []


def test_gate_skips_when_invocation_fails(tmp_path):
    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")
    runner = _RecordingDbtRunner(error=DbtInvocationError("dbt disappeared"))

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "skipped"
    assert verdict.passed is False


# ---------------------------------------------------------------------------
# dbt_parse_gate_skip_on_scratch_copy_failure
# ---------------------------------------------------------------------------


def test_gate_skips_on_scratch_copy_failure(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    diff = _candidate_diff(repo, tmp_path, "select 1\nfrom x\nwhere y = 1\n")
    runner = _RecordingDbtRunner(ProcessOutcome(returncode=0, stdout="", stderr=""))

    def _boom(*args, **kwargs):
        raise ScratchCopyError("simulated filesystem/permission error")

    monkeypatch.setattr("dbt_fixer.dbt_parse.scratch_copy", _boom)

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "skipped"
    assert verdict.passed is False
    assert "scratch copy" in verdict.reason.lower()
    assert runner.calls == []


def test_gate_skips_when_no_project_dir_found(tmp_path):
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "a.sql").write_text("select 1\n")
    # Deliberately no dbt_project.yml anywhere in this tree.

    after = tmp_path / "after"
    shutil.copytree(repo, after)
    (after / "models" / "a.sql").write_text("select 2\n")
    diff = generate_unified_diff(repo, after, ["models/a.sql"])

    runner = _RecordingDbtRunner(ProcessOutcome(returncode=0, stdout="", stderr=""))
    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "skipped"
    assert runner.calls == []


def test_gate_skips_when_candidate_diff_does_not_apply(tmp_path):
    repo = _make_repo(tmp_path)
    bad_diff = (
        "diff --git a/models/a.sql b/models/a.sql\n"
        "--- a/models/a.sql\n"
        "+++ b/models/a.sql\n"
        "@@ -1,1 +1,1 @@\n"
        "-this line does not exist in the file\n"
        "+replacement\n"
    )
    runner = _RecordingDbtRunner(ProcessOutcome(returncode=0, stdout="", stderr=""))

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=bad_diff,
        timeout_seconds=10.0,
        subprocess_runner=runner,
        which=_fake_which(),
    )

    assert verdict.outcome == "skipped"
    assert runner.calls == []


# ---------------------------------------------------------------------------
# find_touched_project_dir (unit-level)
# ---------------------------------------------------------------------------


def test_find_touched_project_dir_returns_none_for_no_matches(tmp_path):
    (tmp_path / "models").mkdir()
    assert find_touched_project_dir(tmp_path, ["models/a.sql"]) is None


def test_find_touched_project_dir_finds_nearest_ancestor(tmp_path):
    (tmp_path / "sub" / "models").mkdir(parents=True)
    (tmp_path / "sub" / "dbt_project.yml").write_text("name: x\n")
    result = find_touched_project_dir(tmp_path, ["sub/models/a.sql"])
    assert result == (tmp_path / "sub").resolve()
