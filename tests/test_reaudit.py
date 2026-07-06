"""Tests for `dbt_fixer.reaudit`: the Re-Audit Gate, fully offline.

Every test injects a fake `SubprocessRunner` -- never a real subprocess --
matching the `conftest.py`-enforced offline contract. Fakes assert on
exactly what args/env/cwd/timeout they were called with, so this suite
also proves the gate invokes the sealed auditor correctly (shadow mode, no
Slack channel, pointed at a patched scratch copy).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import pytest

from dbt_fixer.diffing import generate_unified_diff
from dbt_fixer.reaudit import (
    AuditorInvocationError,
    ENTRYPOINT_MODULE,
    ProcessOutcome,
    build_auditor_env,
    run_reaudit_gate,
)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "a.sql").write_text("select 1\nfrom x\n")
    return repo


def _candidate_diff(repo: Path, tmp_path: Path) -> str:
    after = tmp_path / "after"
    import shutil

    shutil.copytree(repo, after)
    (after / "models" / "a.sql").write_text("select 1\nfrom x\nwhere y = 1\n")
    return generate_unified_diff(repo, after, ["models/a.sql"])


class _RecordingRunner:
    """A fake `SubprocessRunner` that records every call and returns a canned outcome."""

    def __init__(self, outcome: ProcessOutcome | None = None, *, raise_invocation_error: bool = False):
        self.calls: List[dict] = []
        self._outcome = outcome
        self._raise = raise_invocation_error

    def __call__(self, args: list, env: Mapping[str, str], cwd: Path, timeout: float) -> ProcessOutcome:
        # Snapshot the scratch repo's content *while it still exists* (the
        # scratch dir is torn down as soon as `run_reaudit_gate` returns).
        scratch_repo_path = Path(env["DBT_AUDITOR_REPO_PATH"])
        sql_snapshot = (scratch_repo_path / "models" / "a.sql").read_text()
        self.calls.append(
            {
                "args": list(args),
                "env": dict(env),
                "cwd": cwd,
                "timeout": timeout,
                "scratch_repo_path": scratch_repo_path,
                "sql_snapshot": sql_snapshot,
            }
        )
        if self._raise:
            raise AuditorInvocationError("no such interpreter")
        assert self._outcome is not None
        return self._outcome


_PASSED_STDOUT = "dbt-auditor-audit-status: completed\ndbt-auditor verdict: PASSED - looks fixed\n"
_BLOCKED_STDOUT = "dbt-auditor-audit-status: completed\ndbt-auditor verdict: BLOCKED - still broken\n"


def _base_kwargs(repo: Path, candidate_diff: str, runner: _RecordingRunner, **overrides) -> dict:
    kwargs = dict(
        repo_root=repo,
        candidate_diff=candidate_diff,
        pr_diff="",
        pr_title="Fix broken test",
        pr_description="restores a deleted line",
        pr_url="https://github.com/example/repo/pull/1",
        auditor_python="/usr/bin/python3.11",
        failure_kind="ci",
        originally_failing_check_ids=(),
        timeout_seconds=30.0,
        subprocess_runner=runner,
    )
    kwargs.update(overrides)
    return kwargs


# --- correct subprocess invocation ------------------------------------------


def test_invokes_auditor_in_shadow_mode_with_no_slack_channel_against_patched_scratch(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    verdict = run_reaudit_gate(**_base_kwargs(repo, candidate_diff, runner))

    assert verdict.passed
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["args"][0] == "/usr/bin/python3.11"
    assert ENTRYPOINT_MODULE in call["args"]
    assert call["env"]["DBT_AUDITOR_SHADOW_MODE"] == "true"
    assert "DBT_AUDITOR_SLACK_CHANNEL" not in call["env"]
    assert call["timeout"] == 30.0

    # The auditor was pointed at a scratch copy (not repo_root itself) that
    # actually has the candidate diff applied.
    scratch_repo_path = Path(call["env"]["DBT_AUDITOR_REPO_PATH"])
    assert scratch_repo_path != repo
    assert call["sql_snapshot"] == "select 1\nfrom x\nwhere y = 1\n"
    # And the original checkout was never mutated.
    assert (repo / "models" / "a.sql").read_text() == "select 1\nfrom x\n"
    # The scratch dir created for this gate call is cleaned up afterward.
    assert not scratch_repo_path.exists()


def test_build_auditor_env_never_includes_a_slack_channel() -> None:
    env = build_auditor_env(
        repo_path=Path("/tmp/x"), pr_diff="d", pr_title="t", pr_description="desc", pr_url="u"
    )
    assert "DBT_AUDITOR_SLACK_CHANNEL" not in env
    assert env["DBT_AUDITOR_SHADOW_MODE"] == "true"


# --- BLOCKED verdict handling -------------------------------------------------


def test_blocked_verdict_fails_the_candidate(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=_BLOCKED_STDOUT))

    verdict = run_reaudit_gate(**_base_kwargs(repo, candidate_diff, runner))

    assert not verdict.passed
    assert not verdict.hard_no_safe_fix
    assert verdict.violation == "auditor_verdict_blocked"
    assert "still broken" in verdict.reason


@pytest.mark.parametrize("verdict_word", ["PASSED", "NEEDS_REVIEW"])
def test_non_blocked_verdict_allows_the_candidate_to_proceed(tmp_path: Path, verdict_word: str) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    stdout = f"dbt-auditor-audit-status: completed\ndbt-auditor verdict: {verdict_word} - ok\n"
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=stdout))

    verdict = run_reaudit_gate(**_base_kwargs(repo, candidate_diff, runner))
    assert verdict.passed


# --- per-check-passing requirement for audit-kind fixes ----------------------

_REPORT_ALL_PASSING = (
    "dbt-auditor-audit-status: completed\n"
    "dbt-auditor verdict: PASSED - ok\n"
    "dbt-auditor-report-begin\n"
    "- check: not_null_orders_id\n"
    "status: PASS\n"
    "- check: unique_orders_id\n"
    "status: PASS\n"
    "dbt-auditor-report-end\n"
)

_REPORT_ONE_STILL_FAILING = (
    "dbt-auditor-audit-status: completed\n"
    "dbt-auditor verdict: PASSED - ok\n"
    "dbt-auditor-report-begin\n"
    "- check: not_null_orders_id\n"
    "status: PASS\n"
    "- check: unique_orders_id\n"
    "status: FAIL\n"
    "dbt-auditor-report-end\n"
)


def test_audit_kind_fails_when_one_originally_failing_check_still_fails(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=_REPORT_ONE_STILL_FAILING))

    verdict = run_reaudit_gate(
        **_base_kwargs(
            repo,
            candidate_diff,
            runner,
            failure_kind="audit",
            originally_failing_check_ids=("not_null_orders_id", "unique_orders_id"),
        )
    )
    assert not verdict.passed
    assert verdict.violation == "auditor_check_still_failing"
    assert "unique_orders_id" in verdict.reason


def test_audit_kind_passes_only_when_every_originally_failing_check_now_passes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=_REPORT_ALL_PASSING))

    verdict = run_reaudit_gate(
        **_base_kwargs(
            repo,
            candidate_diff,
            runner,
            failure_kind="audit",
            originally_failing_check_ids=("not_null_orders_id", "unique_orders_id"),
        )
    )
    assert verdict.passed


def test_ci_kind_does_not_require_the_report_block_at_all(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    verdict = run_reaudit_gate(
        **_base_kwargs(repo, candidate_diff, runner, failure_kind="ci", originally_failing_check_ids=())
    )
    assert verdict.passed


# --- missing interpreter is a hard no_safe_fix -------------------------------


def test_unconfigured_auditor_python_is_a_hard_no_safe_fix(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    verdict = run_reaudit_gate(**_base_kwargs(repo, candidate_diff, runner, auditor_python=None))

    assert not verdict.passed
    assert verdict.hard_no_safe_fix
    assert verdict.violation == "auditor_interpreter_missing"
    # The gate must never even attempt to invoke the subprocess in this case.
    assert runner.calls == []


def test_uninvokable_auditor_interpreter_is_a_hard_no_safe_fix(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(raise_invocation_error=True)

    verdict = run_reaudit_gate(
        **_base_kwargs(repo, candidate_diff, runner, auditor_python="/does/not/exist")
    )

    assert not verdict.passed
    assert verdict.hard_no_safe_fix
    assert verdict.violation == "auditor_interpreter_missing"
    # Distinguishable from a normal gate failure/skip by the explicit flag.
    assert verdict.violation != "auditor_nonzero_exit"


# --- nonzero exit / unparsable output ----------------------------------------


def test_nonzero_exit_is_a_gate_failure_distinct_from_blocked_and_missing_interpreter(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(ProcessOutcome(returncode=1, stdout="", stderr="traceback..."))

    verdict = run_reaudit_gate(**_base_kwargs(repo, candidate_diff, runner))

    assert not verdict.passed
    assert not verdict.hard_no_safe_fix
    assert verdict.violation == "auditor_nonzero_exit"


def test_timeout_outcome_is_a_gate_failure(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    # A subprocess_runner implementation surfaces a timeout as a completed,
    # non-zero-exit outcome (never raising) -- this gate treats it exactly
    # like any other nonzero exit.
    runner = _RecordingRunner(ProcessOutcome(returncode=124, stdout="", stderr="timed out"))

    verdict = run_reaudit_gate(**_base_kwargs(repo, candidate_diff, runner))

    assert not verdict.passed
    assert verdict.violation == "auditor_nonzero_exit"


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "some unrelated log noise\n",
        "dbt-auditor-audit-status: completed\n",  # no verdict line at all
        "dbt-auditor verdict: PASSED - ok\n",  # no status line at all
    ],
)
def test_unparsable_stdout_is_a_gate_failure_not_a_crash(tmp_path: Path, stdout: str) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=stdout))

    verdict = run_reaudit_gate(**_base_kwargs(repo, candidate_diff, runner))

    assert not verdict.passed
    assert not verdict.hard_no_safe_fix
    assert verdict.violation == "auditor_output_unparsable"


def test_status_failed_is_treated_as_a_gate_failure(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    candidate_diff = _candidate_diff(repo, tmp_path)
    stdout = "dbt-auditor-audit-status: failed\ndbt-auditor verdict: PASSED - ok\n"
    runner = _RecordingRunner(ProcessOutcome(returncode=0, stdout=stdout))

    verdict = run_reaudit_gate(**_base_kwargs(repo, candidate_diff, runner))

    assert not verdict.passed
    assert verdict.violation == "auditor_output_unparsable"
