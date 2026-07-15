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
    assert call["args"][1:3] == ["-P", "-m"]
    assert ENTRYPOINT_MODULE in call["args"]
    assert call["env"]["PYTHONSAFEPATH"] == "1"
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


# ---------------------------------------------------------------------------
# live e2e findings (run 8): env passthrough, knobs, report-file handoff
# ---------------------------------------------------------------------------


def test_auditor_env_passes_credentials_knobs_and_report_path(monkeypatch):
    from dbt_fixer.reaudit import build_auditor_env

    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/tok")
    monkeypatch.setenv("NOT_ALLOWED", "nope")
    env = build_auditor_env(
        repo_path="/scratch", pr_diff="d", pr_title="t",
        pr_description="", pr_url="u", report_path="/tmp/r.md",
    )
    assert env["AWS_REGION"] == "us-east-1"
    assert env["AWS_WEB_IDENTITY_TOKEN_FILE"] == "/var/run/tok"
    assert "NOT_ALLOWED" not in env
    assert env["DBT_AUDITOR_TIMEOUT_SECONDS"] == "900"
    assert env["DBT_AUDITOR_MAX_TOOL_CALLS"] == "50"
    assert env["DBT_AUDITOR_REPORT_PATH"] == "/tmp/r.md"
    assert env["PYTHONSAFEPATH"] == "1"
    assert "DBT_AUDITOR_SLACK_CHANNEL" not in env


def test_check_statuses_parsed_from_real_report_markdown():
    from dbt_fixer.reaudit import _check_statuses_from_report

    report = (
        "# Verdict: **PASSED**\n\n"
        "### Schema Contract Verification (`schema_contract_verification`)\n\n"
        "**Score:** 90/100 &nbsp;&nbsp; **State:** **PASS**\n\n"
        "### Tenant Isolation Integrity (`tenant_isolation_integrity`)\n\n"
        "**Score:** 60/100 &nbsp;&nbsp; **State:** **FAIL**\n"
    )
    statuses = _check_statuses_from_report(report)
    assert statuses == {
        "schema_contract_verification": "PASS",
        "tenant_isolation_integrity": "FAIL",
    }


def test_report_file_statuses_satisfy_the_efficacy_requirement(tmp_path):
    """A fake auditor that writes the real report file (no stdout block)
    must satisfy the originally-failing-check requirement via the file."""
    from dbt_fixer.reaudit import run_reaudit_gate

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yml").write_text("version: 2\n")
    diff = (
        "diff --git a/models/x.yml b/models/x.yml\n"
        "--- a/models/x.yml\n"
        "+++ b/models/x.yml\n"
        "@@ -1 +1,2 @@\n"
        " version: 2\n"
        "+# fix\n"
    )

    def fake_runner(args, env, cwd, timeout_seconds):
        from dbt_fixer.reaudit import ProcessOutcome

        report_path = env["DBT_AUDITOR_REPORT_PATH"]
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(
                "# Verdict: **PASSED**\n\n"
                "### Schema Contract Verification (`schema_contract_verification`)\n\n"
                "**Score:** 95/100 &nbsp;&nbsp; **State:** **PASS**\n"
            )
        return ProcessOutcome(
            returncode=0,
            stdout=(
                "dbt-auditor verdict: PASSED - all clear\n"
                "dbt-auditor-audit-status: completed\n"
            ),
            stderr="",
        )

    verdict = run_reaudit_gate(
        repo_root=tmp_path,
        candidate_diff=diff,
        pr_diff="",
        pr_title="t",
        pr_description="",
        pr_url="u",
        auditor_python="/fake/python",
        failure_kind="audit",
        originally_failing_check_ids=("schema_contract_verification",),
        timeout_seconds=60.0,
        subprocess_runner=fake_runner,
    )
    assert verdict.passed, verdict.reason


def test_reaudit_receives_the_effective_diff(tmp_path):
    """Live findings (bi-dbt #2533 rounds 2-3): the re-audit must judge the
    EFFECTIVE diff, base -> (PR + fix), as one clean change. A fix that
    rewrites a PR-added line must appear in the effective diff as if the
    author had written the fixed line in the first place - the PR's broken
    version of the line must not appear at all."""
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome

    # repo_root is the PR HEAD: it contains the PR's added files.
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "y.sql").write_text("select 1 as id\n")
    (tmp_path / "models" / "x.yml").write_text(
        "version: 2\nmodels:\n  - name: y\n    columns:\n      - name: team_uid\n"
    )
    pr_diff = (
        "diff --git a/models/y.sql b/models/y.sql\n"
        "--- /dev/null\n+++ b/models/y.sql\n@@ -0,0 +1 @@\n+select 1 as id\n"
        "diff --git a/models/x.yml b/models/x.yml\n"
        "--- /dev/null\n+++ b/models/x.yml\n@@ -0,0 +1,5 @@\n"
        "+version: 2\n+models:\n+  - name: y\n+    columns:\n+      - name: team_uid\n"
    )
    # the fix rewrites the PR-added phantom column
    candidate = (
        "diff --git a/models/x.yml b/models/x.yml\n"
        "--- a/models/x.yml\n+++ b/models/x.yml\n"
        "@@ -1,5 +1,5 @@\n version: 2\n models:\n   - name: y\n     columns:\n"
        "-      - name: team_uid\n+      - name: id\n"
    )
    seen = {}

    def fake_runner(args, env, cwd, timeout_seconds):
        seen["diff"] = env["DBT_AUDITOR_PR_DIFF"]
        return ProcessOutcome(
            returncode=0,
            stdout="dbt-auditor verdict: PASSED - ok\ndbt-auditor-audit-status: completed\n",
            stderr="",
        )

    run_reaudit_gate(
        repo_root=tmp_path, candidate_diff=candidate, pr_diff=pr_diff,
        pr_title="t", pr_description="", pr_url="u",
        auditor_python="/fake/python", failure_kind="audit",
        originally_failing_check_ids=(), timeout_seconds=60.0,
        subprocess_runner=fake_runner,
    )
    diff = seen["diff"]
    assert "models/y.sql" in diff              # the PR's own added file
    assert "+      - name: id" in diff         # the FIXED line, as a clean addition
    assert "team_uid" not in diff              # the broken version never appears
    assert diff.count("diff --git a/models/x.yml") == 1  # one clean block per file


def test_invert_diff_round_trips(tmp_path):
    from dbt_fixer.diffparse import apply_diff, invert_diff

    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.sql").write_text("select 1\n")
    diff = (
        "diff --git a/m/a.sql b/m/a.sql\n--- a/m/a.sql\n+++ b/m/a.sql\n"
        "@@ -1 +1 @@\n-select 1\n+select 2\n"
        "diff --git a/m/new.yml b/m/new.yml\n--- /dev/null\n+++ b/m/new.yml\n"
        "@@ -0,0 +1 @@\n+version: 2\n"
    )
    apply_diff(tmp_path, diff)
    assert (tmp_path / "m" / "a.sql").read_text() == "select 2\n"
    assert (tmp_path / "m" / "new.yml").exists()
    apply_diff(tmp_path, invert_diff(diff))
    assert (tmp_path / "m" / "a.sql").read_text() == "select 1\n"
    assert not (tmp_path / "m" / "new.yml").exists()


def test_combine_diffs_handles_empty_sides():
    from dbt_fixer.reaudit import combine_diffs

    assert combine_diffs("", "") == ""
    assert combine_diffs("a\n", "") == "a\n"
    assert combine_diffs("", "b\n") == "b\n"
    assert combine_diffs("a\n\n", "b") == "a\nb\n"


def test_reaudit_retries_a_completion_artifact_then_succeeds(tmp_path):
    """A 'status=failed' artifact is a failure to judge, not a rejection:
    the gate retries and accepts a subsequent COMPLETED, non-blocked run.
    (Live finding: bi-dbt #2533 hit a transient re-audit artifact on an
    otherwise-sound fix.)"""
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yml").write_text("version: 2\n")
    candidate = (
        "diff --git a/models/x.yml b/models/x.yml\n"
        "--- a/models/x.yml\n+++ b/models/x.yml\n@@ -1 +1,2 @@\n version: 2\n+# fix\n"
    )
    calls = {"n": 0}

    def flaky_runner(args, env, cwd, timeout_seconds):
        calls["n"] += 1
        if calls["n"] == 1:  # transient artifact
            return ProcessOutcome(
                returncode=0,
                stdout="dbt-auditor-audit-status: failed\n",
                stderr="",
            )
        return ProcessOutcome(  # completes cleanly on retry
            returncode=0,
            stdout="dbt-auditor verdict: PASSED - ok\ndbt-auditor-audit-status: completed\n",
            stderr="",
        )

    verdict = run_reaudit_gate(
        repo_root=tmp_path, candidate_diff=candidate, pr_diff="",
        pr_title="t", pr_description="", pr_url="u",
        auditor_python="/fake/python", failure_kind="audit",
        originally_failing_check_ids=(), timeout_seconds=60.0,
        subprocess_runner=flaky_runner,
    )
    assert verdict.passed, verdict.reason
    assert calls["n"] == 2  # retried once, then accepted


def test_reaudit_does_not_retry_a_genuine_blocked_verdict(tmp_path):
    """A BLOCKED verdict is a real judgment - it must NOT be retried."""
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yml").write_text("version: 2\n")
    candidate = (
        "diff --git a/models/x.yml b/models/x.yml\n"
        "--- a/models/x.yml\n+++ b/models/x.yml\n@@ -1 +1,2 @@\n version: 2\n+# fix\n"
    )
    calls = {"n": 0}

    def blocking_runner(args, env, cwd, timeout_seconds):
        calls["n"] += 1
        return ProcessOutcome(
            returncode=0,
            stdout="dbt-auditor verdict: BLOCKED - nope\ndbt-auditor-audit-status: completed\n",
            stderr="",
        )

    verdict = run_reaudit_gate(
        repo_root=tmp_path, candidate_diff=candidate, pr_diff="",
        pr_title="t", pr_description="", pr_url="u",
        auditor_python="/fake/python", failure_kind="audit",
        originally_failing_check_ids=(), timeout_seconds=60.0,
        subprocess_runner=blocking_runner,
    )
    assert not verdict.passed
    assert calls["n"] == 1  # BLOCKED is a judgment, not retried


def test_reaudit_gives_up_after_max_artifact_attempts(tmp_path):
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome, _MAX_REAUDIT_ARTIFACT_ATTEMPTS

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yml").write_text("version: 2\n")
    candidate = (
        "diff --git a/models/x.yml b/models/x.yml\n"
        "--- a/models/x.yml\n+++ b/models/x.yml\n@@ -1 +1,2 @@\n version: 2\n+# fix\n"
    )
    calls = {"n": 0}

    def always_artifact(args, env, cwd, timeout_seconds):
        calls["n"] += 1
        return ProcessOutcome(returncode=0, stdout="dbt-auditor-audit-status: failed\n", stderr="")

    verdict = run_reaudit_gate(
        repo_root=tmp_path, candidate_diff=candidate, pr_diff="",
        pr_title="t", pr_description="", pr_url="u",
        auditor_python="/fake/python", failure_kind="audit",
        originally_failing_check_ids=(), timeout_seconds=60.0,
        subprocess_runner=always_artifact,
    )
    assert not verdict.passed
    assert verdict.violation == "auditor_output_unparsable"
    assert calls["n"] == _MAX_REAUDIT_ARTIFACT_ATTEMPTS


def test_reaudit_does_not_retry_a_timeout(tmp_path):
    """A timeout already consumed the full budget; retrying would just eat it
    again (the wall-clock spiral). It is terminal, not retried."""
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yml").write_text("version: 2\n")
    candidate = (
        "diff --git a/models/x.yml b/models/x.yml\n"
        "--- a/models/x.yml\n+++ b/models/x.yml\n@@ -1 +1,2 @@\n version: 2\n+# fix\n"
    )
    calls = {"n": 0}

    def timeout_runner(args, env, cwd, timeout_seconds):
        calls["n"] += 1
        return ProcessOutcome(returncode=-1, stdout="", stderr="the sealed auditor timed out")

    verdict = run_reaudit_gate(
        repo_root=tmp_path, candidate_diff=candidate, pr_diff="",
        pr_title="t", pr_description="", pr_url="u",
        auditor_python="/fake/python", failure_kind="audit",
        originally_failing_check_ids=(), timeout_seconds=60.0,
        subprocess_runner=timeout_runner,
    )
    assert not verdict.passed
    assert calls["n"] == 1  # NOT retried


def test_reaudit_passes_when_fix_fully_reverts_the_pr(tmp_path):
    """A fix that restores exactly what the PR removed yields an empty
    effective diff (PR + fix == base). The gate must PASS without invoking
    the auditor (auditing an empty diff would fail-closed as an artifact).
    Found via the downstream-restore coverage test."""
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome

    (tmp_path / "models").mkdir()
    # base/repo_root is the PR HEAD: the PR removed a line ('b')
    (tmp_path / "models" / "m.sql").write_text("a\nc\n")
    pr_diff = (
        "diff --git a/models/m.sql b/models/m.sql\n"
        "--- a/models/m.sql\n+++ b/models/m.sql\n@@ -1,3 +1,2 @@\n a\n-b\n c\n"
    )
    # candidate restores 'b' -> PR+fix == base -> empty effective diff
    candidate = (
        "diff --git a/models/m.sql b/models/m.sql\n"
        "--- a/models/m.sql\n+++ b/models/m.sql\n@@ -1,2 +1,3 @@\n a\n+b\n c\n"
    )
    called = {"n": 0}

    def runner(args, env, cwd, timeout_seconds):
        called["n"] += 1
        return ProcessOutcome(returncode=0, stdout="dbt-auditor-audit-status: failed\n", stderr="")

    verdict = run_reaudit_gate(
        repo_root=tmp_path, candidate_diff=candidate, pr_diff=pr_diff,
        pr_title="t", pr_description="", pr_url="u",
        auditor_python="/fake/python", failure_kind="audit",
        originally_failing_check_ids=("downstream_dependency_impact",),
        timeout_seconds=60.0, subprocess_runner=runner,
    )
    assert verdict.passed, verdict.reason
    assert "reverts" in verdict.reason
    assert called["n"] == 0  # auditor never invoked on an empty effective diff


# --- red-team A: UNCONFIRMED originally-failing check must NOT count as pass ---
def test_reaudit_rejects_unconfirmed_originally_failing_check(tmp_path):
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yml").write_text("version: 2\n")
    candidate = (
        "diff --git a/models/x.yml b/models/x.yml\n--- a/models/x.yml\n+++ b/models/x.yml\n"
        "@@ -1 +1,2 @@\n version: 2\n+# fix\n"
    )
    def runner(args, env, cwd, timeout_seconds):
        rp = env["DBT_AUDITOR_REPORT_PATH"]
        with open(rp, "w", encoding="utf-8") as h:
            h.write(
                "# Verdict: **NEEDS_REVIEW**\n\n"
                "### Schema Contract Verification (`schema_contract_verification`)\n\n"
                "**Severity:** Critical &nbsp; **State:** **UNCONFIRMED**\n"
            )
        return ProcessOutcome(returncode=0,
            stdout="dbt-auditor verdict: NEEDS_REVIEW - x\ndbt-auditor-audit-status: completed\n", stderr="")
    v = run_reaudit_gate(repo_root=tmp_path, candidate_diff=candidate, pr_diff="",
        pr_title="t", pr_description="", pr_url="u", auditor_python="/fake/py",
        failure_kind="audit", originally_failing_check_ids=("schema_contract_verification",),
        timeout_seconds=60.0, subprocess_runner=runner)
    assert not v.passed
    assert v.violation == "auditor_check_still_failing"


# --- red-team B: a NEW non-advisory check failing must reject even if verdict != BLOCKED ---
def test_reaudit_rejects_newly_regressed_critical_check(tmp_path):
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yml").write_text("version: 2\n")
    candidate = (
        "diff --git a/models/x.yml b/models/x.yml\n--- a/models/x.yml\n+++ b/models/x.yml\n"
        "@@ -1 +1,2 @@\n version: 2\n+# fix\n"
    )
    def runner(args, env, cwd, timeout_seconds):
        rp = env["DBT_AUDITOR_REPORT_PATH"]
        with open(rp, "w", encoding="utf-8") as h:
            h.write(
                "# Verdict: **NEEDS_REVIEW**\n\n"
                "### Schema Contract Verification (`schema_contract_verification`)\n\n"
                "**Severity:** Critical &nbsp; **State:** **PASS**\n\n"
                "### Tenant Isolation Integrity (`tenant_isolation_integrity`)\n\n"
                "**Severity:** Critical &nbsp; **State:** **FAIL**\n"
            )
        return ProcessOutcome(returncode=0,
            stdout="dbt-auditor verdict: NEEDS_REVIEW - x\ndbt-auditor-audit-status: completed\n", stderr="")
    v = run_reaudit_gate(repo_root=tmp_path, candidate_diff=candidate, pr_diff="",
        pr_title="t", pr_description="", pr_url="u", auditor_python="/fake/py",
        failure_kind="audit", originally_failing_check_ids=("schema_contract_verification",),
        timeout_seconds=60.0, subprocess_runner=runner)
    assert not v.passed
    assert v.violation == "auditor_check_regressed"
    assert "tenant_isolation_integrity" in v.reason


def test_reaudit_advisory_failing_still_passes(tmp_path):
    """An advisory check failing (sql_style) must NOT block the gate."""
    from dbt_fixer.reaudit import run_reaudit_gate, ProcessOutcome
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yml").write_text("version: 2\n")
    candidate = (
        "diff --git a/models/x.yml b/models/x.yml\n--- a/models/x.yml\n+++ b/models/x.yml\n"
        "@@ -1 +1,2 @@\n version: 2\n+# fix\n"
    )
    def runner(args, env, cwd, timeout_seconds):
        rp = env["DBT_AUDITOR_REPORT_PATH"]
        with open(rp, "w", encoding="utf-8") as h:
            h.write(
                "# Verdict: **NEEDS_REVIEW**\n\n"
                "### Schema Contract Verification (`schema_contract_verification`)\n\n"
                "**Severity:** Critical &nbsp; **State:** **PASS**\n\n"
                "### SQL Style and Testability (`sql_style_and_testability`)\n\n"
                "**Severity:** Advisory &nbsp; **State:** **FAIL**\n"
            )
        return ProcessOutcome(returncode=0,
            stdout="dbt-auditor verdict: NEEDS_REVIEW - x\ndbt-auditor-audit-status: completed\n", stderr="")
    v = run_reaudit_gate(repo_root=tmp_path, candidate_diff=candidate, pr_diff="",
        pr_title="t", pr_description="", pr_url="u", auditor_python="/fake/py",
        failure_kind="audit", originally_failing_check_ids=("schema_contract_verification",),
        timeout_seconds=60.0, subprocess_runner=runner)
    assert v.passed, v.reason
