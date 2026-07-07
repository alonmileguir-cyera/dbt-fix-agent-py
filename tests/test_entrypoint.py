"""Tests for `dbt_fixer.entrypoint`: the always-exit-0, single-status-line
CLI contract.

These tests call `main()`/`compute_run_result()` in-process (never spawning
a real subprocess -- that would be blocked by `conftest` anyway outside a
`real_process` module) and assert on captured stdout, matching exactly how
a future orchestrator would grep this process's output.
"""

from __future__ import annotations

import re

import pytest

import dbt_fixer.pipeline as pipeline_module
from dbt_fixer.entrypoint import compute_run_result, main, render_stdout_lines
from dbt_fixer.env import ENV_FAILURE_KIND, ENV_FAILURE_CONTEXT, ENV_REPO_PATH
from dbt_fixer.slack_delivery import SlackDeliveryResult
from dbt_fixer.status import (
    RunResult,
    STDOUT_PATCH_BEGIN,
    STDOUT_PATCH_END,
    STDOUT_REASON_PREFIX,
    STDOUT_STATUS_PREFIX,
)

_STATUS_LINE_RE = re.compile(r"^dbt-fixer-status: (proposed|no_safe_fix|failed)$")


def _lines(capsys) -> list[str]:
    out = capsys.readouterr().out
    return out.splitlines()


def _stub_deliver_shadow_report(**kwargs) -> SlackDeliveryResult:
    """A no-op fake `deliver_slack`: every entrypoint test that reaches
    Stage 2 must inject this (or an equivalent fake) so the offline suite
    never attempts a real Slack API call."""

    return SlackDeliveryResult(
        skipped=True,
        summary_posted=False,
        summary_ts=None,
        detail_chunks_posted=0,
        detail_chunks_total=0,
        reason="test stub: Slack delivery not exercised by this test",
    )


# --- the fixed stdout contract ----------------------------------------------


def test_empty_environment_resolves_to_failed_with_single_status_line(capsys):
    exit_code = main(env={})
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines, "expected at least the status line"
    assert _STATUS_LINE_RE.match(lines[-1])
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    # exactly one status line in the whole run
    assert sum(1 for line in lines if _STATUS_LINE_RE.match(line)) == 1


def test_missing_repo_path_resolves_to_failed_with_named_reason(capsys):
    exit_code = main(env={ENV_FAILURE_KIND: "ci"})
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    reason_lines = [line for line in lines if line.startswith(STDOUT_REASON_PREFIX)]
    assert reason_lines, "a failed run must state a specific reason"
    assert ENV_REPO_PATH in reason_lines[0]


def test_malformed_failure_kind_resolves_to_failed(tmp_path, capsys):
    env = {ENV_FAILURE_KIND: "not-a-real-kind", ENV_REPO_PATH: str(tmp_path)}
    exit_code = main(env=env)
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    reason_lines = [line for line in lines if line.startswith(STDOUT_REASON_PREFIX)]
    assert ENV_FAILURE_KIND in reason_lines[0]


def test_empty_failure_context_resolves_to_no_safe_fix(tmp_path, capsys):
    env = {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path)}
    exit_code = main(env=env)
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: no_safe_fix"


def test_unparseable_failure_context_resolves_to_no_safe_fix(tmp_path, capsys):
    env = {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: "totally unrelated garbage text",
    }
    exit_code = main(env=env)
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: no_safe_fix"


def test_valid_ci_target_is_wired_into_the_bounded_fix_attempt(tmp_path, monkeypatch, capsys):
    """Sprint 4: a cleanly-parsed target is now wired all the way into the
    bounded fix attempt (`dbt_fixer.retry_loop.run_bounded_fix_attempt`),
    where `proposed` was architecturally unreachable in Sprint 1. This
    injects a fake `run_fix_attempt` (never the real, network-touching
    production pipeline) and asserts the identified target actually
    reaches it, and that its resulting reason is what stdout reports."""

    import dbt_fixer.entrypoint as entrypoint_module

    captured: dict = {}

    def fake_run_fix_attempt(config, target, fenced_context, repo_root):
        captured["target"] = target
        captured["repo_root"] = repo_root
        return entrypoint_module.FixAttemptResult(
            run_result=RunResult(
                status="no_safe_fix",
                reason=f"no candidate passed every gate for {target.identifiers[0]}",
            ),
        )

    monkeypatch.setattr(entrypoint_module, "default_run_fix_attempt", fake_run_fix_attempt)
    monkeypatch.setattr(
        entrypoint_module, "deliver_shadow_report", _stub_deliver_shadow_report
    )

    env = {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: (
            "Completed with 1 error\n"
            "Failure in test my_test (models/x.sql)\n"
            "Got 1 results, configured to fail if != 0\n"
        ),
    }
    exit_code = main(env=env)
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: no_safe_fix"
    reason_lines = [line for line in lines if line.startswith(STDOUT_REASON_PREFIX)]
    assert "my_test" in reason_lines[0]
    assert "my_test" in captured["target"].identifiers[0]
    assert captured["repo_root"] == str(tmp_path)


def test_status_line_is_always_the_last_line_and_appears_exactly_once(tmp_path, capsys):
    scenarios = [
        {},
        {ENV_FAILURE_KIND: "ci"},
        {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path)},
        {
            ENV_FAILURE_KIND: "ci",
            ENV_REPO_PATH: str(tmp_path),
            ENV_FAILURE_CONTEXT: "garbage",
        },
    ]
    for env in scenarios:
        exit_code = main(env=env)
        assert exit_code == 0
        lines = _lines(capsys)
        assert lines, f"expected output for env={env!r}"
        assert _STATUS_LINE_RE.match(lines[-1]), f"bad last line for env={env!r}: {lines[-1]!r}"
        assert sum(1 for line in lines if _STATUS_LINE_RE.match(line)) == 1


# --- exit code is always 0, even on internal exceptions ---------------------


def test_exit_code_is_zero_even_when_run_stage1_raises(monkeypatch, capsys):
    def _boom(env=None):
        raise RuntimeError("simulated internal defect")

    monkeypatch.setattr(pipeline_module, "run_stage1", _boom)
    # entrypoint imports run_stage1 by name, so patch it there too.
    import dbt_fixer.entrypoint as entrypoint_module

    monkeypatch.setattr(entrypoint_module, "run_stage1", _boom)

    exit_code = main(env={})
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    reason_lines = [line for line in lines if line.startswith(STDOUT_REASON_PREFIX)]
    assert reason_lines and "simulated internal defect" in reason_lines[0]


def test_exit_code_is_zero_even_when_compute_run_result_itself_raises(monkeypatch, capsys):
    import dbt_fixer.entrypoint as entrypoint_module

    def _boom(env=None):
        raise RuntimeError("even more unexpected")

    monkeypatch.setattr(entrypoint_module, "compute_run_result", _boom)

    exit_code = main(env={})
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"


def test_no_unhandled_traceback_reaches_stdout(tmp_path, capsys):
    exit_code = main(env={ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path / "missing")})
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Traceback (most recent call last)" not in out


# --- render_stdout_lines / compute_run_result as pure functions ------------


def test_render_stdout_lines_puts_status_line_last():
    result = RunResult(status="failed", reason="line one\nline two")
    lines = render_stdout_lines(result)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    # multi-line reasons are collapsed so they can never be mistaken for
    # (or push past) the true last line
    assert "\n" not in lines[0]
    assert lines[0] == f"{STDOUT_REASON_PREFIX}: line one line two"


def test_render_stdout_lines_omits_reason_line_when_reason_is_empty():
    result = RunResult(status="no_safe_fix", reason="")
    lines = render_stdout_lines(result)
    assert lines == [f"{STDOUT_STATUS_PREFIX}: no_safe_fix"]


@pytest.mark.parametrize(
    "env",
    [
        {},
        {ENV_FAILURE_KIND: "audit"},
    ],
)
def test_compute_run_result_never_raises_for_bad_input(env):
    result = compute_run_result(env)
    assert result.status in ("failed", "no_safe_fix", "proposed")


# --- Sprint 4: the bounded fix attempt is wired end-to-end, via fakes ------
#
# Every test below reaches Stage 2 (a cleanly-parsed target), so every one
# injects a fake `run_fix_attempt` -- never the real, network-touching
# `default_run_fix_attempt` -- and a fake `deliver_slack`, matching
# `no_network_or_real_subprocess_in_tests`.


def _valid_env(tmp_path) -> dict:
    return {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: (
            "Completed with 1 error\n"
            "Failure in test my_test (models/x.sql)\n"
            "Got 1 results, configured to fail if != 0\n"
        ),
    }


def _install_fake_pipeline(monkeypatch, *, run_fix_attempt, deliver_slack=None):
    import dbt_fixer.entrypoint as entrypoint_module

    monkeypatch.setattr(entrypoint_module, "default_run_fix_attempt", run_fix_attempt)
    monkeypatch.setattr(
        entrypoint_module,
        "deliver_shadow_report",
        deliver_slack or _stub_deliver_shadow_report,
    )
    return entrypoint_module


_FAKE_DIFF = "--- a/models/x.sql\n+++ b/models/x.sql\n-select 1\n+select 1 where y = 1\n"


def test_proposed_status_emits_exactly_one_patch_block_bracketing_the_diff(tmp_path, monkeypatch, capsys):
    import dbt_fixer.entrypoint as entrypoint_module

    def fake_run_fix_attempt(config, target, fenced_context, repo_root):
        return entrypoint_module.FixAttemptResult(
            run_result=RunResult(status="proposed", reason="round 1 passed every gate"),
            diff=_FAKE_DIFF,
            rounds_used=1,
        )

    _install_fake_pipeline(monkeypatch, run_fix_attempt=fake_run_fix_attempt)

    exit_code = main(env=_valid_env(tmp_path))
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: proposed"

    begin_indices = [i for i, line in enumerate(lines) if line == STDOUT_PATCH_BEGIN]
    end_indices = [i for i, line in enumerate(lines) if line == STDOUT_PATCH_END]
    assert len(begin_indices) == 1
    assert len(end_indices) == 1
    assert begin_indices[0] < end_indices[0] < len(lines) - 1

    patch_body = "\n".join(lines[begin_indices[0] + 1 : end_indices[0]])
    assert patch_body == _FAKE_DIFF.rstrip("\n")


def test_no_patch_block_anchors_when_status_is_no_safe_fix(tmp_path, monkeypatch, capsys):
    import dbt_fixer.entrypoint as entrypoint_module

    def fake_run_fix_attempt(config, target, fenced_context, repo_root):
        return entrypoint_module.FixAttemptResult(
            run_result=RunResult(status="no_safe_fix", reason="no candidate passed every gate"),
        )

    _install_fake_pipeline(monkeypatch, run_fix_attempt=fake_run_fix_attempt)

    exit_code = main(env=_valid_env(tmp_path))
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: no_safe_fix"
    assert STDOUT_PATCH_BEGIN not in lines
    assert STDOUT_PATCH_END not in lines


def test_no_patch_block_anchors_when_status_is_failed_even_with_a_diff_present(tmp_path, monkeypatch, capsys):
    """Defensive: a `diff` on the outcome must never leak a patch block
    unless `status == "proposed"`, even if some future bug left a stray
    diff attached to a `failed`/`no_safe_fix` result."""

    import dbt_fixer.entrypoint as entrypoint_module

    def fake_run_fix_attempt(config, target, fenced_context, repo_root):
        return entrypoint_module.FixAttemptResult(
            run_result=RunResult(status="failed", reason="unexpected internal error"),
            diff=_FAKE_DIFF,
        )

    _install_fake_pipeline(monkeypatch, run_fix_attempt=fake_run_fix_attempt)

    exit_code = main(env=_valid_env(tmp_path))
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: failed"
    assert STDOUT_PATCH_BEGIN not in lines
    assert STDOUT_PATCH_END not in lines


def test_secret_like_token_is_redacted_identically_in_stdout_reason_and_diff(tmp_path, monkeypatch, capsys):
    """redaction_applied_to_slack_and_stdout (stdout half): a secret-shaped
    token present in either the reason or the diff must never appear raw
    in stdout."""

    fake_aws_key = "AKIA" + "ABCDEFGHIJKLMNOP"
    tainted_diff = (
        "--- a/models/x.sql\n+++ b/models/x.sql\n"
        f"-select 1 -- key {fake_aws_key}\n+select 1 where y = 1\n"
    )
    tainted_reason = f"round 1 passed every gate (rotate {fake_aws_key} before merging)"

    import dbt_fixer.entrypoint as entrypoint_module

    def fake_run_fix_attempt(config, target, fenced_context, repo_root):
        return entrypoint_module.FixAttemptResult(
            run_result=RunResult(status="proposed", reason=tainted_reason),
            diff=tainted_diff,
            rounds_used=1,
        )

    _install_fake_pipeline(monkeypatch, run_fix_attempt=fake_run_fix_attempt)

    exit_code = main(env=_valid_env(tmp_path))
    assert exit_code == 0

    out = capsys.readouterr().out
    assert fake_aws_key not in out


def test_slack_delivery_raising_never_changes_the_reported_status(tmp_path, monkeypatch, capsys):
    """A `deliver_slack` that raises must never affect the already-computed
    terminal status or crash the process."""

    import dbt_fixer.entrypoint as entrypoint_module

    def fake_run_fix_attempt(config, target, fenced_context, repo_root):
        return entrypoint_module.FixAttemptResult(
            run_result=RunResult(status="proposed", reason="round 1 passed every gate"),
            diff=_FAKE_DIFF,
            rounds_used=1,
        )

    def _boom_deliver_slack(**kwargs):
        raise RuntimeError("simulated Slack outage")

    _install_fake_pipeline(
        monkeypatch, run_fix_attempt=fake_run_fix_attempt, deliver_slack=_boom_deliver_slack
    )

    exit_code = main(env=_valid_env(tmp_path))
    assert exit_code == 0

    lines = _lines(capsys)
    assert lines[-1] == f"{STDOUT_STATUS_PREFIX}: proposed"
    assert STDOUT_PATCH_BEGIN in lines
    assert STDOUT_PATCH_END in lines


def test_status_line_contract_holds_for_the_newly_reachable_proposed_outcome(tmp_path, monkeypatch, capsys):
    """stdout_status_line_contract, extended to the now-reachable `proposed`
    outcome: exactly one status line, always last, for all three statuses."""

    import dbt_fixer.entrypoint as entrypoint_module

    for status, diff in (
        ("proposed", _FAKE_DIFF),
        ("no_safe_fix", None),
        ("failed", None),
    ):

        def fake_run_fix_attempt(config, target, fenced_context, repo_root, _status=status, _diff=diff):
            return entrypoint_module.FixAttemptResult(
                run_result=RunResult(status=_status, reason=f"resolved to {_status}"),
                diff=_diff,
            )

        _install_fake_pipeline(monkeypatch, run_fix_attempt=fake_run_fix_attempt)

        exit_code = main(env=_valid_env(tmp_path))
        assert exit_code == 0

        lines = _lines(capsys)
        assert lines, f"expected output for status={status!r}"
        assert _STATUS_LINE_RE.match(lines[-1]), f"bad last line for status={status!r}: {lines[-1]!r}"
        assert sum(1 for line in lines if _STATUS_LINE_RE.match(line)) == 1
