"""Tests for `dbt_fixer.pipeline.run_stage1`: env + intake errors always map
to a clean, typed terminal `RunResult`, never an unhandled exception."""

from __future__ import annotations

from dbt_fixer.env import ENV_FAILURE_KIND, ENV_FAILURE_CONTEXT, ENV_REPO_PATH
from dbt_fixer.pipeline import run_stage1


def test_missing_required_env_resolves_to_failed():
    outcome = run_stage1({})
    assert outcome.terminal is not None
    assert outcome.terminal.status == "failed"
    assert outcome.config is None
    assert outcome.intake is None


def test_invalid_repo_path_resolves_to_failed(tmp_path):
    env = {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path / "nope")}
    outcome = run_stage1(env)
    assert outcome.terminal is not None
    assert outcome.terminal.status == "failed"


def test_unparseable_context_resolves_to_no_safe_fix(tmp_path):
    env = {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: "garbage unrelated text",
    }
    outcome = run_stage1(env)
    assert outcome.terminal is not None
    assert outcome.terminal.status == "no_safe_fix"
    assert outcome.terminal.reason
    assert outcome.config is not None  # env validation succeeded before intake ran


def test_empty_context_resolves_to_no_safe_fix(tmp_path):
    env = {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path)}
    outcome = run_stage1(env)
    assert outcome.terminal is not None
    assert outcome.terminal.status == "no_safe_fix"


def test_valid_target_does_not_resolve_terminal_yet(tmp_path):
    env = {
        ENV_FAILURE_KIND: "ci",
        ENV_REPO_PATH: str(tmp_path),
        ENV_FAILURE_CONTEXT: (
            "Completed with 1 error\n\n"
            "Failure in test x (models/y.sql)\n  bad\n\nDone."
        ),
    }
    outcome = run_stage1(env)
    assert outcome.terminal is None
    assert outcome.config is not None
    assert outcome.intake is not None
    assert outcome.intake.ok
    assert outcome.intake.target.identifiers == ("x",)
