"""Tests for the `DBT_FIXER_*` environment contract owned by `dbt_fixer.env`."""

from __future__ import annotations

import pytest

from dbt_fixer.env import (
    DEFAULT_MAX_ROUNDS,
    ENV_AUDITOR_PYTHON,
    ENV_FAILURE_KIND,
    ENV_MAX_ROUNDS,
    ENV_PR_DESCRIPTION,
    ENV_PR_DIFF,
    ENV_PR_TITLE,
    ENV_PR_URL,
    ENV_REPO_PATH,
    ENV_SLACK_CHANNEL,
    EnvValidationError,
    load_config,
)


def _base_env(tmp_path, **overrides):
    env = {ENV_FAILURE_KIND: "ci", ENV_REPO_PATH: str(tmp_path)}
    env.update(overrides)
    return env


def test_missing_failure_kind_names_the_variable(tmp_path):
    env = _base_env(tmp_path)
    del env[ENV_FAILURE_KIND]
    with pytest.raises(EnvValidationError, match=ENV_FAILURE_KIND):
        load_config(env)


def test_missing_repo_path_names_the_variable(tmp_path):
    env = _base_env(tmp_path)
    del env[ENV_REPO_PATH]
    with pytest.raises(EnvValidationError, match=ENV_REPO_PATH):
        load_config(env)


def test_blank_failure_kind_is_treated_as_missing(tmp_path):
    env = _base_env(tmp_path, **{ENV_FAILURE_KIND: "   "})
    with pytest.raises(EnvValidationError, match=ENV_FAILURE_KIND):
        load_config(env)


def test_invalid_failure_kind_value_raises(tmp_path):
    env = _base_env(tmp_path, **{ENV_FAILURE_KIND: "bogus"})
    with pytest.raises(EnvValidationError, match="bogus"):
        load_config(env)


def test_nonexistent_repo_path_raises(tmp_path):
    env = _base_env(tmp_path, **{ENV_REPO_PATH: str(tmp_path / "does-not-exist")})
    with pytest.raises(EnvValidationError, match=ENV_REPO_PATH):
        load_config(env)


def test_repo_path_pointing_at_a_file_raises(tmp_path):
    a_file = tmp_path / "not-a-dir.txt"
    a_file.write_text("hello")
    env = _base_env(tmp_path, **{ENV_REPO_PATH: str(a_file)})
    with pytest.raises(EnvValidationError, match=ENV_REPO_PATH):
        load_config(env)


def test_all_optional_fields_default_when_unset(tmp_path):
    config = load_config(_base_env(tmp_path))
    assert config.pr_title == ""
    assert config.pr_description == ""
    assert config.pr_diff == ""
    assert config.pr_url == ""
    assert config.failure_context == ""
    assert config.slack_channel is None
    assert config.auditor_python is None
    assert config.max_rounds == DEFAULT_MAX_ROUNDS
    assert config.warnings == ()


def test_optional_fields_are_respected_when_set(tmp_path):
    env = _base_env(
        tmp_path,
        **{
            ENV_PR_TITLE: "Fix broken not_null test",
            ENV_PR_DESCRIPTION: "restores a deleted line",
            ENV_PR_DIFF: "--- a/x\n+++ b/x\n",
            ENV_PR_URL: "https://github.com/example/repo/pull/1",
            ENV_SLACK_CHANNEL: "#data-eng-ci",
            ENV_AUDITOR_PYTHON: "/usr/bin/python3.11",
            ENV_MAX_ROUNDS: "2",
        },
    )
    config = load_config(env)
    assert config.pr_title == "Fix broken not_null test"
    assert config.pr_description == "restores a deleted line"
    assert config.pr_diff.startswith("--- a/x")
    assert config.pr_url.endswith("/pull/1")
    assert config.slack_channel == "#data-eng-ci"
    assert config.auditor_python == "/usr/bin/python3.11"
    assert config.max_rounds == 2


@pytest.mark.parametrize("bad_value", ["not-a-number", "-1", "0", "999", "3.5"])
def test_malformed_max_rounds_falls_back_to_default(tmp_path, bad_value):
    env = _base_env(tmp_path, **{ENV_MAX_ROUNDS: bad_value})
    config = load_config(env)
    assert config.max_rounds == DEFAULT_MAX_ROUNDS
    assert config.warnings
    assert any(ENV_MAX_ROUNDS in w for w in config.warnings)


@pytest.mark.parametrize("good_value,expected", [("1", 1), ("10", 10), ("5", 5)])
def test_valid_max_rounds_at_and_within_boundaries_is_respected(tmp_path, good_value, expected):
    env = _base_env(tmp_path, **{ENV_MAX_ROUNDS: good_value})
    config = load_config(env)
    assert config.max_rounds == expected
    assert config.warnings == ()


def test_unset_max_rounds_produces_no_warning(tmp_path):
    config = load_config(_base_env(tmp_path))
    assert config.warnings == ()


def test_load_config_defaults_to_os_environ(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_FAILURE_KIND, "audit")
    monkeypatch.setenv(ENV_REPO_PATH, str(tmp_path))
    config = load_config()
    assert config.failure_kind == "audit"
    assert config.repo_path == tmp_path
