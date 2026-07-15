"""Tests for the `DBT_FIXER_*` environment contract owned by `dbt_fixer.env`."""

from __future__ import annotations

import pytest

from dbt_fixer.env import (
    DEFAULT_DBT_PARSE_MODE,
    DEFAULT_DBT_PARSE_TIMEOUT_SECONDS,
    DEFAULT_MAX_CHANGED_FILES,
    DEFAULT_MAX_CHANGED_LINES,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_REAUDIT_TIMEOUT_SECONDS,
    DEFAULT_REFUTER_TIMEOUT_SECONDS,
    ENV_AUDITOR_PYTHON,
    ENV_DBT_PARSE_MODE,
    ENV_DBT_PARSE_TIMEOUT_SECONDS,
    ENV_FAILURE_KIND,
    ENV_MAX_CHANGED_FILES,
    ENV_MAX_CHANGED_LINES,
    ENV_MAX_ROUNDS,
    ENV_PR_DESCRIPTION,
    ENV_PR_DIFF,
    ENV_PR_TITLE,
    ENV_PR_URL,
    ENV_REAUDIT_TIMEOUT_SECONDS,
    ENV_REFUTER_TIMEOUT_SECONDS,
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
    assert config.max_changed_files == DEFAULT_MAX_CHANGED_FILES
    assert config.max_changed_lines == DEFAULT_MAX_CHANGED_LINES
    assert config.reaudit_timeout_seconds == DEFAULT_REAUDIT_TIMEOUT_SECONDS
    assert config.refuter_timeout_seconds == DEFAULT_REFUTER_TIMEOUT_SECONDS
    assert config.dbt_parse_mode == DEFAULT_DBT_PARSE_MODE
    assert config.dbt_parse_timeout_seconds == DEFAULT_DBT_PARSE_TIMEOUT_SECONDS
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
            ENV_MAX_CHANGED_FILES: "10",
            ENV_MAX_CHANGED_LINES: "200",
            ENV_REAUDIT_TIMEOUT_SECONDS: "30",
            ENV_REFUTER_TIMEOUT_SECONDS: "45",
            ENV_DBT_PARSE_MODE: "enabled",
            ENV_DBT_PARSE_TIMEOUT_SECONDS: "15",
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
    assert config.max_changed_files == 10
    assert config.max_changed_lines == 200
    assert config.reaudit_timeout_seconds == 30
    assert config.refuter_timeout_seconds == 45
    assert config.dbt_parse_mode == "enabled"
    assert config.dbt_parse_timeout_seconds == 15


@pytest.mark.parametrize("value", ["enabled", "ENABLED", " enabled "])
def test_dbt_parse_requires_explicit_recognized_opt_in(tmp_path, value):
    config = load_config(_base_env(tmp_path, **{ENV_DBT_PARSE_MODE: value}))
    assert config.dbt_parse_mode == "enabled"
    assert config.warnings == ()


@pytest.mark.parametrize("value", ["true", "on", "1", "unexpected"])
def test_unrecognized_dbt_parse_mode_fails_safe_to_disabled(tmp_path, value):
    config = load_config(_base_env(tmp_path, **{ENV_DBT_PARSE_MODE: value}))
    assert config.dbt_parse_mode == "disabled"
    assert any(ENV_DBT_PARSE_MODE in warning for warning in config.warnings)


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


@pytest.mark.parametrize("bad_value", ["not-a-number", "-1", "0", "999", "3.5"])
def test_malformed_max_changed_files_falls_back_to_default(tmp_path, bad_value):
    env = _base_env(tmp_path, **{ENV_MAX_CHANGED_FILES: bad_value})
    config = load_config(env)
    assert config.max_changed_files == DEFAULT_MAX_CHANGED_FILES
    assert any(ENV_MAX_CHANGED_FILES in w for w in config.warnings)


@pytest.mark.parametrize("good_value,expected", [("1", 1), ("50", 50), ("5", 5)])
def test_valid_max_changed_files_at_and_within_boundaries_is_respected(tmp_path, good_value, expected):
    env = _base_env(tmp_path, **{ENV_MAX_CHANGED_FILES: good_value})
    config = load_config(env)
    assert config.max_changed_files == expected
    assert config.warnings == ()


@pytest.mark.parametrize("bad_value", ["not-a-number", "-1", "0", "5000", "3.5"])
def test_malformed_max_changed_lines_falls_back_to_default(tmp_path, bad_value):
    env = _base_env(tmp_path, **{ENV_MAX_CHANGED_LINES: bad_value})
    config = load_config(env)
    assert config.max_changed_lines == DEFAULT_MAX_CHANGED_LINES
    assert any(ENV_MAX_CHANGED_LINES in w for w in config.warnings)


@pytest.mark.parametrize("good_value,expected", [("1", 1), ("2000", 2000), ("60", 60)])
def test_valid_max_changed_lines_at_and_within_boundaries_is_respected(tmp_path, good_value, expected):
    env = _base_env(tmp_path, **{ENV_MAX_CHANGED_LINES: good_value})
    config = load_config(env)
    assert config.max_changed_lines == expected
    assert config.warnings == ()


@pytest.mark.parametrize("bad_value", ["not-a-number", "-1", "0", "9999", "nan", "inf"])
def test_malformed_reaudit_timeout_seconds_falls_back_to_default(tmp_path, bad_value):
    env = _base_env(tmp_path, **{ENV_REAUDIT_TIMEOUT_SECONDS: bad_value})
    config = load_config(env)
    assert config.reaudit_timeout_seconds == DEFAULT_REAUDIT_TIMEOUT_SECONDS
    assert any(ENV_REAUDIT_TIMEOUT_SECONDS in w for w in config.warnings)


@pytest.mark.parametrize("good_value,expected", [("1", 1), ("1800", 1800), ("120", 120)])
def test_valid_reaudit_timeout_seconds_at_and_within_boundaries_is_respected(
    tmp_path, good_value, expected
):
    env = _base_env(tmp_path, **{ENV_REAUDIT_TIMEOUT_SECONDS: good_value})
    config = load_config(env)
    assert config.reaudit_timeout_seconds == expected
    assert config.warnings == ()


@pytest.mark.parametrize("bad_value", ["not-a-number", "-1", "0", "9999", "nan", "inf"])
def test_malformed_refuter_timeout_seconds_falls_back_to_default(tmp_path, bad_value):
    env = _base_env(tmp_path, **{ENV_REFUTER_TIMEOUT_SECONDS: bad_value})
    config = load_config(env)
    assert config.refuter_timeout_seconds == DEFAULT_REFUTER_TIMEOUT_SECONDS
    assert any(ENV_REFUTER_TIMEOUT_SECONDS in w for w in config.warnings)


@pytest.mark.parametrize("good_value,expected", [("1", 1), ("600", 600), ("60", 60)])
def test_valid_refuter_timeout_seconds_at_and_within_boundaries_is_respected(
    tmp_path, good_value, expected
):
    env = _base_env(tmp_path, **{ENV_REFUTER_TIMEOUT_SECONDS: good_value})
    config = load_config(env)
    assert config.refuter_timeout_seconds == expected
    assert config.warnings == ()


@pytest.mark.parametrize("bad_value", ["not-a-number", "-1", "0", "9999", "nan", "inf"])
def test_malformed_dbt_parse_timeout_seconds_falls_back_to_default(tmp_path, bad_value):
    env = _base_env(tmp_path, **{ENV_DBT_PARSE_TIMEOUT_SECONDS: bad_value})
    config = load_config(env)
    assert config.dbt_parse_timeout_seconds == DEFAULT_DBT_PARSE_TIMEOUT_SECONDS
    assert any(ENV_DBT_PARSE_TIMEOUT_SECONDS in w for w in config.warnings)


@pytest.mark.parametrize("good_value,expected", [("1", 1), ("300", 300), ("30", 30)])
def test_valid_dbt_parse_timeout_seconds_at_and_within_boundaries_is_respected(
    tmp_path, good_value, expected
):
    env = _base_env(tmp_path, **{ENV_DBT_PARSE_TIMEOUT_SECONDS: good_value})
    config = load_config(env)
    assert config.dbt_parse_timeout_seconds == expected
    assert config.warnings == ()


def test_load_config_defaults_to_os_environ(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_FAILURE_KIND, "audit")
    monkeypatch.setenv(ENV_REPO_PATH, str(tmp_path))
    config = load_config()
    assert config.failure_kind == "audit"
    assert config.repo_path == tmp_path
