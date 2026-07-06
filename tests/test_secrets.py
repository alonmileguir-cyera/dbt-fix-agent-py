"""Tests for Slack bot token retrieval (AWS Secrets Manager + env fallback)."""

from __future__ import annotations

from dbt_fixer.secrets import (
    SLACK_BOT_TOKEN_ENV_VAR,
    SLACK_TOKEN_SECRET_ID,
    get_slack_bot_token,
)


class _FakeSecretsManagerClient:
    def __init__(self, secret_string: str | None = None, error: Exception | None = None):
        self._secret_string = secret_string
        self._error = error
        self.calls: list[str] = []

    def get_secret_value(self, SecretId: str):
        self.calls.append(SecretId)
        if self._error is not None:
            raise self._error
        return {"SecretString": self._secret_string}


def test_returns_token_from_secrets_manager_when_available():
    client = _FakeSecretsManagerClient(secret_string="xoxb-secretsmanager-token")
    token = get_slack_bot_token(env={}, client_factory=lambda: client)
    assert token == "xoxb-secretsmanager-token"
    assert client.calls == [SLACK_TOKEN_SECRET_ID]


def test_falls_back_to_env_var_when_secrets_manager_raises():
    client = _FakeSecretsManagerClient(error=RuntimeError("access denied"))
    token = get_slack_bot_token(
        env={SLACK_BOT_TOKEN_ENV_VAR: "xoxb-env-fallback"},
        client_factory=lambda: client,
    )
    assert token == "xoxb-env-fallback"


def test_falls_back_to_env_var_when_secrets_manager_returns_no_string():
    client = _FakeSecretsManagerClient(secret_string=None)
    token = get_slack_bot_token(
        env={SLACK_BOT_TOKEN_ENV_VAR: "xoxb-env-fallback"},
        client_factory=lambda: client,
    )
    assert token == "xoxb-env-fallback"


def test_falls_back_to_env_var_when_client_factory_itself_raises():
    def _boom():
        raise RuntimeError("no AWS credentials available")

    token = get_slack_bot_token(
        env={SLACK_BOT_TOKEN_ENV_VAR: "xoxb-env-fallback"},
        client_factory=_boom,
    )
    assert token == "xoxb-env-fallback"


def test_returns_none_when_neither_source_has_a_token():
    client = _FakeSecretsManagerClient(secret_string=None)
    token = get_slack_bot_token(env={}, client_factory=lambda: client)
    assert token is None


def test_returns_none_when_env_var_is_blank():
    client = _FakeSecretsManagerClient(error=RuntimeError("boom"))
    token = get_slack_bot_token(env={SLACK_BOT_TOKEN_ENV_VAR: "   "}, client_factory=lambda: client)
    assert token is None


def test_never_raises_even_when_everything_fails():
    def _boom():
        raise RuntimeError("network unreachable")

    token = get_slack_bot_token(env={}, client_factory=_boom)
    assert token is None


def test_secrets_manager_result_preferred_over_env_var():
    client = _FakeSecretsManagerClient(secret_string="xoxb-from-sm")
    token = get_slack_bot_token(
        env={SLACK_BOT_TOKEN_ENV_VAR: "xoxb-from-env"}, client_factory=lambda: client
    )
    assert token == "xoxb-from-sm"


def test_parses_json_object_secret_and_extracts_token_key():
    # Cyera stores the token as {"slack-bot-token": "xoxb-..."}. The whole JSON
    # blob must NOT be sent as the token (that produced a live invalid_auth).
    import json as _json

    client = _FakeSecretsManagerClient(
        secret_string=_json.dumps({"slack-bot-token": "xoxb-real-token"})
    )
    token = get_slack_bot_token(env={}, client_factory=lambda: client)
    assert token == "xoxb-real-token"


def test_json_object_with_unknown_key_falls_back_to_env():
    import json as _json

    client = _FakeSecretsManagerClient(
        secret_string=_json.dumps({"some_other_field": "not-a-token", "and": "more"})
    )
    token = get_slack_bot_token(
        env={SLACK_BOT_TOKEN_ENV_VAR: "xoxb-env-fallback"},
        client_factory=lambda: client,
    )
    assert token == "xoxb-env-fallback"


def test_json_object_with_single_entry_uses_its_value():
    import json as _json

    client = _FakeSecretsManagerClient(
        secret_string=_json.dumps({"weirdly_named": "xoxb-single"})
    )
    token = get_slack_bot_token(env={}, client_factory=lambda: client)
    assert token == "xoxb-single"
