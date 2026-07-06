"""Tests for secret-like pattern redaction."""

from __future__ import annotations

from dbt_fixer.redaction import contains_secret_like_pattern, redact_secrets

# Built by concatenation so no AWS-key-shaped literal lands in this blob
# (GitHub push protection flags the joined form).
FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"
FAKE_SLACK_TOKEN = "xoxb-" + "1234567890-abcdefghijklmno"
FAKE_SLACK_TOKEN_2 = "xoxb-" + "111-abcdefghijklmno"
FAKE_GITHUB_TOKEN = "ghp_" + "1234567890abcdefghijklmnopqrstuv"


def test_redacts_aws_access_key_id():
    text = f"found {FAKE_AWS_KEY} in the diff"
    redacted = redact_secrets(text)
    assert FAKE_AWS_KEY not in redacted
    assert "[REDACTED:aws_access_key_id]" in redacted


def test_redacts_slack_bot_token():
    text = f"token={FAKE_SLACK_TOKEN}"
    redacted = redact_secrets(text)
    assert "xoxb-1234567890" not in redacted
    assert "[REDACTED:slack_token]" in redacted


def test_redacts_github_token():
    text = f"{FAKE_GITHUB_TOKEN} is embedded"
    redacted = redact_secrets(text)
    assert FAKE_GITHUB_TOKEN not in redacted
    assert "[REDACTED:github_token]" in redacted


def test_redacts_private_key_block():
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA...\n"
        "-----END RSA PRIVATE KEY-----"
    )
    redacted = redact_secrets(text)
    assert "MIIEpAIBAAKCAQEA" not in redacted
    assert "[REDACTED:private_key_block]" in redacted


def test_redacts_jwt():
    text = "auth=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456ghi789"
    redacted = redact_secrets(text)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
    assert "[REDACTED:jwt]" in redacted


def test_redacts_bearer_token():
    text = "Authorization: Bearer sk-abcdef1234567890ghijklmno"
    redacted = redact_secrets(text)
    assert "sk-abcdef1234567890ghijklmno" not in redacted
    assert "[REDACTED:bearer_token]" in redacted


def test_redacts_generic_password_assignment():
    text = "password: SuperSecretValue123"
    redacted = redact_secrets(text)
    assert "SuperSecretValue123" not in redacted
    assert "[REDACTED:generic_api_key_assignment]" in redacted


def test_redacts_connection_string_credentials():
    text = "postgresql://admin:hunter2@db.internal.cyera.io:5432/prod"
    redacted = redact_secrets(text)
    assert "hunter2" not in redacted
    assert "[REDACTED:connection_string_credentials]" in redacted


def test_does_not_touch_ordinary_sql_identifiers():
    text = "SELECT customer_id, tenant_id FROM stg_customers WHERE deleted_at IS NULL"
    assert redact_secrets(text) == text


def test_does_not_flag_ordinary_column_names_as_secrets():
    assert contains_secret_like_pattern("column customer_id references tenant_id") is False


def test_none_and_empty_input_never_raises():
    assert redact_secrets(None) == ""
    assert redact_secrets("") == ""
    assert contains_secret_like_pattern(None) is False
    assert contains_secret_like_pattern("") is False


def test_multiple_secrets_in_same_text_all_redacted():
    text = f"key1={FAKE_AWS_KEY} and token={FAKE_SLACK_TOKEN_2}"
    redacted = redact_secrets(text)
    assert FAKE_AWS_KEY not in redacted
    assert "xoxb-111" not in redacted


def test_contains_secret_like_pattern_true_for_detected_secret():
    assert contains_secret_like_pattern(FAKE_AWS_KEY) is True
