"""Slack bot token retrieval.

The Slack bot token used for shadow-mode delivery is retrieved from AWS
Secrets Manager (secret id :data:`SLACK_TOKEN_SECRET_ID`), matching
Cyera's shared secrets-manager module convention, with a ``SLACK_BOT_TOKEN``
environment variable as a fallback for local development and CI runners
that inject the token directly.

Every failure mode here - no AWS credentials, the secret not existing, a
network error, a malformed secret payload, or neither source being
configured at all - degrades to returning ``None`` rather than raising.
Slack delivery is best-effort reporting, never a hard dependency of the
fix attempt itself: a missing/unavailable token must never crash or block
the run, it must only skip Slack delivery (see
:mod:`dbt_fixer.slack_delivery`).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

# The AWS Secrets Manager secret id Cyera's shared secrets-manager module
# is configured to read the Slack bot token from.
SLACK_TOKEN_SECRET_ID = "cyera-bi/slack-bot-token"

# Environment variable fallback, consulted whenever Secrets Manager
# retrieval is unavailable or fails for any reason.
SLACK_BOT_TOKEN_ENV_VAR = "SLACK_BOT_TOKEN"

__all__ = [
    "SLACK_TOKEN_SECRET_ID",
    "SLACK_BOT_TOKEN_ENV_VAR",
    "SecretsManagerClient",
    "get_slack_bot_token",
]


class SecretsManagerClient(Protocol):
    """The minimal shape of a boto3 ``secretsmanager`` client this module
    depends on, so tests can inject a fake without a live AWS connection.
    """

    def get_secret_value(self, SecretId: str) -> dict[str, Any]: ...


def _default_secrets_manager_client_factory() -> SecretsManagerClient:
    """Build a real boto3 Secrets Manager client via the default credential
    chain (environment variables, shared config/credentials files,
    EC2/ECS/Lambda instance roles, or an assumed role via ``AWS_PROFILE``).

    No access key, secret key, or session token is ever hardcoded here.
    """

    import boto3  # local import: keeps boto3 an optional cost only paid
    # when Secrets Manager retrieval is actually attempted.

    return boto3.client("secretsmanager")


def _fetch_from_secrets_manager(
    client_factory: Callable[[], SecretsManagerClient],
) -> "str | None":
    """Best-effort fetch of the Slack bot token from AWS Secrets Manager.

    Returns ``None`` (and logs the specific failure mode) on any error:
    missing/invalid AWS credentials, the secret not existing, a network
    failure, throttling, or a secret payload that doesn't contain a
    usable string value. Never raises.
    """

    try:
        client = client_factory()
    except Exception as exc:  # noqa: BLE001 - fail-closed-to-None, by design.
        logger.warning(
            "slack_bot_token: could not construct AWS Secrets Manager client "
            "(%s: %s) - falling back to %s env var",
            type(exc).__name__,
            exc,
            SLACK_BOT_TOKEN_ENV_VAR,
        )
        return None

    try:
        response = client.get_secret_value(SecretId=SLACK_TOKEN_SECRET_ID)
    except Exception as exc:  # noqa: BLE001 - fail-closed-to-None, by design.
        logger.warning(
            "slack_bot_token: AWS Secrets Manager lookup for %r failed "
            "(%s: %s) - falling back to %s env var",
            SLACK_TOKEN_SECRET_ID,
            type(exc).__name__,
            exc,
            SLACK_BOT_TOKEN_ENV_VAR,
        )
        return None

    if not isinstance(response, dict):
        logger.warning(
            "slack_bot_token: AWS Secrets Manager returned a non-mapping "
            "response for %r - falling back to %s env var",
            SLACK_TOKEN_SECRET_ID,
            SLACK_BOT_TOKEN_ENV_VAR,
        )
        return None

    value = response.get("SecretString")
    if not isinstance(value, str) or not value.strip():
        logger.warning(
            "slack_bot_token: AWS Secrets Manager response for %r had no "
            "usable SecretString - falling back to %s env var",
            SLACK_TOKEN_SECRET_ID,
            SLACK_BOT_TOKEN_ENV_VAR,
        )
        return None

    return _extract_token_from_secret_string(value)


# The JSON keys the token may live under inside the secret payload, in
# preference order. Cyera's secret stores the token as {"slack-bot-token":
# "xoxb-..."} (the last path segment of the secret id); the others are
# accepted so a differently-keyed payload still resolves rather than sending
# the whole JSON blob to Slack (which yields invalid_auth).
_TOKEN_KEYS = (
    "slack-bot-token",
    SLACK_TOKEN_SECRET_ID.rsplit("/", 1)[-1],
    SLACK_BOT_TOKEN_ENV_VAR,
    "slack_bot_token",
    "token",
)


def _extract_token_from_secret_string(value: str) -> "str | None":
    """Extract the bot token from a Secrets Manager ``SecretString``.

    The secret is stored as a JSON object keyed by the token name (matching
    Cyera's shared ``secrets_manager`` helper, which does
    ``json.loads(SecretString)[key]``), so the raw ``SecretString`` must be
    parsed rather than used verbatim - sending the whole ``{"slack-bot-token":
    ...}`` blob as the token is exactly what produced a live ``invalid_auth``.

    Resolution, all fail-closed to ``None`` (never raising):

    1. If the payload is a JSON object, use the first of :data:`_TOKEN_KEYS`
       present with a non-blank string value; failing that, if the object has
       exactly one entry, use that entry's value.
    2. If the payload is not JSON but is already a bare token string (e.g.
       ``xoxb-...``), use it verbatim.
    """

    raw = value.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        # Not JSON: treat as an already-bare token string.
        return raw or None

    if isinstance(parsed, dict):
        for key in _TOKEN_KEYS:
            candidate = parsed.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        if len(parsed) == 1:
            (only_value,) = parsed.values()
            if isinstance(only_value, str) and only_value.strip():
                return only_value.strip()
        logger.warning(
            "slack_bot_token: AWS Secrets Manager payload for %r was a JSON "
            "object with no recognizable token key - falling back to %s env var",
            SLACK_TOKEN_SECRET_ID,
            SLACK_BOT_TOKEN_ENV_VAR,
        )
        return None

    # A JSON scalar string is itself the token; anything else is unusable.
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    return None


def get_slack_bot_token(
    *,
    env: "dict[str, str] | None" = None,
    client_factory: "Callable[[], SecretsManagerClient] | None" = None,
) -> "str | None":
    """Resolve the Slack bot token to use for delivery.

    Resolution order:

    1. AWS Secrets Manager, secret id :data:`SLACK_TOKEN_SECRET_ID`.
    2. The :data:`SLACK_BOT_TOKEN_ENV_VAR` environment variable.

    Returns ``None`` (never raises) if neither source yields a usable,
    non-blank token - callers must treat ``None`` as "Slack delivery is
    unavailable this run" and degrade to a no-op, not an error.
    """

    environment = env if env is not None else os.environ
    factory = client_factory if client_factory is not None else _default_secrets_manager_client_factory

    token = _fetch_from_secrets_manager(factory)
    if token:
        return token

    env_token = environment.get(SLACK_BOT_TOKEN_ENV_VAR, "")
    if env_token and env_token.strip():
        return env_token.strip()

    logger.warning(
        "slack_bot_token: no Slack bot token available from AWS Secrets "
        "Manager (%s) or the %s environment variable - Slack delivery "
        "will be skipped for this run",
        SLACK_TOKEN_SECRET_ID,
        SLACK_BOT_TOKEN_ENV_VAR,
    )
    return None
