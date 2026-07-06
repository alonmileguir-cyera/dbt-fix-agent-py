"""Secret-like pattern redaction for anything posted outside the tool it
came from (Slack messages, stdout).

This module is a defense-in-depth safety net, not a substitute for
upstream data handling discipline: the fixer's own report content is
derived from diff/PR/failure-context text, and that text can legitimately
contain accidentally-committed secrets (that is, after all, sometimes
exactly what a security-conscious reviewer needs to be warned about).
Rather than reproduce a raw secret verbatim in a third-party chat surface
or in process stdout, any text leaving this package through a delivery
channel (Slack) or through the machine-readable stdout contract is passed
through :func:`redact_secrets` first, which replaces recognized
secret-shaped substrings with a fixed, informative placeholder that still
tells the reviewer *that* something sensitive was found and *what kind*
it looked like, without reproducing the value itself.

Patterns here are deliberately concrete and specific (named credential
formats), not a generic high-entropy-string detector - a generic entropy
detector would have an unacceptable false-positive rate against normal
SQL/dbt identifiers, hashes, and UUIDs that are not secrets at all and
are routinely useful evidence in a candidate diff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["redact_secrets", "contains_secret_like_pattern"]

_PLACEHOLDER = "[REDACTED:{kind}]"


@dataclass(frozen=True)
class _SecretPattern:
    kind: str
    pattern: "re.Pattern[str]"


# Order matters only in that more specific patterns are listed before more
# generic ones so a single matched substring isn't redacted twice under
# two different labels; :func:`redact_secrets` applies every pattern in a
# single left-to-right pass per compiled pattern, which is order-agnostic
# in practice since the patterns below do not overlap in practice.
_SECRET_PATTERNS: "tuple[_SecretPattern, ...]" = (
    _SecretPattern("aws_access_key_id", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    _SecretPattern(
        "aws_secret_access_key",
        re.compile(
            r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?"
        ),
    ),
    _SecretPattern("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    _SecretPattern(
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ),
    _SecretPattern(
        "private_key_block",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    _SecretPattern("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    _SecretPattern(
        "bearer_token",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}\b"),
    ),
    _SecretPattern(
        "generic_api_key_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*"
            r"['\"]?[A-Za-z0-9._~+/=-]{8,}['\"]?"
        ),
    ),
    _SecretPattern(
        "connection_string_credentials",
        re.compile(r"(?i)\b\w+://[^\s:@/'\"]+:[^\s:@/'\"]+@[^\s/'\"]+"),
    ),
)


def redact_secrets(text: "str | None") -> str:
    """Replace every recognized secret-shaped substring in ``text``.

    Never raises. ``None``/empty input returns an empty string. Each
    match is replaced with ``[REDACTED:<kind>]`` so the reviewer still
    sees *that* a credential-shaped value was present (and what kind)
    without the raw value itself ever reaching a third-party surface or
    process stdout.
    """

    if not text:
        return ""

    redacted = text
    for secret_pattern in _SECRET_PATTERNS:
        redacted = secret_pattern.pattern.sub(
            _PLACEHOLDER.format(kind=secret_pattern.kind), redacted
        )
    return redacted


def contains_secret_like_pattern(text: "str | None") -> bool:
    """True if any recognized secret-shaped pattern is present in ``text``."""

    if not text:
        return False
    return any(secret_pattern.pattern.search(text) for secret_pattern in _SECRET_PATTERNS)
