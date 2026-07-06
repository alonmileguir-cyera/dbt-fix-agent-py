"""Failure-context intake: structured target parsing + untrusted fencing.

This module turns the raw `DBT_FIXER_FAILURE_CONTEXT` (a dbt Cloud CI
failure log, or a sibling auditor's rendered `BLOCKED` report) into a
`FailureTarget` -- the specific failing model/test/check identifiers plus
their evidence and any suggested remediation -- or, if the content cannot be
confidently parsed, into an explicit, specific `no_safe_fix` reason. It
never guesses: a low-confidence or unrecognized shape is a `no_safe_fix`,
not a best-effort target.

It also builds the `FencedContext` (via `dbt_fixer.fencing`) covering every
untrusted field on the config -- PR URL/title/description/diff and the raw
failure context -- so every downstream, model-bound payload always reads
this fenced rendering, never the raw fields on `FixerConfig` directly.

Two recognized failure-context shapes:

- **CI failure log** (`failure_kind="ci"`): dbt's own test-runner output,
  recognized by a `Completed with N error...` header and/or one or more
  `Failure in test <name> (<path>)` lines, each followed by an evidence
  line (e.g. `Got N results, configured to fail if != 0`).
- **Audit report** (`failure_kind="audit"`): the sibling auditor's rendered
  report, recognized by a `dbt-auditor-status: verdict=...` line and/or one
  or more `- check: <identifier>` entries (each optionally followed by
  `status:`, `evidence:`, and `suggestion:` lines). A `PASSED` (or any
  non-`BLOCKED`) verdict is explicitly *not* a target -- there is nothing to
  fix -- and resolves to `no_safe_fix` with that verdict named in the
  reason.

Nothing in this module makes a network or model call; parsing is pure,
deterministic text processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .env import FailureKind, FixerConfig
from .fencing import FencedContext, fence_context

# --- CI failure log parsing -------------------------------------------------

_CI_ERROR_HEADER_RE = re.compile(r"Completed with \d+ error", re.IGNORECASE)

_CI_TEST_FAILURE_RE = re.compile(
    r"Failure in test\s+(?P<name>[\w.$-]+)\s*\((?P<path>[^)]*)\)"
    r"(?P<body>(?:\n(?!Failure in test\b|Done\.).*)*)"
)

# --- Audit report parsing ---------------------------------------------------

_AUDIT_STATUS_RE = re.compile(r"dbt-auditor-status:\s*verdict=(?P<verdict>\w+)")

_AUDIT_CHECK_RE = re.compile(
    r"-\s*check:\s*(?P<id>[^\n]+)\n"
    r"(?:[ \t]*status:\s*(?P<status>[^\n]+)\n)?"
    r"(?:[ \t]*evidence:\s*(?P<evidence>[^\n]+)\n)?"
    r"(?:[ \t]*suggestion:\s*(?P<suggestion>[^\n]+)\n?)?"
)

_FAILING_STATUS_TOKENS = ("FAIL", "FAILING", "BLOCKED", "ERROR")

# Public aliases for the audit-report check-entry grammar, reused as-is by
# `dbt_fixer.reaudit` to parse the re-audit gate's own rendered-report
# extension block without duplicating (and risking drift from) this regex.
AUDIT_CHECK_ENTRY_RE = _AUDIT_CHECK_RE
FAILING_STATUS_TOKENS = _FAILING_STATUS_TOKENS


@dataclass(frozen=True)
class FailingCheck:
    """One failing model/test/check, with whatever evidence/suggestion was found."""

    identifier: str
    evidence: str = ""
    suggestion: str = ""


@dataclass(frozen=True)
class FailureTarget:
    """The structured target this run is meant to fix."""

    kind: FailureKind
    checks: Tuple[FailingCheck, ...]

    @property
    def identifiers(self) -> Tuple[str, ...]:
        return tuple(c.identifier for c in self.checks)


@dataclass(frozen=True)
class IntakeResult:
    """The outcome of intake: either a parsed target, or an honest reason there isn't one."""

    target: Optional[FailureTarget]
    no_safe_fix_reason: Optional[str]
    fenced_context: Optional[FencedContext]

    @property
    def ok(self) -> bool:
        return self.target is not None


def _parse_ci_failure(raw: str) -> Tuple[FailingCheck, ...]:
    checks = []
    for match in _CI_TEST_FAILURE_RE.finditer(raw):
        name = match.group("name").strip()
        body_lines = [line for line in match.group("body").strip().splitlines() if line.strip()]
        evidence = body_lines[0].strip() if body_lines else ""
        checks.append(FailingCheck(identifier=name, evidence=evidence))
    return tuple(checks)


def _parse_audit_report(raw: str) -> Tuple[FailingCheck, ...]:
    checks = []
    for match in _AUDIT_CHECK_RE.finditer(raw):
        identifier = match.group("id").strip()
        status = (match.group("status") or "").strip().upper()
        if status and not any(token in status for token in _FAILING_STATUS_TOKENS):
            continue
        evidence = (match.group("evidence") or "").strip()
        suggestion = (match.group("suggestion") or "").strip()
        checks.append(FailingCheck(identifier=identifier, evidence=evidence, suggestion=suggestion))
    return tuple(checks)


def parse_failure_target(
    kind: FailureKind, raw_failure_context: str
) -> Tuple[Optional[FailureTarget], Optional[str]]:
    """Parse `raw_failure_context` into a `FailureTarget`, or return `(None, reason)`.

    Never raises for malformed input -- an unrecognized or empty-of-signal
    shape always yields a specific `reason`. Callers must still guard
    against truly unexpected exceptions (see `resolve_intake`), but this
    function's own control flow never throws for bad *content*, only for
    genuinely-unexpected programming errors.
    """

    raw = raw_failure_context or ""

    if kind == "ci":
        if not _CI_ERROR_HEADER_RE.search(raw) and "Failure in test" not in raw:
            return None, (
                "failure_context does not match a recognized dbt CI failure format "
                "(no 'Completed with N error' header or 'Failure in test' line found)"
            )
        checks = _parse_ci_failure(raw)
        if not checks:
            return None, (
                "failure_context looked like a dbt CI failure log but no individual "
                "'Failure in test <name> (<path>)' entry could be extracted"
            )
        return FailureTarget(kind="ci", checks=checks), None

    if kind == "audit":
        verdict_match = _AUDIT_STATUS_RE.search(raw)
        if verdict_match is None and "check:" not in raw:
            return None, (
                "failure_context does not match a recognized auditor report format "
                "(no 'dbt-auditor-status: verdict=...' line or '- check:' entries found)"
            )
        if verdict_match is not None and verdict_match.group("verdict").upper() != "BLOCKED":
            return None, (
                f"audit verdict is {verdict_match.group('verdict')!r}, not BLOCKED; "
                "there is nothing to fix"
            )
        checks = _parse_audit_report(raw)
        if not checks:
            return None, (
                "failure_context looked like an auditor report but no failing "
                "'- check: <identifier>' entry could be extracted"
            )
        return FailureTarget(kind="audit", checks=checks), None

    return None, f"unrecognized failure_kind {kind!r}; cannot parse failure_context"


def build_fenced_context(config: FixerConfig) -> FencedContext:
    """Fence every untrusted field on `config` under one shared nonce."""

    return fence_context(
        {
            "pr_url": config.pr_url,
            "pr_title": config.pr_title,
            "pr_description": config.pr_description,
            "pr_diff": config.pr_diff,
            "failure_context": config.failure_context,
        }
    )


def resolve_intake(config: FixerConfig) -> IntakeResult:
    """Run the full intake step for one validated `FixerConfig`.

    Always returns an `IntakeResult`; never raises. An empty
    `failure_context` is treated the same as an unparseable one: an honest,
    specific `no_safe_fix` reason, never a guess.
    """

    fenced = build_fenced_context(config)

    raw = (config.failure_context or "").strip()
    if not raw:
        return IntakeResult(
            target=None,
            no_safe_fix_reason="failure_context is empty; there is nothing to diagnose",
            fenced_context=fenced,
        )

    target, reason = parse_failure_target(config.failure_kind, raw)
    if target is None:
        return IntakeResult(target=None, no_safe_fix_reason=reason, fenced_context=fenced)

    return IntakeResult(target=target, no_safe_fix_reason=None, fenced_context=fenced)
