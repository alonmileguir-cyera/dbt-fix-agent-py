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

# The REAL rendered auditor report (dbt_auditor.report.render_report) is
# markdown: a "# <glyph> Verdict: **BLOCKED**" banner and one
# "### <Name> (`identifier`)" section per check carrying
# "**State:** **FAIL|PASS|UNCONFIRMED**" and a blockquoted "**Evidence:**".
_REAL_VERDICT_RE = re.compile(r"^#\s+.*?Verdict:\s*\*\*(?P<verdict>[A-Z_]+)\*\*", re.MULTILINE)
_REAL_SECTION_RE = re.compile(
    r"^###\s+.+?\(`(?P<id>[a-z0-9_]+)`\)\s*$(?P<body>.*?)(?=^###\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
_REAL_STATE_RE = re.compile(r"\*\*State:\*\*\s*\*\*(?P<state>[A-Z]+)\*\*")
_REAL_SEVERITY_RE = re.compile(r"\*\*Severity:\*\*\s*(?P<severity>\w+)")
_REAL_EVIDENCE_RE = re.compile(
    r"\*\*Evidence:\*\*\s*(?P<quote>(?:^>.*$\n?)+)", re.MULTILINE
)

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


# Judgment-critical audit checks the fixer must NEVER attempt to auto-fix -
# they require human review. Mechanical criticals (schema_contract_verification,
# downstream_dependency_impact) and the advisory check are fixable; these are
# not. Kept lowercase for case-insensitive matching.
JUDGMENT_CRITICAL_CHECK_IDS = frozenset({
    "tenant_isolation_integrity",
    "rap_bypass_logic",
    "destructive_operation_safety",
    "credentials_exposure",
})


@dataclass(frozen=True)
class FailingCheck:
    """One failing model/test/check, with whatever evidence/suggestion was found."""

    identifier: str
    evidence: str = ""
    suggestion: str = ""
    # "critical" | "advisory" | "" (unknown). Only KNOWN-advisory checks are
    # dropped from the re-audit efficacy requirement; unknown stays required.
    severity: str = ""


@dataclass(frozen=True)
class FailureTarget:
    """The structured target this run is meant to fix."""

    kind: FailureKind
    checks: Tuple[FailingCheck, ...]

    @property
    def identifiers(self) -> Tuple[str, ...]:
        return tuple(c.identifier for c in self.checks)

    @property
    def judgment_critical_blocking_ids(self) -> Tuple[str, ...]:
        """Blocking checks that are judgment-critical - tenant isolation, RAP
        bypass, destructive ops, credentials exposure. These are flagged precisely
        because they need HUMAN judgment; the fixer must never auto-fix them
        (its re-audit gate is the same non-deterministic auditor, least
        reliable on exactly these calls). Their presence => decline up front."""
        return tuple(
            c.identifier
            for c in self.checks
            if c.severity.lower() != "advisory"
            and c.identifier.lower() in JUDGMENT_CRITICAL_CHECK_IDS
        )

    @property
    def problem_summary(self) -> str:
        """A terse, BULLETED summary of WHAT was flagged, for the Slack post --
        one `- ` bullet per failing check with a short evidence snippet, so it
        reads at a glance instead of as a run-on. Blocking checks lead (advisory
        appended), capped at 3. The auditor's evidence is itself a bulleted
        list, so only its first point is used and it's stripped of list/quote
        markers and truncated. Purely presentational: never affects the run."""
        ordered = sorted(
            self.checks, key=lambda c: (c.severity.lower() == "advisory", self.checks.index(c))
        )
        lines: "list[str]" = []
        for check in ordered[:3]:
            # Reduce the (bulleted) evidence to its first non-empty point on one
            # line, marker-free, so the summary stays scannable.
            snippet = ""
            for raw in check.evidence.splitlines():
                stripped = raw.strip().lstrip("->*•").strip()
                if stripped:
                    snippet = " ".join(stripped.split())
                    break
            if len(snippet) > 110:
                snippet = snippet[:107].rstrip() + "..."
            lines.append(f"- `{check.identifier}`" + (f" — {snippet}" if snippet else ""))
        if len(ordered) > 3:
            lines.append(f"- (+{len(ordered) - 3} more)")
        return "\n".join(lines)

    @property
    def blocking_identifiers(self) -> Tuple[str, ...]:
        """Identifiers the fixer is actually responsible for resolving:
        every failing check EXCEPT those explicitly marked advisory. An
        advisory check ('a human should look') never causes a BLOCK on its
        own, so requiring the fix to also clear it is overreach - and the
        allowlist forbids the doc/style edits that would (live finding:
        bi-dbt #2533 round 8, a clean schema fix rejected because advisory
        sql_style_and_testability stayed failing). Unknown severity (CI,
        legacy) is treated as blocking, so those paths are unchanged."""
        return tuple(c.identifier for c in self.checks if c.severity.lower() != "advisory")


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


def _parse_real_audit_report(raw: str) -> Tuple[Tuple[FailingCheck, ...], Optional[str], bool]:
    """Parse the real rendered auditor report markdown.

    Returns ``(failing_checks, verdict, recognized)``. ``recognized`` is
    True when the text is structurally a real report (verdict banner
    present), regardless of whether any actionable failing check was
    found. UNCONFIRMED sections (a failed-audit artifact's skeleton) are
    deliberately never treated as fixable targets.
    """
    verdict_match = _REAL_VERDICT_RE.search(raw)
    if verdict_match is None:
        return (), None, False
    verdict = verdict_match.group("verdict").upper()

    checks = []
    for match in _REAL_SECTION_RE.finditer(raw):
        body = match.group("body")
        state_match = _REAL_STATE_RE.search(body)
        if state_match is None or state_match.group("state") != "FAIL":
            continue
        evidence = ""
        evidence_match = _REAL_EVIDENCE_RE.search(body)
        if evidence_match is not None:
            evidence = "\n".join(
                line.lstrip("> ").rstrip()
                for line in evidence_match.group("quote").splitlines()
            ).strip()
        severity_match = _REAL_SEVERITY_RE.search(body)
        severity = severity_match.group("severity").lower() if severity_match else ""
        checks.append(
            FailingCheck(identifier=match.group("id"), evidence=evidence, severity=severity)
        )
    return tuple(checks), verdict, True


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
        # Primary: the REAL rendered report markdown the auditor produces
        # (via DBT_AUDITOR_REPORT_PATH or Slack). The legacy machine format
        # below remains as a fallback.
        real_checks, real_verdict, recognized = _parse_real_audit_report(raw)
        if recognized:
            if real_verdict == "PASSED":
                return None, "audit verdict is PASSED; there is nothing to fix"
            if not real_checks:
                return None, (
                    f"report verdict is {real_verdict} but contains no check in "
                    "State FAIL (an UNCONFIRMED-only skeleton is a failed-audit "
                    "artifact, not a fixable finding)"
                )
            return FailureTarget(kind="audit", checks=real_checks), None

        verdict_match = _AUDIT_STATUS_RE.search(raw)
        if verdict_match is None and "check:" not in raw:
            return None, (
                "failure_context does not match a recognized auditor report format "
                "(no verdict banner, 'dbt-auditor-status: verdict=...' line, or "
                "'- check:' entries found)"
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
