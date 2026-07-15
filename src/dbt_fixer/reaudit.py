"""The Re-Audit Gate: independent proof from the sealed sibling `dbt_auditor`.

Where the allowlist gate (`dbt_fixer.allowlist`) is pure code with an
opinion about *shape*, this gate asks a completely independent process --
the same sealed auditor package that found the original problem -- whether
the candidate fix actually *works*. It never trusts the fixer's own model
pass's narration; it only ever reads the auditor subprocess's own,
line-anchored stdout.

**Subprocess invocation.** The auditor is invoked as a subprocess (never
imported and called in-process -- it is a sealed, independent package) via
an injectable `SubprocessRunner`, always:

- pointed at a scratch copy of `repo_root` with the candidate diff applied
  on top -- since `repo_root` already reflects the original PR diff, this
  scratch copy is the "original PR diff + candidate fix diff" combined
  state the auditor re-audits;
- run with `DBT_AUDITOR_SHADOW_MODE` explicitly on and no
  `DBT_AUDITOR_SLACK_CHANNEL` set at all, so the sealed auditor never posts
  anywhere on this run's behalf;
- bounded by an explicit timeout.

**Stdout contract.** Only two real, confirmed lines from the sibling
package's actual `entrypoint.py` are read: `dbt-auditor-audit-status:
completed|failed` and `dbt-auditor verdict: <VERDICT> - <reason>`. This
module additionally recognizes its own report-block extension --
`dbt-auditor-report-begin` / `dbt-auditor-report-end`, wrapping the same
`- check: <id>` grammar `dbt_fixer.intake` already parses for inbound audit
reports -- so the gate can check every originally-failing identifier
individually for `kind="audit"` runs; that block is optional, and its
absence for a `kind="ci"` run (which never needs it) is not an error.

**Three distinct honest failure shapes**, never conflated:

1. `hard_no_safe_fix` -- the auditor interpreter is missing/unconfigured or
   cannot be invoked at all. This is never a skipped or passed gate.
2. An ordinary gate failure -- a nonzero exit, a timeout, unparsable
   stdout, a `BLOCKED` verdict, or (for `kind="audit"`) a still-failing
   originally-named check.
3. Success -- a non-`BLOCKED` verdict and, for `kind="audit"`, every
   originally-failing check identifier now scoring passing.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

from .diffparse import DiffParseError, PatchApplyError, apply_diff
from .env import FailureKind
from .intake import AUDIT_CHECK_ENTRY_RE

from .pathsafe import PathTraversalError
from .scratch import ScratchCopyError, scratch_copy

logger = logging.getLogger(__name__)

# A re-audit that fails to COMPLETE is a transient artifact (Bedrock
# throttle/crash/no-result), not a judgment of the fix; retry it this many
# times before giving up. A BLOCKED verdict is never retried (see the gate).
# Kept low: retries only help FAST artifacts (crashes/status=failed fail in
# seconds); a TIMEOUT is never retried (it already consumed the full budget,
# so a second attempt would just eat it again and spiral the wall-clock).
_MAX_REAUDIT_ARTIFACT_ATTEMPTS = 2

# The subprocess runner marks a timed-out auditor with this sentinel
# returncode (see real_reaudit_subprocess_runner). Such an outcome is a
# completion artifact but must NOT be retried within budget.
_TIMEOUT_RETURNCODE = -1

__all__ = [
    "AuditorInvocationError",
    "ProcessOutcome",
    "SubprocessRunner",
    "ReAuditVerdict",
    "build_auditor_env",
    "build_auditor_args",
    "parse_auditor_stdout",
    "run_reaudit_gate",
]

ENTRYPOINT_MODULE = "dbt_auditor.entrypoint"

_AUDIT_STATUS_LINE_RE = re.compile(r"^dbt-auditor-audit-status:\s*(?P<status>\w+)\s*$", re.MULTILINE)
_VERDICT_LINE_RE = re.compile(
    r"^dbt-auditor verdict:\s*(?P<verdict>\w+)\s*-\s*(?P<reason>.*)$", re.MULTILINE
)
_REPORT_BLOCK_RE = re.compile(
    r"dbt-auditor-report-begin\n(?P<body>.*?)\ndbt-auditor-report-end", re.DOTALL
)

_NON_BLOCKED_VERDICTS = ("PASSED", "NEEDS_REVIEW")
_BLOCKED_VERDICT = "BLOCKED"


class AuditorInvocationError(RuntimeError):
    """Raised by a `SubprocessRunner` when the auditor interpreter cannot be found/started.

    Distinct from a completed process that exits nonzero: this is for the
    case the process never even started (missing interpreter, permission
    denied, etc.), which `run_reaudit_gate` always maps to a hard
    `no_safe_fix`, never an ordinary gate failure.
    """


@dataclass(frozen=True)
class ProcessOutcome:
    """The result of one completed subprocess invocation."""

    returncode: int
    stdout: str
    stderr: str = ""


SubprocessRunner = Callable[[list, Mapping[str, str], Path, float], ProcessOutcome]


@dataclass(frozen=True)
class ReAuditVerdict:
    """The re-audit gate's outcome for one candidate diff.

    `hard_no_safe_fix=True` marks the one case (missing/uninvokable
    auditor interpreter) that must never be treated as an ordinary gate
    failure or a skip.
    """

    passed: bool
    hard_no_safe_fix: bool = False
    violation: Optional[str] = None
    reason: str = ""
    auditor_verdict: Optional[str] = None
    checked_identifiers: Tuple[str, ...] = field(default_factory=tuple)


def build_auditor_env(
    *,
    repo_path: "str | Path",
    pr_diff: str,
    pr_title: str,
    pr_description: str,
    pr_url: str,
    report_path: "Optional[str | Path]" = None,
) -> Dict[str, str]:
    """Build the sealed auditor's real `DBT_AUDITOR_*` env contract for this run.

    Always sets shadow mode on and deliberately never sets
    `DBT_AUDITOR_SLACK_CHANNEL` -- the re-audit gate never lets the sealed
    auditor post anywhere on this run's behalf.

    The ambient passthrough is load-bearing (live e2e finding, run 8): the
    auditor authenticates to Bedrock via AWS_* env vars; a bare env leaves
    it credential-starved and every re-audit fails as an artifact. The
    execution knobs are equally load-bearing: the auditor's package default
    (120s/pass) guarantees timeouts on real diffs.
    """

    import os

    env: Dict[str, str] = {}
    for key in _AMBIENT_PASSTHROUGH_KEYS:
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(
        {
            # The re-audit runs with its cwd set to a scratch copy of the PR.
            # Keep that untrusted directory out of Python's import search so a
            # PR-supplied ``dbt_auditor`` package cannot shadow the sealed
            # installed sibling and execute with the re-audit's AWS identity.
            "PYTHONSAFEPATH": "1",
            "DBT_AUDITOR_REPO_PATH": str(repo_path),
            "DBT_AUDITOR_PR_DIFF": pr_diff,
            "DBT_AUDITOR_PR_TITLE": pr_title,
            "DBT_AUDITOR_PR_DESCRIPTION": pr_description,
            "DBT_AUDITOR_PR_URL": pr_url,
            "DBT_AUDITOR_SHADOW_MODE": "true",
            # 900s, not 600s: in CI (slower runner + Bedrock throttling on a
            # minimal role) the inner audit consistently hit a 600s budget
            # and self-reported status=failed at ~12 min, cascading through
            # retries into a ~46-min run (bi-dbt #2533 round 9). Letting the
            # single audit COMPLETE is faster overall than failing + retrying.
            "DBT_AUDITOR_TIMEOUT_SECONDS": "900",
            "DBT_AUDITOR_MAX_TURNS": "25",
            "DBT_AUDITOR_MAX_TOOL_CALLS": "50",
        }
    )
    if report_path:
        env["DBT_AUDITOR_REPORT_PATH"] = str(report_path)
    return env


# Mirrors the DAG invocation layer's allowlist: PATH/HOME for the
# interpreter and boto3 config, AWS_* for Bedrock/Secrets credentials,
# CA bundles for TLS interception environments.
_AMBIENT_PASSTHROUGH_KEYS = (
    "PATH",
    "HOME",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_STS_REGIONAL_ENDPOINTS",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
)


def build_effective_diff(
    *, repo_root: "str | Path", final_root: "str | Path", pr_diff: str, candidate_diff: str
) -> str:
    """The diff the re-audit should judge: base -> (PR + fix), as ONE clean
    change - exactly what the PR would look like had the author pushed the
    fix themselves.

    Raw PR diff contradicts the patched repo (live finding, bi-dbt #2533
    round 2); concatenating PR + candidate makes the fix read as deleting
    the PR's tested lines - the exact "weakening" pattern the auditor is
    primed to block (round 3). The effective diff has neither problem.

    Reconstruction is pure Python: `repo_root` is the PR-head checkout, so
    applying the INVERSE of the PR diff to a scratch copy of it yields the
    base tree for every touched file; `final_root` (the candidate-patched
    scratch) is the after state; `generate_unified_diff` renders base ->
    final for the union of touched paths.

    Raises on any parse/apply failure - the caller falls back to the
    cumulative concatenation, which is imperfect but never contradictory.
    """

    from .diffing import generate_unified_diff
    from .diffparse import invert_diff, _diff_paths, apply_diff as _apply

    touched = sorted(set(_diff_paths(pr_diff)) | set(_diff_paths(candidate_diff)))
    with scratch_copy(Path(repo_root)) as base_root:
        if pr_diff.strip():
            _apply(base_root, invert_diff(pr_diff))
        return generate_unified_diff(base_root, final_root, touched)


def combine_diffs(pr_diff: str, candidate_diff: str) -> str:
    """Concatenate the PR diff and the candidate diff into one cumulative
    unified diff (later hunks describe changes applied on top of earlier
    ones - the same way sequential commits read)."""

    parts = [d.strip("\n") for d in (pr_diff, candidate_diff) if d and d.strip()]
    return "\n".join(parts) + ("\n" if parts else "")


def build_auditor_args(auditor_python: str) -> list:
    """Build the subprocess argv for invoking the sealed auditor as a module."""

    # ``-P`` is the interpreter-level guarantee that the untrusted scratch cwd
    # is not prepended to sys.path. PYTHONSAFEPATH in build_auditor_env is the
    # matching environment-level defense in depth.
    return [auditor_python, "-P", "-m", ENTRYPOINT_MODULE]


@dataclass(frozen=True)
class _ParsedStdout:
    status: Optional[str]
    verdict: Optional[str]
    verdict_reason: str
    check_statuses: Dict[str, str]


def parse_auditor_stdout(stdout: str) -> _ParsedStdout:
    """Parse the auditor's line-anchored stdout contract.

    Returns a `_ParsedStdout` with `status`/`verdict` set to `None` when
    their respective line is absent or does not match -- this function
    never raises, so `run_reaudit_gate` can treat "absent" and "present but
    unparsable" identically as an honest gate failure.
    """

    status_match = _AUDIT_STATUS_LINE_RE.search(stdout)
    verdict_match = _VERDICT_LINE_RE.search(stdout)

    check_statuses: Dict[str, str] = {}
    report_match = _REPORT_BLOCK_RE.search(stdout)
    if report_match is not None:
        for entry in AUDIT_CHECK_ENTRY_RE.finditer(report_match.group("body") + "\n"):
            identifier = entry.group("id").strip()
            status = (entry.group("status") or "").strip()
            check_statuses[identifier] = status

    return _ParsedStdout(
        status=status_match.group("status") if status_match else None,
        verdict=verdict_match.group("verdict").upper() if verdict_match else None,
        verdict_reason=verdict_match.group("reason").strip() if verdict_match else "",
        check_statuses=check_statuses,
    )


# The real rendered report's check sections: "### <Name> (`identifier`)"
# followed by "**State:** **PASS|FAIL|UNCONFIRMED**" (see
# dbt_auditor.report.render_report).
_REPORT_SECTION_RE = re.compile(
    r"^###\s+.+?\(`(?P<id>[a-z0-9_]+)`\)\s*$(?P<body>.*?)(?=^###\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
_REPORT_STATE_RE = re.compile(r"\*\*State:\*\*\s*\*\*(?P<state>[A-Z]+)\*\*")
_REPORT_SEVERITY_RE = re.compile(r"\*\*Severity:\*\*\s*(?P<severity>\w+)")


def _check_statuses_from_report(report_text: str) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for match in _REPORT_SECTION_RE.finditer(report_text):
        state_match = _REPORT_STATE_RE.search(match.group("body"))
        if state_match is not None:
            statuses[match.group("id")] = state_match.group("state")
    return statuses


def _check_severities_from_report(report_text: str) -> Dict[str, str]:
    severities: Dict[str, str] = {}
    for match in _REPORT_SECTION_RE.finditer(report_text):
        sev_match = _REPORT_SEVERITY_RE.search(match.group("body"))
        if sev_match is not None:
            severities[match.group("id")] = sev_match.group("severity").lower()
    return severities


# Explicit PASS allowlist, NOT a failing-token denylist: a check the auditor
# could not CONFIRM (UNCONFIRMED / SKIPPED / WARN / pending / n-a / empty) is
# NOT a pass. Reading a failure-to-judge as "fixed" was a critical fail-open
# (red-team finding A) - an UNCONFIRMED originally-failing check would clear
# the efficacy gate. Only an affirmative pass state counts.
_PASSING_STATES = frozenset({"PASS", "PASSED", "PASSING", "OK"})


def _check_is_passing(status: str) -> bool:
    return status.strip().upper() in _PASSING_STATES


def run_reaudit_gate(
    *,
    repo_root: "str | Path",
    candidate_diff: str,
    pr_diff: str,
    pr_title: str,
    pr_description: str,
    pr_url: str,
    auditor_python: Optional[str],
    failure_kind: FailureKind,
    originally_failing_check_ids: Tuple[str, ...],
    timeout_seconds: float,
    subprocess_runner: SubprocessRunner,
) -> ReAuditVerdict:
    """Run the re-audit gate for one candidate diff against the sealed auditor.

    Args:
        repo_root: The original checkout (already reflecting the PR's own
            diff). Never mutated: a fresh scratch copy is made, the
            candidate diff is applied to *that*, and only the scratch copy
            is ever handed to the auditor subprocess.
        candidate_diff: The unified diff text for this round's candidate.
        pr_diff, pr_title, pr_description, pr_url: Forwarded verbatim into
            the sealed auditor's own `DBT_AUDITOR_*` env contract.
        auditor_python: Path to the Python interpreter to run the sealed
            auditor with, or `None` if unconfigured.
        failure_kind: `"ci"` or `"audit"` -- selects whether the
            all-originally-failing-checks-must-now-pass rule applies.
        originally_failing_check_ids: The check identifiers this run was
            meant to fix (only consulted for `failure_kind="audit"`).
        timeout_seconds: The bound handed to `subprocess_runner`.
        subprocess_runner: The injectable subprocess invocation callable
            (a real one in production, a fake in every test). Raising
            `AuditorInvocationError` signals the interpreter could not be
            found or started at all.

    Returns:
        A `ReAuditVerdict`. Never raises: every failure mode (missing
        interpreter, scratch-copy error, candidate-apply error, nonzero
        exit, timeout, unparsable stdout, `BLOCKED` verdict, a still-
        failing named check) resolves to a `ReAuditVerdict` field, never an
        exception escaping this function.
    """

    if not auditor_python:
        return ReAuditVerdict(
            passed=False,
            hard_no_safe_fix=True,
            violation="auditor_interpreter_missing",
            reason="DBT_FIXER_AUDITOR_PYTHON is not configured; the sealed auditor cannot be invoked",
        )

    repo_root = Path(repo_root)

    try:
        with scratch_copy(repo_root) as scratch_root:
            try:
                apply_diff(scratch_root, candidate_diff)
            except (DiffParseError, PatchApplyError, PathTraversalError) as exc:
                return ReAuditVerdict(
                    passed=False,
                    violation="candidate_diff_did_not_apply",
                    reason=f"candidate diff could not be applied for re-audit: {exc}",
                )

            args = build_auditor_args(auditor_python)

            # The re-audit judges the EFFECTIVE diff: base -> (PR+fix) as one
            # clean change. See build_effective_diff for why the two simpler
            # choices both produced false blocks in production (bi-dbt #2533
            # rounds 2 and 3). Deterministic, so built once outside the loop.
            try:
                reaudit_diff = build_effective_diff(
                    repo_root=repo_root,
                    final_root=scratch_root,
                    pr_diff=pr_diff,
                    candidate_diff=candidate_diff,
                )
            except Exception as exc:  # noqa: BLE001 - presentation fallback
                logger.warning(
                    "re-audit: could not build the effective diff (%s); "
                    "falling back to the cumulative PR+candidate diff",
                    exc,
                )
                reaudit_diff = combine_diffs(pr_diff, candidate_diff)

            # An EMPTY effective diff means the candidate fully reverts the
            # PR's flagged change (PR + fix == base), so there is no residual
            # change to audit and no residual risk - the originally-failing
            # checks are vacuously satisfied. Auditing an empty diff yields no
            # check results, which the auditor reports as a fail-closed
            # ARTIFACT; short-circuit to PASS instead. (A no-op *candidate* is
            # already rejected upstream by the allowlist, so an empty
            # *effective* diff here always means a genuine full revert - the
            # correct fix for e.g. a downstream break caused purely by a
            # removal. Found via the downstream-restore coverage test.)
            if not reaudit_diff.strip():
                return ReAuditVerdict(
                    passed=True,
                    reason=(
                        "the candidate fully reverts the PR's flagged change "
                        "(empty effective diff); nothing remains to audit"
                    ),
                    checked_identifiers=tuple(originally_failing_check_ids),
                )

            # A re-audit that fails to COMPLETE (nonzero exit, timeout, or a
            # 'status != completed' artifact) is a failure to JUDGE, not a
            # judgment, so it is retried a bounded number of times - mirroring
            # the sealed auditor's own artifact-retry philosophy (bi-dbt #2533
            # hit a transient artifact on an otherwise-sound fix). A BLOCKED
            # verdict or a still-failing named check is a real judgment and is
            # never retried here: the loop breaks on the first COMPLETED run,
            # and the classification below handles every terminal case.
            import tempfile

            outcome = None
            report_text = ""
            for attempt in range(1, _MAX_REAUDIT_ARTIFACT_ATTEMPTS + 1):
                report_fd, report_file = tempfile.mkstemp(suffix=".md", prefix="dbt-reaudit-")
                os.close(report_fd)
                try:
                    env = build_auditor_env(
                        repo_path=scratch_root,
                        pr_diff=reaudit_diff,
                        pr_title=pr_title,
                        pr_description=pr_description,
                        pr_url=pr_url,
                        report_path=report_file,
                    )
                    try:
                        outcome = subprocess_runner(args, env, scratch_root, timeout_seconds)
                    except AuditorInvocationError as exc:
                        return ReAuditVerdict(
                            passed=False,
                            hard_no_safe_fix=True,
                            violation="auditor_interpreter_missing",
                            reason=f"the sealed auditor could not be invoked: {exc}",
                        )
                    try:
                        with open(report_file, "r", encoding="utf-8") as handle:
                            report_text = handle.read()
                    except OSError:
                        report_text = ""
                finally:
                    try:
                        os.unlink(report_file)
                    except OSError:
                        pass

                completed = outcome.returncode == 0 and (
                    (parse_auditor_stdout(outcome.stdout).status or "").lower() == "completed"
                )
                # Retry only FAST artifacts. A timeout already burned the full
                # budget; retrying it would just burn it again (the wall-clock
                # spiral we are avoiding), so treat a timeout as terminal here.
                timed_out = outcome.returncode == _TIMEOUT_RETURNCODE
                if completed or timed_out or attempt == _MAX_REAUDIT_ARTIFACT_ATTEMPTS:
                    break
                logger.warning(
                    "re-audit attempt %d/%d did not complete (exit=%s); retrying",
                    attempt,
                    _MAX_REAUDIT_ARTIFACT_ATTEMPTS,
                    outcome.returncode,
                )
    except ScratchCopyError as exc:
        return ReAuditVerdict(
            passed=False,
            violation="scratch_copy_failed",
            reason=f"could not create a scratch copy for re-audit: {exc}",
        )

    if outcome.returncode != 0:
        return ReAuditVerdict(
            passed=False,
            violation="auditor_nonzero_exit",
            reason=(
                f"the sealed auditor exited with code {outcome.returncode}; "
                f"stderr: {outcome.stderr.strip()!r}"
            ),
        )

    parsed = parse_auditor_stdout(outcome.stdout)

    # The real auditor delivers per-check states via the report FILE
    # (DBT_AUDITOR_REPORT_PATH), not stdout. Prefer it; the stdout block
    # remains a fallback for fakes/older pins.
    if report_text:
        file_statuses = _check_statuses_from_report(report_text)
        if file_statuses:
            parsed = _ParsedStdout(
                status=parsed.status,
                verdict=parsed.verdict,
                verdict_reason=parsed.verdict_reason,
                check_statuses={**file_statuses, **parsed.check_statuses},
            )

    if parsed.status is None or parsed.status.lower() != "completed":
        return ReAuditVerdict(
            passed=False,
            violation="auditor_output_unparsable",
            reason=(
                "the sealed auditor's stdout does not contain a recognized "
                f"'dbt-auditor-audit-status: completed' line (status={parsed.status!r})"
            ),
        )

    if parsed.verdict is None:
        return ReAuditVerdict(
            passed=False,
            violation="auditor_output_unparsable",
            reason="the sealed auditor's stdout does not contain a recognized verdict line",
        )

    if parsed.verdict == _BLOCKED_VERDICT:
        return ReAuditVerdict(
            passed=False,
            violation="auditor_verdict_blocked",
            reason=f"the sealed auditor's re-audit verdict is BLOCKED: {parsed.verdict_reason}",
            auditor_verdict=parsed.verdict,
        )

    if parsed.verdict not in _NON_BLOCKED_VERDICTS:
        return ReAuditVerdict(
            passed=False,
            violation="auditor_output_unparsable",
            reason=f"the sealed auditor returned an unrecognized verdict: {parsed.verdict!r}",
            auditor_verdict=parsed.verdict,
        )

    if failure_kind == "audit" and originally_failing_check_ids:
        still_failing = [
            identifier
            for identifier in originally_failing_check_ids
            if not _check_is_passing(parsed.check_statuses.get(identifier, ""))
        ]
        if still_failing:
            return ReAuditVerdict(
                passed=False,
                violation="auditor_check_still_failing",
                reason=(
                    "the following originally-failing check(s) do not yet score passing "
                    f"in the re-audit report: {', '.join(still_failing)}"
                ),
                auditor_verdict=parsed.verdict,
                checked_identifiers=tuple(originally_failing_check_ids),
            )

    # Defense-in-depth (red-team finding B): a fix must not clear the named
    # check while REGRESSING another. The gate previously trusted the stdout
    # verdict line + only the originally-named checks, so a fix that broke a
    # different (non-originally-failing) critical check could still pass if
    # the verdict line wasn't literally BLOCKED. Reject if the authoritative
    # per-check report shows ANY non-advisory check failing - regardless of
    # whether it was originally failing. Advisory checks (never blocking) are
    # exempt; unknown severity is treated as non-advisory (fail closed).
    if report_text:
        severities = _check_severities_from_report(report_text)
        regressed = sorted(
            cid
            for cid, state in parsed.check_statuses.items()
            if not _check_is_passing(state)
            and severities.get(cid, "critical").lower() != "advisory"
        )
        if regressed:
            return ReAuditVerdict(
                passed=False,
                violation="auditor_check_regressed",
                reason=(
                    "the re-audit report shows non-advisory check(s) not passing, so the "
                    f"fix is not safe: {', '.join(regressed)}"
                ),
                auditor_verdict=parsed.verdict,
            )

    return ReAuditVerdict(
        passed=True,
        reason=f"re-audit verdict {parsed.verdict} is non-blocked; all required checks pass",
        auditor_verdict=parsed.verdict,
        checked_identifiers=tuple(originally_failing_check_ids),
    )
