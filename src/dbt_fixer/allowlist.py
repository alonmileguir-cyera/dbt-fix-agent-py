"""The Allowlist Classifier Gate: pure, deterministic, no model calls.

This gate is the platform's own opinion of what a "safe" candidate fix can
possibly look like, applied *after* a proposal has been applied to a
scratch copy and diffed (`dbt_fixer.fix_pipeline`), but *before* the far
more expensive re-audit gate (`dbt_fixer.reaudit`) is ever invoked. Every
rule here is ordinary Python control flow over already-parsed diff text
(`dbt_fixer.diffparse`) -- no model runner, no Bedrock client, no network
call, nothing non-deterministic. Running the same candidate through
`run_allowlist_gate` any number of times always produces the same
`AllowlistVerdict`.

Checks, applied in this fixed order (the first violation found is the one
reported -- this function never accumulates or ranks multiple violations):

1. **Malformed or no-op candidate.** The candidate diff must parse
   (`dbt_fixer.diffparse.parse_diff`) and apply cleanly to a *fresh* scratch
   copy of `repo_root` that this gate makes and tears down itself,
   independent of whatever scratch copy produced the diff. A diff that is
   empty, fails to parse, or fails to apply is rejected. A diff that
   applies but leaves every touched file byte-identical to its original
   content (a true no-op, even if it superficially looks like a change) is
   also rejected.
2. **File-type restriction.** Every touched path must be a `.yml`, `.yaml`,
   `.md`, or `.sql` file *under* `models/`. Anything else is rejected.
3. **Hard caps.** The number of touched files and the total changed-line
   count (removed + added, across every file) must each stay within the
   caller-supplied `AllowlistCaps`. Independently of the caps, any changed
   line -- added or removed, in any touched file -- matching a sensitive
   pattern (a dbt hook, a `materialized` config change, or a masking/bypass
   keyword) is rejected outright, even if every cap is satisfied.
4. **PR-authored-only SQL deletion.** Every line a `.sql` candidate diff
   *removes* must consume one matching line that the original PR diff added
   at that same path. This lets a proposal correct or revert SQL introduced
   by the PR while preventing it from deleting pre-existing base SQL. The
   rule applies identically for `kind="ci"` and `kind="audit"`.
5. **Test weakening, by failure kind.** A removed dbt schema-test entry (a
   `tests:` list item, a bare `tests:` header, or a loosened `severity:
   error` line) in a `.yml`/`.yaml` file is:
   - always rejected under `kind="ci"` -- CI tests are never allowed to be
     deleted or weakened by this gate;
   - rejected under `kind="audit"` too, *unless* one of the originally
     failing checks (`failing_checks`) both (a) identifies that same test
     and (b) has evidence text that explicitly proves the test wrong (see
     `_EVIDENCE_PROVES_WRONG_RE`); only then is the removal accepted.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .diffparse import DiffParseError, FileDiffBlock, PatchApplyError, apply_diff, parse_diff
from .env import FailureKind
from .intake import FailingCheck
from .pathsafe import PathTraversalError
from .scratch import ScratchCopyError, scratch_copy

__all__ = [
    "AllowlistCaps",
    "AllowlistVerdict",
    "run_allowlist_gate",
    "ALLOWED_EXTENSIONS",
    "ALLOWED_PATH_PREFIX",
    "SENSITIVE_PATTERNS",
]

ALLOWED_EXTENSIONS: Tuple[str, ...] = (".yml", ".yaml", ".md", ".sql")
ALLOWED_PATH_PREFIX = "models/"

# Sensitive patterns that are rejected regardless of file type or cap
# compliance: dbt hooks, a materialization change, or any masking/bypass
# keyword that could be used to weaken Cyera's own data-masking posture.
SENSITIVE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"pre[-_]hook", re.IGNORECASE),
    re.compile(r"post[-_]hook", re.IGNORECASE),
    re.compile(r"materialized\s*[:=]", re.IGNORECASE),
    re.compile(r"\bmask(ing)?\b", re.IGNORECASE),
    re.compile(r"\bunmask(ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bbypass\b", re.IGNORECASE),
)

_TEST_LIST_ITEM_RE = re.compile(
    r"^\s*-\s*(not_null|unique|accepted_values|relationships"
    r"|dbt_utils\.[\w.]+|dbt_expectations\.[\w.]+|\w*test\w*)\s*:?\s*$",
    re.IGNORECASE,
)
_TEST_HEADER_RE = re.compile(r"^\s*tests\s*:\s*$")
_SEVERITY_ERROR_RE = re.compile(r"^\s*severity\s*:\s*error\s*$", re.IGNORECASE)

_EVIDENCE_PROVES_WRONG_RE = re.compile(r"proven wrong", re.IGNORECASE)


@dataclass(frozen=True)
class AllowlistCaps:
    """The two hard numeric caps this gate enforces, sourced from `FixerConfig`."""

    max_changed_files: int
    max_changed_lines: int


@dataclass(frozen=True)
class AllowlistVerdict:
    """The allowlist gate's outcome for one candidate diff.

    `violation` is a short, stable machine-readable code (e.g.
    `"file_type_not_allowed"`), always `None` when `passed` is `True`.
    `reason` is always a human-readable explanation -- populated on
    failure, and a fixed confirmation string on success.
    """

    passed: bool
    violation: Optional[str] = None
    reason: str = ""


def _is_test_related_removed_line(line: str) -> Optional[str]:
    """Return a short description if `line` looks like a schema-test entry, else `None`."""

    if _TEST_HEADER_RE.match(line):
        return "tests: block header"
    if _SEVERITY_ERROR_RE.match(line):
        return "severity: error"
    match = _TEST_LIST_ITEM_RE.match(line)
    if match:
        return match.group(1)
    return None


def _sensitive_pattern_hit(text: str) -> Optional[str]:
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def _check_file_types(blocks: Tuple[FileDiffBlock, ...]) -> Optional[AllowlistVerdict]:
    for block in blocks:
        path = Path(block.path)
        # bi-dbt is a multi-project repo: model files live under
        # <project>/models/... (bi-dbt-us/models/, bi-dbt-multiregion/models/,
        # ...), not a repo-root models/. The containment rule is therefore
        # "has a models/ path segment", not "starts with models/". A bare
        # repo-root models/ layout still satisfies it.
        if "models" not in path.parts[:-1]:
            return AllowlistVerdict(
                passed=False,
                violation="file_type_not_allowed",
                reason=(
                    f"{block.path!r} has no models/ path segment - only files "
                    "inside a dbt project's models/ tree may be touched"
                ),
            )
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return AllowlistVerdict(
                passed=False,
                violation="file_type_not_allowed",
                reason=(
                    f"{block.path!r} has extension {path.suffix!r}, not one of "
                    f"{ALLOWED_EXTENSIONS}"
                ),
            )
    return None


def _check_hard_caps(
    blocks: Tuple[FileDiffBlock, ...], caps: AllowlistCaps
) -> Optional[AllowlistVerdict]:
    if len(blocks) > caps.max_changed_files:
        return AllowlistVerdict(
            passed=False,
            violation="max_changed_files_exceeded",
            reason=f"candidate touches {len(blocks)} file(s), exceeding the cap of {caps.max_changed_files}",
        )

    total_changed_lines = sum(len(b.removed_lines()) + len(b.added_lines()) for b in blocks)
    if total_changed_lines > caps.max_changed_lines:
        return AllowlistVerdict(
            passed=False,
            violation="max_changed_lines_exceeded",
            reason=(
                f"candidate changes {total_changed_lines} line(s), exceeding the cap of "
                f"{caps.max_changed_lines}"
            ),
        )

    for block in blocks:
        for line in (*block.removed_lines(), *block.added_lines()):
            hit = _sensitive_pattern_hit(line)
            if hit is not None:
                return AllowlistVerdict(
                    passed=False,
                    violation="sensitive_pattern_detected",
                    reason=f"{block.path!r} touches a sensitive pattern ({hit!r}): {line!r}",
                )

    return None


def _check_restore_only_sql_deletions(
    blocks: Tuple[FileDiffBlock, ...], pr_added_lines_by_path: dict[str, Tuple[str, ...]]
) -> Optional[AllowlistVerdict]:
    for block in blocks:
        if not block.path.lower().endswith(".sql"):
            continue
        # Preserve multiplicity: one line added by the PR authorizes at most
        # one candidate removal, even when identical SQL appears repeatedly.
        allowed_removals = Counter(pr_added_lines_by_path.get(block.path, ()))
        for removed_line in block.removed_lines():
            if allowed_removals[removed_line] <= 0:
                return AllowlistVerdict(
                    passed=False,
                    violation="sql_deletion_not_a_restore",
                    reason=(
                        f"{block.path!r} deletes a line the original PR diff did not add "
                        "(not a correction/revert of PR-authored SQL): "
                        f"{removed_line!r}"
                    ),
                )
            allowed_removals[removed_line] -= 1
    return None


_COLUMN_NAME_RE = re.compile(r"^\s*-\s*name\s*:", re.IGNORECASE)


def _restructure_exempt_removed_tests(block: FileDiffBlock) -> Counter:
    """Removed test lines that belong to a column whose OWN `- name:` line is
    also removed - i.e. the column is being renamed/restructured, the only
    case where a test legitimately "moves" (bi-dbt #2533 round 4). A test
    removed from a column that PERSISTS (its `- name:` unchanged) is genuine
    weakening and is NEVER exempt (red-team finding 2: relocating a
    sensitive column's test under a throwaway column defeated the file-wide
    Counter). Walks the OLD side (removed + context lines) tracking whether
    the current column header was removed."""
    exempt: Counter = Counter()
    for hunk in block.hunks:
        current_col_removed = False
        for line in hunk.lines:
            if line.kind == "added":
                continue  # old-side walk only
            if _COLUMN_NAME_RE.match(line.text):
                current_col_removed = line.kind == "removed"
                continue
            if (
                line.kind == "removed"
                and current_col_removed
                and _is_test_related_removed_line(line.text.rstrip("\n")) is not None
            ):
                exempt[line.text.strip()] += 1
    return exempt


def _check_test_weakening(
    blocks: Tuple[FileDiffBlock, ...],
    failure_kind: FailureKind,
    failing_checks: Tuple[FailingCheck, ...],
) -> Optional[AllowlistVerdict]:
    for block in blocks:
        suffix = Path(block.path).suffix.lower()
        if suffix not in (".yml", ".yaml"):
            continue
        # A removed test line is a legitimate MOVE only when (a) its owning
        # column is itself being removed/renamed AND (b) the identical test
        # line is re-added elsewhere in the file - the column-rename fix
        # shape (bi-dbt #2533 round 4). A test stripped from a column that
        # persists is weakening even if an identical string reappears under a
        # different column (red-team finding 2), so the exemption is scoped
        # to restructure-removed columns, not the whole file.
        readded = Counter(line.strip() for line in block.added_lines())
        restructure_exempt = _restructure_exempt_removed_tests(block)
        for removed_line in block.removed_lines():
            test_desc = _is_test_related_removed_line(removed_line)
            if test_desc is None:
                continue
            normalized = removed_line.strip()
            if restructure_exempt[normalized] > 0 and readded[normalized] > 0:
                restructure_exempt[normalized] -= 1
                readded[normalized] -= 1
                continue

            if failure_kind == "ci":
                return AllowlistVerdict(
                    passed=False,
                    violation="test_weakening_rejected_ci",
                    reason=(
                        f"{block.path!r} deletes or weakens a test under kind=ci "
                        f"({test_desc!r}: {removed_line!r}), which is categorically rejected"
                    ),
                )

            # kind == "audit": only accepted if a failing check both names
            # this test and has evidence explicitly proving it wrong.
            proven = any(
                test_desc.lower() in check.identifier.lower()
                and _EVIDENCE_PROVES_WRONG_RE.search(check.evidence or "")
                for check in failing_checks
            )
            if not proven:
                return AllowlistVerdict(
                    passed=False,
                    violation="test_weakening_rejected_audit_unproven",
                    reason=(
                        f"{block.path!r} removes a schema test ({test_desc!r}: {removed_line!r}) "
                        "under kind=audit without matching, explicit evidence proving it wrong"
                    ),
                )
    return None


def run_allowlist_gate(
    *,
    repo_root: "str | Path",
    candidate_diff: str,
    pr_diff: str,
    failure_kind: FailureKind,
    failing_checks: Tuple[FailingCheck, ...] = (),
    caps: AllowlistCaps,
) -> AllowlistVerdict:
    """Run every deterministic allowlist rule against one candidate diff.

    Args:
        repo_root: The original, unmutated checkout. This gate makes and
            tears down its own scratch copy to independently verify the
            candidate diff applies cleanly; it never mutates `repo_root`.
        candidate_diff: The unified diff text produced for this round's
            candidate fix (see `dbt_fixer.fix_pipeline`).
        pr_diff: The original, untrusted PR diff text (already fenced for
            any model-facing rendering elsewhere; this gate only ever reads
            its raw text to compare deleted lines, never renders it to a
            model).
        failure_kind: `"ci"` or `"audit"` -- selects the test-weakening
            rule variant.
        failing_checks: The originally-failing check identifiers/evidence
            (from `dbt_fixer.intake.FailureTarget.checks`), used only by
            the `kind="audit"` test-weakening exception.
        caps: The hard file-count/changed-line-count caps to enforce.

    Returns:
        An `AllowlistVerdict`. Never raises: every failure mode (parse
        error, apply error, scratch-copy infrastructure error) is folded
        into a `passed=False` verdict with a specific `violation` code.
    """

    repo_root = Path(repo_root)

    if candidate_diff.strip() == "":
        return AllowlistVerdict(
            passed=False, violation="no_op_candidate", reason="candidate diff is empty"
        )

    try:
        blocks = parse_diff(candidate_diff)
    except DiffParseError as exc:
        return AllowlistVerdict(
            passed=False, violation="patch_apply_failed", reason=f"candidate diff is malformed: {exc}"
        )

    if not blocks:
        return AllowlistVerdict(
            passed=False,
            violation="no_op_candidate",
            reason="candidate diff parses to zero changed files",
        )

    try:
        with scratch_copy(repo_root) as verification_root:
            try:
                apply_diff(verification_root, candidate_diff)
            except (PatchApplyError, DiffParseError, PathTraversalError) as exc:
                return AllowlistVerdict(
                    passed=False,
                    violation="patch_apply_failed",
                    reason=f"candidate diff does not apply cleanly to the repo: {exc}",
                )

            any_real_change = False
            for block in blocks:
                original_path = repo_root / block.path
                after_path = verification_root / block.path
                original_text = (
                    original_path.read_text(encoding="utf-8", errors="replace")
                    if original_path.exists() and original_path.is_file()
                    else None
                )
                after_text = (
                    after_path.read_text(encoding="utf-8", errors="replace")
                    if after_path.exists() and after_path.is_file()
                    else None
                )
                if original_text != after_text:
                    any_real_change = True
                    break

            if not any_real_change:
                return AllowlistVerdict(
                    passed=False,
                    violation="no_op_candidate",
                    reason="candidate diff produces no net change versus the original file content",
                )
    except ScratchCopyError as exc:
        return AllowlistVerdict(
            passed=False,
            violation="patch_apply_failed",
            reason=f"could not create a scratch copy to verify the candidate diff: {exc}",
        )

    file_type_violation = _check_file_types(blocks)
    if file_type_violation is not None:
        return file_type_violation

    caps_violation = _check_hard_caps(blocks, caps)
    if caps_violation is not None:
        return caps_violation

    try:
        pr_blocks = parse_diff(pr_diff) if pr_diff.strip() else ()
    except DiffParseError:
        # A malformed PR diff can never authorize a candidate deletion; fail
        # closed by treating it as if the PR diff added nothing at all.
        pr_blocks = ()
    pr_added_lines_by_path = {block.path: block.added_lines() for block in pr_blocks}

    restore_violation = _check_restore_only_sql_deletions(blocks, pr_added_lines_by_path)
    if restore_violation is not None:
        return restore_violation

    weakening_violation = _check_test_weakening(blocks, failure_kind, failing_checks)
    if weakening_violation is not None:
        return weakening_violation

    return AllowlistVerdict(passed=True, violation=None, reason="allowlist checks passed")
