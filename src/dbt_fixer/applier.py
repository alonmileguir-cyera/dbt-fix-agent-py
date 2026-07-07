"""Fail-closed application of a structured `Proposal` onto a scratch copy.

This is the *only* place a `dbt_fixer.proposal.Proposal`'s edits are ever
turned into real file mutations, and it only ever operates on an isolated
scratch copy (see `dbt_fixer.scratch.scratch_copy`) -- never on the
original checkout the model read from. No function in this module is ever
handed to a model as a tool; the model only ever *proposes* edits (via
`dbt_fixer.proposal`), it never applies them.

**Fail-closed, two-phase application.** Every edit in a proposal is
validated -- target exists, target is a file (not a directory), every
line-range edit's bounds are in range for its target's current line count,
and no two edits conflict (two edits targeting the same whole file, or two
line-range edits on the same file with overlapping ranges) -- *before* a
single byte of any file is mutated. A proposal with one invalid or
conflicting edit is rejected in its entirety: this module never applies a
"best effort" subset of a proposal's edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .pathsafe import resolve_within_root
from .proposal import Edit, Proposal

__all__ = [
    "ApplyError",
    "EditTargetNotFoundError",
    "EditTargetIsDirectoryError",
    "InvalidLineRangeError",
    "ConflictingEditsError",
    "AppliedProposal",
    "apply_proposal",
]


class ApplyError(RuntimeError):
    """Base class for anything that prevents a proposal from being applied.

    Every subclass is raised only *before* any file has been mutated (see
    module docstring); catching `ApplyError` at a call site is always safe
    in the sense that the scratch copy is guaranteed untouched.
    """


class EditTargetAlreadyExistsError(ApplyError):
    """A `create_file` edit targets a path that already exists."""


class EditTargetNotFoundError(ApplyError):
    """An edit's `path` does not exist in the scratch copy."""


class EditTargetIsDirectoryError(ApplyError):
    """An edit's `path` resolves to a directory, not a file."""


class InvalidLineRangeError(ApplyError):
    """A `line_range_edit`'s `start_line`/`end_line` is out of bounds for its target."""


class ConflictingEditsError(ApplyError):
    """Two or more edits in the same proposal conflict with each other."""


@dataclass(frozen=True)
class AppliedProposal:
    """The outcome of successfully applying every edit in a `Proposal`."""

    proposal: Proposal
    changed_paths: Tuple[str, ...]


def _resolved_target(scratch_root: Path, edit: Edit) -> Path:
    """Resolve one edit's target path within `scratch_root`, validating it exists and is a file.

    Path traversal (`..`, absolute paths, a symlink escaping `scratch_root`)
    raises `dbt_fixer.pathsafe.PathTraversalError`, propagated unchanged --
    the same containment guard used by `RepoTools` is reused here rather
    than duplicated.
    """

    resolved = resolve_within_root(scratch_root, edit.path)
    if edit.kind == "create_file":
        if resolved.exists():
            raise EditTargetAlreadyExistsError(
                f"create_file target already exists: {edit.path!r}"
            )
        return resolved
    if not resolved.exists():
        raise EditTargetNotFoundError(f"edit target does not exist: {edit.path!r}")
    if resolved.is_dir():
        raise EditTargetIsDirectoryError(f"edit target is a directory, not a file: {edit.path!r}")
    return resolved


def _check_conflicts(edits_by_path: Dict[str, List[Edit]]) -> None:
    """Raise `ConflictingEditsError` if any path has mutually-conflicting edits.

    Conflict rules, applied per path:

    - More than one `whole_file_replace` edit for the same path conflicts.
    - A `whole_file_replace` edit alongside any `line_range_edit` for the
      same path conflicts (the whole-file edit makes any line-range edit's
      line numbers meaningless).
    - Two `line_range_edit` edits for the same path whose `[start_line,
      end_line]` (inclusive) ranges overlap conflict.
    """

    for path, edits in edits_by_path.items():
        whole_file_edits = [e for e in edits if e.kind == "whole_file_replace"]
        line_range_edits = [e for e in edits if e.kind == "line_range_edit"]
        create_edits = [e for e in edits if e.kind == "create_file"]

        if create_edits and (len(edits) > 1):
            raise ConflictingEditsError(
                f"a create_file edit must be the only edit for its path: {path!r}"
            )

        if len(whole_file_edits) > 1:
            raise ConflictingEditsError(
                f"multiple whole_file_replace edits target the same path: {path!r}"
            )
        if whole_file_edits and line_range_edits:
            raise ConflictingEditsError(
                f"a whole_file_replace and a line_range_edit both target the same path: {path!r}"
            )

        sorted_ranges = sorted(line_range_edits, key=lambda e: e.start_line)  # type: ignore[arg-type]
        for first, second in zip(sorted_ranges, sorted_ranges[1:]):
            if second.start_line <= first.end_line:  # type: ignore[operator]
                raise ConflictingEditsError(
                    f"overlapping line_range_edit ranges for path {path!r}: "
                    f"[{first.start_line}, {first.end_line}] and "
                    f"[{second.start_line}, {second.end_line}]"
                )


def _validate_line_range(edit: Edit, target: Path) -> None:
    """Raise `InvalidLineRangeError` if `edit`'s line range is out of bounds for `target`."""

    text = target.read_text(encoding="utf-8", errors="replace")
    line_count = len(text.splitlines())
    assert edit.start_line is not None and edit.end_line is not None  # guaranteed by parse_proposal
    if edit.start_line > line_count or edit.end_line > line_count:
        raise InvalidLineRangeError(
            f"line range [{edit.start_line}, {edit.end_line}] is out of bounds for "
            f"{edit.path!r}, which has {line_count} line(s)"
        )


def _apply_create_file(target: Path, edit: Edit) -> None:
    assert edit.content is not None  # guaranteed by parse_proposal
    # Parent dirs are descendants of the already-containment-checked target,
    # so creating them cannot escape the scratch root.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(edit.content, encoding="utf-8", newline="")


def _apply_whole_file_replace(target: Path, edit: Edit) -> None:
    assert edit.content is not None  # guaranteed by parse_proposal
    target.write_text(edit.content, encoding="utf-8", newline="")


def _apply_line_range_edits(target: Path, edits: List[Edit]) -> None:
    """Apply every `line_range_edit` targeting one file, in a single pass.

    Edits are applied from the bottom of the file upward (descending
    `start_line`) so that splicing one edit's replacement never shifts the
    absolute line numbers the remaining (already-validated, non-overlapping)
    edits were computed against.
    """

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    for edit in sorted(edits, key=lambda e: e.start_line, reverse=True):  # type: ignore[arg-type]
        assert edit.start_line is not None and edit.end_line is not None
        assert edit.replacement is not None
        lines[edit.start_line - 1 : edit.end_line] = [edit.replacement]

    target.write_text("".join(lines), encoding="utf-8", newline="")


def apply_proposal(scratch_root: "str | Path", proposal: Proposal) -> AppliedProposal:
    """Apply every edit in `proposal` to the scratch copy rooted at `scratch_root`.

    Two-phase, fail-closed: every edit is fully validated (target exists,
    target is a file, line ranges are in bounds, no edit conflicts with
    another) before any file is mutated. If validation fails for *any*
    edit, an `ApplyError` subclass is raised and the scratch copy is left
    completely untouched -- there is no partial-application outcome.

    Args:
        scratch_root: Root of an isolated, writable scratch copy (see
            `dbt_fixer.scratch.scratch_copy`). Never the original checkout.
        proposal: A validated `dbt_fixer.proposal.Proposal`.

    Returns:
        An `AppliedProposal` naming every path that was changed.

    Raises:
        dbt_fixer.pathsafe.PathTraversalError: If an edit's path attempts
            traversal or resolves outside `scratch_root`.
        EditTargetNotFoundError: If an edit's path does not exist.
        EditTargetIsDirectoryError: If an edit's path is a directory.
        InvalidLineRangeError: If a `line_range_edit`'s bounds are invalid.
        ConflictingEditsError: If two or more edits conflict.
    """

    scratch_root = Path(scratch_root)

    edits_by_path: Dict[str, List[Edit]] = {}
    targets_by_path: Dict[str, Path] = {}
    for edit in proposal.edits:
        target = _resolved_target(scratch_root, edit)
        targets_by_path.setdefault(edit.path, target)
        edits_by_path.setdefault(edit.path, []).append(edit)

    _check_conflicts(edits_by_path)

    for path, edits in edits_by_path.items():
        target = targets_by_path[path]
        for edit in edits:
            if edit.kind == "line_range_edit":
                _validate_line_range(edit, target)

    # Validation is complete for every edit in the proposal; only now do we
    # mutate any file.
    for path, edits in edits_by_path.items():
        target = targets_by_path[path]
        whole_file_edits = [e for e in edits if e.kind == "whole_file_replace"]
        line_range_edits = [e for e in edits if e.kind == "line_range_edit"]
        create_edits = [e for e in edits if e.kind == "create_file"]
        if create_edits:
            _apply_create_file(target, create_edits[0])
        elif whole_file_edits:
            _apply_whole_file_replace(target, whole_file_edits[0])
        elif line_range_edits:
            _apply_line_range_edits(target, line_range_edits)

    return AppliedProposal(proposal=proposal, changed_paths=tuple(sorted(edits_by_path.keys())))
