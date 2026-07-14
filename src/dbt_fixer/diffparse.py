"""Pure-Python parsing and application of the unified-diff text this package produces.

`dbt_fixer.diffing` generates candidate-fix diffs (and the untrusted PR diff
arrives as plain text too) using nothing but `difflib.unified_diff` -- no
real `git` binary, no subprocess. This module is the mirror-image reader:
it parses that same, narrow diff dialect back into structured
`FileDiffBlock` objects, and can re-apply those blocks onto a directory
tree. Nothing here shells out to `git apply` or any other subprocess --
parsing and application are both pure Python, which keeps this module
fully offline-testable like the rest of the package.

Two independent consumers drive this module's shape:

- The allowlist gate (`dbt_fixer.allowlist`) needs to inspect a diff's
  added/removed lines per file (e.g. to check whether a SQL deletion in a
  candidate diff merely restores a line the *original PR diff* itself
  deleted) without mutating anything on disk.
- The retry loop needs to materialize a candidate diff onto a fresh
  scratch copy of the repo (so the re-audit gate's sibling `dbt_auditor`
  subprocess can be pointed at a real, patched checkout) -- this is a
  distinct code path from `dbt_fixer.applier.apply_proposal`, which
  applies a *structured* `Proposal`'s edits directly, never a literal
  unified-diff text.

**Fail-closed application.** Like `dbt_fixer.applier`, `apply_diff`
validates that every parsed file block applies cleanly against the
current on-disk content -- context and removed lines must match
byte-for-byte at the position the hunk claims -- *before* any file is
mutated. A diff with one hunk that does not apply cleanly is rejected in
its entirety; there is no partial-application outcome.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

from .pathsafe import resolve_within_root

__all__ = [
    "DiffParseError",
    "PatchApplyError",
    "HunkLineKind",
    "ChangeKind",
    "HunkLine",
    "Hunk",
    "FileDiffBlock",
    "parse_diff",
    "apply_file_diff",
    "apply_diff",
]

HunkLineKind = Literal["context", "added", "removed"]
ChangeKind = Literal["added", "deleted", "modified"]

_DEV_NULL = "/dev/null"
_NO_NEWLINE_MARKER = r"\ No newline at end of file"

_FILE_HEADER_RE = re.compile(r"^diff --git a/(?P<a>.+) b/(?P<b>.+)$")
_FROM_FILE_RE = re.compile(r"^--- (?P<path>.+)$")
_TO_FILE_RE = re.compile(r"^\+\+\+ (?P<path>.+)$")
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


class DiffParseError(ValueError):
    """Raised when text does not parse as this package's unified-diff dialect.

    Never raised by `dbt_fixer.diffing` itself (which only ever produces
    well-formed output); this exception exists for callers parsing
    diff text of unknown provenance.
    """


class PatchApplyError(RuntimeError):
    """Raised when a parsed diff does not apply cleanly to the given content.

    Always raised *before* any file is mutated (see module docstring);
    catching this at a call site is always safe in the sense that no
    partial write has occurred.
    """


@dataclass(frozen=True)
class HunkLine:
    """One line inside a hunk body, tagged with its unified-diff role.

    `text` is the line's content *with* its trailing newline, exactly as
    it appeared in the file (matching `str.splitlines(keepends=True)`
    semantics) -- except for the final line of a file lacking a trailing
    newline, where `text` has none either.
    """

    kind: HunkLineKind
    text: str


@dataclass(frozen=True)
class Hunk:
    """One `@@ -old_start,old_count +new_start,new_count @@` hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: Tuple[HunkLine, ...]


@dataclass(frozen=True)
class FileDiffBlock:
    """One file's parsed diff: its change kind and every hunk touching it."""

    path: str
    change_kind: ChangeKind
    hunks: Tuple[Hunk, ...]

    def removed_lines(self) -> Tuple[str, ...]:
        """Every line this diff removes from `path`, in hunk order, newline stripped."""

        return tuple(
            line.text.rstrip("\n")
            for hunk in self.hunks
            for line in hunk.lines
            if line.kind == "removed"
        )

    def added_lines(self) -> Tuple[str, ...]:
        """Every line this diff adds to `path`, in hunk order, newline stripped."""

        return tuple(
            line.text.rstrip("\n")
            for hunk in self.hunks
            for line in hunk.lines
            if line.kind == "added"
        )


_GIT_METADATA_RE = re.compile(
    r"^(index [0-9a-f]+\.\.[0-9a-f]+( \d+)?"
    r"|old mode \d+"
    r"|new mode \d+"
    r"|new file mode \d+"
    r"|deleted file mode \d+"
    r"|similarity index \d+%"
    r"|dissimilarity index \d+%"
    r"|rename (from|to) .*"
    r"|copy (from|to) .*"
    r"|Binary files .*)$"
)


def _strip_ab_prefix(path: str) -> str:
    """Strip a leading `a/` or `b/` prefix from a diff header path, if present."""

    if path in (_DEV_NULL,):
        return path
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _parse_hunk_body(lines: List[str], index: int, old_count: int, new_count: int, path: str) -> Tuple[Tuple[HunkLine, ...], int]:
    """Parse one hunk's body lines starting at `lines[index]`.

    Returns the parsed `HunkLine` tuple and the index of the first line
    after the hunk body. Raises `DiffParseError` if the body's actual
    old-side/new-side line counts do not match the hunk header's declared
    counts, or if a body line has no recognized prefix.
    """

    body: List[HunkLine] = []
    old_seen = 0
    new_seen = 0
    while True:
        if index >= len(lines):
            if old_seen == old_count and new_seen == new_count:
                break
            raise DiffParseError(f"hunk body for {path!r} ended before its declared line counts")
        raw = lines[index]
        if raw == _NO_NEWLINE_MARKER:
            if not body:
                raise DiffParseError(
                    f"no-newline marker in {path!r} does not follow a hunk body line"
                )
            previous = body[-1]
            if not previous.text.endswith("\n"):
                raise DiffParseError(
                    f"duplicate no-newline marker for a hunk body line in {path!r}"
                )
            body[-1] = HunkLine(kind=previous.kind, text=previous.text[:-1])
            index += 1
            continue
        if old_seen == old_count and new_seen == new_count:
            break
        if raw.startswith("@@ ") or raw.startswith("diff --git "):
            raise DiffParseError(
                f"hunk body for {path!r} ended before its declared line counts"
            )
        if raw.startswith(" "):
            body.append(HunkLine(kind="context", text=raw[1:] + "\n"))
            old_seen += 1
            new_seen += 1
        elif raw.startswith("-"):
            body.append(HunkLine(kind="removed", text=raw[1:] + "\n"))
            old_seen += 1
        elif raw.startswith("+"):
            body.append(HunkLine(kind="added", text=raw[1:] + "\n"))
            new_seen += 1
        elif raw == "":
            # A blank line with no prefix at all only ever occurs as the
            # trailing separator difflib.unified_diff emits between file
            # blocks; it is never part of a hunk body.
            break
        else:
            raise DiffParseError(f"unrecognized hunk body line in {path!r}: {raw!r}")
        index += 1

    if old_seen != old_count or new_seen != new_count:
        raise DiffParseError(
            f"hunk body for {path!r} declares -{old_count}/+{new_count} lines "
            f"but contains -{old_seen}/+{new_seen}"
        )

    return tuple(body), index


def parse_diff(diff_text: str) -> Tuple[FileDiffBlock, ...]:
    """Parse `diff_text` -- this package's `dbt_fixer.diffing` dialect -- into blocks.

    Args:
        diff_text: The concatenated multi-file unified diff text, exactly
            as `dbt_fixer.diffing.generate_unified_diff` produces it (a
            `diff --git a/{path} b/{path}` header per file, followed by
            `--- `/`+++ ` file lines and one or more `@@ ... @@` hunks).
            An empty string parses to an empty tuple.

    Returns:
        One `FileDiffBlock` per `diff --git` block, in the order they
        appear in `diff_text`.

    Raises:
        DiffParseError: If `diff_text` does not conform to this dialect
            (a missing header, an unparseable hunk header, a hunk body
            whose actual line counts disagree with its declared counts,
            or any other structural mismatch).
    """

    if diff_text.strip() == "":
        return ()

    # difflib.unified_diff's own lines never end in a trailing newline
    # (each entry already carries its own "\n"), but the concatenated,
    # multi-block text this module receives is line-oriented, so splitting
    # on "\n" and dropping a single trailing empty element (from the final
    # line's own newline) reconstructs the original logical lines exactly.
    raw_lines = diff_text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines = raw_lines[:-1]

    blocks: List[FileDiffBlock] = []
    index = 0
    while index < len(raw_lines):
        header_match = _FILE_HEADER_RE.match(raw_lines[index])
        if header_match is None:
            raise DiffParseError(f"expected a 'diff --git' header, got: {raw_lines[index]!r}")
        path = header_match.group("b")
        index += 1

        # Real `git diff` output carries metadata lines between the
        # `diff --git` header and the `---` line (index/mode/rename/
        # similarity/Binary). This package's own differ never emits them,
        # but the PR diff handed in from CI does (first hit live:
        # bi-dbt #2533 round 5, `index e6a745c2..11a2589e 100644`).
        # Skip them; a pure-rename/binary block with no hunks is handled
        # by the no-'---' branch below.
        while index < len(raw_lines) and _GIT_METADATA_RE.match(raw_lines[index]):
            index += 1

        if index >= len(raw_lines):
            raise DiffParseError(f"diff header for {path!r} has no '---'/'+++' lines")
        from_match = _FROM_FILE_RE.match(raw_lines[index])
        if from_match is None:
            raise DiffParseError(f"expected a '--- ' line for {path!r}, got: {raw_lines[index]!r}")
        from_file = _strip_ab_prefix(from_match.group("path"))
        index += 1

        if index >= len(raw_lines):
            raise DiffParseError(f"diff header for {path!r} has no '+++' line")
        to_match = _TO_FILE_RE.match(raw_lines[index])
        if to_match is None:
            raise DiffParseError(f"expected a '+++ ' line for {path!r}, got: {raw_lines[index]!r}")
        to_file = _strip_ab_prefix(to_match.group("path"))
        index += 1

        if from_file == _DEV_NULL:
            change_kind: ChangeKind = "added"
        elif to_file == _DEV_NULL:
            change_kind = "deleted"
        else:
            change_kind = "modified"

        hunks: List[Hunk] = []
        while index < len(raw_lines) and raw_lines[index].startswith("@@ "):
            hunk_match = _HUNK_HEADER_RE.match(raw_lines[index])
            if hunk_match is None:
                raise DiffParseError(f"unparseable hunk header for {path!r}: {raw_lines[index]!r}")
            old_start = int(hunk_match.group("old_start"))
            old_count = int(hunk_match.group("old_count") or "1")
            new_start = int(hunk_match.group("new_start"))
            new_count = int(hunk_match.group("new_count") or "1")
            index += 1

            body, index = _parse_hunk_body(raw_lines, index, old_count, new_count, path)
            hunks.append(
                Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=body,
                )
            )

        # Skip any blank separator line(s) between file blocks.
        while index < len(raw_lines) and raw_lines[index] == "":
            index += 1

        blocks.append(FileDiffBlock(path=path, change_kind=change_kind, hunks=tuple(hunks)))

    return tuple(blocks)


def apply_file_diff(before_text: Optional[str], block: FileDiffBlock) -> Optional[str]:
    """Apply one `FileDiffBlock` to its file's prior content.

    Args:
        before_text: The file's current content, or `None` if the file
            does not currently exist. Required (non-`None`) for
            `"modified"` blocks; ignored for `"added"` blocks.
        block: The parsed diff block to apply.

    Returns:
        The file's new content, or `None` to signal the file should be
        deleted (for a `"deleted"` block).

    Raises:
        PatchApplyError: If `block.change_kind == "modified"` and
            `before_text` is `None`, or if any hunk's context/removed
            lines do not match `before_text` at the position the hunk
            claims.
    """

    if block.change_kind == "added":
        return "".join(
            line.text for hunk in block.hunks for line in hunk.lines if line.kind != "removed"
        )

    if block.change_kind == "deleted":
        return None

    if before_text is None:
        raise PatchApplyError(
            f"cannot apply a 'modified' diff block for {block.path!r}: no prior content given"
        )

    lines = before_text.splitlines(keepends=True)
    result: List[str] = []
    cursor = 0  # 0-indexed offset into `lines` already copied into `result`

    for hunk in block.hunks:
        start = hunk.old_start - 1
        if start < cursor:
            raise PatchApplyError(
                f"hunks for {block.path!r} are out of order or overlap at line {hunk.old_start}"
            )
        result.extend(lines[cursor:start])
        pos = start
        for hunk_line in hunk.lines:
            if hunk_line.kind == "context":
                if pos >= len(lines) or lines[pos] != hunk_line.text:
                    raise PatchApplyError(
                        f"context mismatch for {block.path!r} at line {pos + 1}"
                    )
                result.append(lines[pos])
                pos += 1
            elif hunk_line.kind == "removed":
                if pos >= len(lines) or lines[pos] != hunk_line.text:
                    raise PatchApplyError(
                        f"removed-line mismatch for {block.path!r} at line {pos + 1}"
                    )
                pos += 1
            else:  # "added"
                result.append(hunk_line.text)
        cursor = pos

    result.extend(lines[cursor:])
    return "".join(result)


def apply_diff(root: "str | Path", diff_text: str) -> Tuple[str, ...]:
    """Apply every file block in `diff_text` onto the directory tree at `root`.

    Fail-closed and two-phase, mirroring `dbt_fixer.applier.apply_proposal`:
    every block's new content is computed first (raising `DiffParseError`
    or `PatchApplyError` without touching disk if anything fails), and only
    once every block is known to apply cleanly are any files actually
    written or deleted.

    Args:
        root: The directory tree to apply the diff onto (typically a
            throwaway scratch copy).
        diff_text: The unified diff text to parse and apply.

    Returns:
        The sorted tuple of repo-relative paths that were changed
        (created, modified, or deleted).

    Raises:
        DiffParseError: If `diff_text` does not parse.
        PatchApplyError: If any block does not apply cleanly.
        dbt_fixer.pathsafe.PathTraversalError: If any block's path
            attempts traversal or resolves outside `root`.
    """

    root = Path(root)
    blocks = parse_diff(diff_text)

    planned: List[Tuple[Path, Optional[str]]] = []
    for block in blocks:
        target = resolve_within_root(root, block.path)
        before_text: Optional[str] = None
        if target.exists() and target.is_file():
            before_text = target.read_text(encoding="utf-8", errors="replace")
        after_text = apply_file_diff(before_text, block)
        planned.append((target, after_text))

    for target, after_text in planned:
        if after_text is None:
            if target.exists():
                target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(after_text, encoding="utf-8", newline="")

    return tuple(sorted(block.path for block in blocks))


def invert_diff(diff_text: str) -> str:
    """Render the inverse of `diff_text`: applying the result to a tree in
    the diff's AFTER state reconstructs its BEFORE state.

    Structural, not textual: parse, then per file swap the a/b sides
    (added <-> deleted), per hunk swap the old/new ranges, and per line
    swap added <-> removed while preserving in-hunk order (the standard
    unified-diff inversion).

    Raises `DiffParseError` if `diff_text` does not parse.
    """

    out: List[str] = []
    for block in parse_diff(diff_text):
        path = block.path
        out.append(f"diff --git a/{path} b/{path}\n")
        if block.change_kind == "added":
            out.append(f"--- a/{path}\n")
            out.append("+++ /dev/null\n")
        elif block.change_kind == "deleted":
            out.append("--- /dev/null\n")
            out.append(f"+++ b/{path}\n")
        else:
            out.append(f"--- a/{path}\n")
            out.append(f"+++ b/{path}\n")
        for hunk in block.hunks:
            out.append(
                f"@@ -{hunk.new_start},{hunk.new_count} +{hunk.old_start},{hunk.old_count} @@\n"
            )
            for line in hunk.lines:
                if line.kind == "added":
                    prefix = "-"
                elif line.kind == "removed":
                    prefix = "+"
                else:
                    prefix = " "
                out.append(prefix + line.text)
                if not line.text.endswith("\n"):
                    out.append(f"\n{_NO_NEWLINE_MARKER}\n")
    return "".join(out)


def _diff_paths(diff_text: str) -> Iterable[str]:
    """Yield every file path touched by `diff_text`, without applying anything."""

    for block in parse_diff(diff_text):
        yield block.path
