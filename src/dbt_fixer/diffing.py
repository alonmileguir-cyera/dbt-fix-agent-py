"""Pure-Python unified diff generation matching `git diff` semantics.

No subprocess and no real `git` binary is ever invoked here --
`difflib.unified_diff` alone drives every diff this package produces, which
keeps diff generation fully deterministic and offline-testable (verified by
a dedicated `tests/real_process` test that compares this module's output
against a real `git diff` on the same before/after tree, modulo the
`diff --git`/`index` header lines this module does not attempt to
replicate byte-for-byte).

Three change kinds are handled, matching git's own conventions:

- **added**: no "before" file exists; the diff's `from` side is `/dev/null`.
- **deleted**: no "after" file exists; the diff's `to` side is `/dev/null`.
- **modified**: both sides exist with different content.

A path whose before/after content is identical produces no output at all
-- exactly like `git diff` silently omitting unchanged files.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional

ChangeKind = Literal["added", "deleted", "modified"]

_DEV_NULL = "/dev/null"
_NO_NEWLINE_MARKER = r"\ No newline at end of file"


@dataclass(frozen=True)
class FileDiff:
    """One file's unified diff, including its synthetic `diff --git` header."""

    path: str
    change_kind: ChangeKind
    unified_diff: str


def _read_or_none(path: Path) -> Optional[str]:
    """Return `path`'s UTF-8 text content, or `None` if it doesn't exist as a file."""

    if not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def diff_one_file(
    before_text: Optional[str], after_text: Optional[str], relative_path: str
) -> Optional[FileDiff]:
    """Compute the unified diff for one file given its before/after content.

    Args:
        before_text: The file's content before the change, or `None` if the
            file did not exist (an "added" file).
        after_text: The file's content after the change, or `None` if the
            file no longer exists (a "deleted" file).
        relative_path: The repo-relative path, used for the `a/`/`b/`
            diff headers.

    Returns:
        `None` if `before_text == after_text` (no change to report).
        Otherwise a `FileDiff` whose `unified_diff` text starts with a
        synthetic `diff --git a/{path} b/{path}` header line followed by
        the real `difflib.unified_diff` hunk body.
    """

    if before_text == after_text:
        return None

    if before_text is None:
        change_kind: ChangeKind = "added"
    elif after_text is None:
        change_kind = "deleted"
    else:
        change_kind = "modified"

    before_lines = before_text.splitlines(keepends=True) if before_text is not None else []
    after_lines = after_text.splitlines(keepends=True) if after_text is not None else []

    from_file = _DEV_NULL if before_text is None else f"a/{relative_path}"
    to_file = _DEV_NULL if after_text is None else f"b/{relative_path}"

    hunk_lines = list(
        difflib.unified_diff(before_lines, after_lines, fromfile=from_file, tofile=to_file)
    )
    # ``difflib`` preserves a source line's missing trailing newline by
    # returning a hunk entry without ``\n``. Joining those entries directly
    # would concatenate the following hunk line (for example ``-old+new``),
    # which is not a valid unified diff. Render the same explicit marker Git
    # uses so the parser can preserve each side's EOF-newline state.
    hunk_text = "".join(
        line
        if line.endswith("\n")
        else f"{line}\n{_NO_NEWLINE_MARKER}\n"
        for line in hunk_lines
    )
    header = f"diff --git a/{relative_path} b/{relative_path}\n"
    return FileDiff(path=relative_path, change_kind=change_kind, unified_diff=header + hunk_text)


def generate_unified_diff(
    before_root: "str | Path", after_root: "str | Path", changed_paths: Iterable[str]
) -> str:
    """Produce the full, multi-file unified diff for `changed_paths`.

    Paths are diffed in sorted, de-duplicated order, so repeated runs over
    the same before/after trees always produce byte-identical output. A
    path whose content is unchanged between `before_root` and `after_root`
    is silently skipped -- no `diff --git` block is emitted for it -- even
    if it appears in `changed_paths`.

    Args:
        before_root: Root of the original (untouched) checkout.
        after_root: Root of the scratch copy after edits were applied.
        changed_paths: Repo-relative paths to consider for diffing.

    Returns:
        The concatenated unified diff text for every path with an actual
        content difference. An empty string if nothing changed.
    """

    before_root = Path(before_root)
    after_root = Path(after_root)

    rendered: List[str] = []
    for relative_path in sorted(set(changed_paths)):
        before_text = _read_or_none(before_root / relative_path)
        after_text = _read_or_none(after_root / relative_path)
        file_diff = diff_one_file(before_text, after_text, relative_path)
        if file_diff is not None:
            rendered.append(file_diff.unified_diff)

    return "".join(rendered)
