"""Shared path-containment guard: reject any path that resolves outside a root.

Both the model-facing `RepoTools` read/search toolkit (`dbt_fixer.tools.repo_tools`)
and the structured-edit applier (`dbt_fixer.applier`) must apply exactly the same
containment rule to every path they are handed, so this logic is defined once,
here, rather than duplicated (and risking drift) across the two call sites.

A relative path is rejected -- before any filesystem access is attempted -- if
it is:

- not a non-empty string,
- absolute,
- contains a literal `..` path component, or
- resolves (after following any symlinks in the joined path) to a location
  that is not the root itself or a descendant of the root.

The last check is what catches a symlink planted *inside* the root that
points *outside* it: `Path.resolve()` follows symlinks, so the final
containment comparison is always against the fully-resolved, real path, not
the literal joined string.
"""

from __future__ import annotations

from pathlib import Path


class PathTraversalError(ValueError):
    """Raised when a relative path would resolve outside its sanctioned root.

    Never carries any content from the rejected path's target -- only the
    (attacker-supplied) path string itself, which is safe to surface in an
    error message or log line.
    """


def resolve_within_root(root: Path, relative_path: str) -> Path:
    """Resolve `relative_path` against `root`, guaranteeing containment.

    Args:
        root: The sanctioned root directory. Does not need to be
            pre-resolved; this function resolves it itself.
        relative_path: A path string that must be relative (not absolute)
            and must not contain a `..` component.

    Returns:
        The fully resolved (symlinks followed) absolute path, guaranteed to
        be `root` itself or a descendant of it.

    Raises:
        PathTraversalError: If `relative_path` is not a non-empty string,
            is absolute, contains a `..` component, or resolves (following
            symlinks) to a location outside `root`.
    """

    if not isinstance(relative_path, str) or relative_path.strip() == "":
        raise PathTraversalError(f"path must be a non-empty string, got {relative_path!r}")

    candidate = Path(relative_path)

    # Reject absolute paths outright: joining an absolute path onto the root
    # with `/` would silently discard the root entirely in pathlib, so this
    # must be checked before any join happens.
    if candidate.is_absolute():
        raise PathTraversalError(f"absolute paths are not allowed: {relative_path!r}")

    # Reject any literal parent-directory traversal component up front, as a
    # clear, explicit signal independent of what resolve() does.
    if ".." in candidate.parts:
        raise PathTraversalError(f"path traversal ('..') is not allowed: {relative_path!r}")

    resolved_root = Path(root).resolve()
    joined = resolved_root / candidate
    # resolve() follows symlinks (and does not require the target to exist),
    # so a symlink inside the root that points outside it is caught by the
    # containment check below.
    resolved = joined.resolve()

    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise PathTraversalError(f"path resolves outside the root: {relative_path!r}")

    return resolved
