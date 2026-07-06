"""Path-safe, read-only repository access.

`RepoTools` is the *only* way anything in this package (model or otherwise)
is ever allowed to look at file content or file names inside a checkout. It
exposes exactly two operations -- read a file, search by glob pattern --
both scoped strictly to a single, fixed repository root and both enforcing
containment via `dbt_fixer.pathsafe.resolve_within_root`.

There is no write, create, delete, or rename method anywhere on this class,
and there never will be: structured file changes only ever happen through
`dbt_fixer.applier`, operating on an isolated scratch copy the model never
has direct tool access to. This asymmetry -- read/search tools for the
model, a separate, non-model-invoked applier for writes -- is the whole
point of this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ..pathsafe import PathTraversalError, resolve_within_root

__all__ = [
    "PathTraversalError",
    "RepoFileNotFoundError",
    "RepoIsADirectoryError",
    "RepoTools",
]

DEFAULT_MAX_READ_BYTES = 1_000_000
DEFAULT_MAX_SEARCH_RESULTS = 500


class RepoFileNotFoundError(FileNotFoundError):
    """Raised when a requested (in-bounds) path does not exist."""


class RepoIsADirectoryError(IsADirectoryError):
    """Raised when a requested (in-bounds) path is a directory, not a file."""


class RepoTools:
    """File-read/search tools scoped to a single repository root.

    Every public method accepts only repo-root-relative path/pattern
    strings. Containment is enforced against the *resolved* (symlinks
    followed) real path, so a symlink inside the repo that points outside
    it is rejected exactly like a literal `../` traversal would be.
    """

    def __init__(self, repo_root: "str | Path") -> None:
        """Construct a toolkit rooted at `repo_root`.

        Raises:
            FileNotFoundError: If `repo_root` does not exist.
            NotADirectoryError: If `repo_root` exists but is not a directory.
        """

        root = Path(repo_root)
        if not root.exists():
            raise FileNotFoundError(f"repo root does not exist: {repo_root!r}")
        if not root.is_dir():
            raise NotADirectoryError(f"repo root is not a directory: {repo_root!r}")
        self._root = root.resolve(strict=True)

    @property
    def root(self) -> Path:
        """The resolved, real repository root path."""

        return self._root

    def read_file(self, relative_path: str, max_bytes: int = DEFAULT_MAX_READ_BYTES) -> str:
        """Read and return the UTF-8 text content of a file within the repo.

        Args:
            relative_path: Path relative to the repo root, e.g.
                `"models/staging/stg_customers.sql"`. Absolute paths, `..`
                traversal, and symlinks resolving outside the root are all
                rejected before any file access is attempted.
            max_bytes: Safety cap on how much of the file is read.

        Returns:
            The file's text content (decoding errors are replaced, never
            raised, so a binary or non-UTF-8 file degrades to a readable
            approximation rather than crashing the caller).

        Raises:
            PathTraversalError: If the path resolves outside the repo root.
            RepoFileNotFoundError: If the (in-bounds) path does not exist.
            RepoIsADirectoryError: If the (in-bounds) path is a directory.
        """

        resolved = resolve_within_root(self._root, relative_path)

        if not resolved.exists():
            raise RepoFileNotFoundError(f"no such file: {relative_path!r}")
        if resolved.is_dir():
            raise RepoIsADirectoryError(f"path is a directory, not a file: {relative_path!r}")

        with resolved.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read(max_bytes)

    def search_files(
        self,
        pattern: str,
        relative_dir: str = ".",
        max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    ) -> Tuple[str, ...]:
        """Search for files matching `pattern` under `relative_dir`.

        `pattern` is a glob pattern matched relative to `relative_dir`,
        supporting `**` for recursive matching (e.g. `"**/*.sql"` or, when
        `relative_dir` is left at its default, the single combined pattern
        `"models/**/*.sql"`). Results are returned as repo-root-relative
        POSIX path strings, sorted for determinism.

        A glob match that turns out to be a symlink resolving outside the
        repo root is silently excluded from the results (never surfaced),
        exactly like a legitimately-absent file would be -- this is a
        different, more permissive failure mode than `read_file`'s, because
        `relative_dir`/`pattern` themselves (the caller-controlled inputs)
        are still fully validated and raise immediately if they attempt
        traversal; only individual matches discovered *during* enumeration
        are filtered rather than aborting the whole search.

        Args:
            pattern: A glob pattern, e.g. `"*.sql"` or `"models/**/*.sql"`.
            relative_dir: Directory (relative to the repo root) to search
                under. Defaults to the repo root itself.
            max_results: Safety cap on the number of matches returned.

        Raises:
            PathTraversalError: If `relative_dir` or `pattern` is absolute,
                contains a `..` component, or `relative_dir` resolves
                outside the repo root.
            RepoFileNotFoundError: If `relative_dir` does not exist.
            NotADirectoryError: If `relative_dir` is not a directory.
        """

        base = resolve_within_root(self._root, relative_dir)

        if not base.exists():
            raise RepoFileNotFoundError(f"no such directory: {relative_dir!r}")
        if not base.is_dir():
            raise NotADirectoryError(f"not a directory: {relative_dir!r}")

        _validate_pattern(pattern)

        matches: list[str] = []
        try:
            candidates = sorted(base.glob(pattern))
        except (ValueError, OSError):
            # A malformed glob pattern (e.g. unbalanced brackets) yields no
            # matches rather than propagating an implementation-specific
            # exception to the caller.
            return ()

        for candidate in candidates:
            if not candidate.is_file():
                continue
            # Re-resolve and re-check containment for every match: guards
            # against a symlink planted *under* an otherwise-legitimate
            # directory that points outside the repo root.
            try:
                real = candidate.resolve()
            except OSError:
                continue
            if real != self._root and self._root not in real.parents:
                continue
            matches.append(real.relative_to(self._root).as_posix())
            if len(matches) >= max_results:
                break

        return tuple(sorted(set(matches)))


def _validate_pattern(pattern: str) -> None:
    """Reject a glob pattern that itself attempts traversal or is absolute."""

    if not isinstance(pattern, str) or pattern.strip() == "":
        raise PathTraversalError(f"pattern must be a non-empty string, got {pattern!r}")

    candidate = Path(pattern)
    if candidate.is_absolute():
        raise PathTraversalError(f"absolute glob patterns are not allowed: {pattern!r}")
    if ".." in candidate.parts:
        raise PathTraversalError(f"glob patterns may not contain '..': {pattern!r}")
