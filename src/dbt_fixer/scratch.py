"""Scratch-copy lifecycle for isolated, mutation-safe repo copies.

Every structural edit this package ever makes happens on a throwaway copy of
the checkout, never on the original. `scratch_copy` is the single sanctioned
way to create that copy: it is a context manager that

1. creates a fresh, uniquely-named temporary directory,
2. deep-copies the source tree into it (excluding `.git`, both because the
   copy is for content edits and diffing, not history, and because there is
   never a reason for this package to hold a real, push-capable git working
   tree),
3. yields the path to the copy, and
4. guarantees the entire temporary directory is removed on *every* exit
   path -- normal return, an exception raised inside the `with` block, or an
   early `return`/`break` out of it -- because cleanup lives in a `finally`
   clause a context manager always runs.

Nothing in this module ever mutates `source`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ScratchCopyError(RuntimeError):
    """Raised when a scratch copy cannot be created (e.g. bad source path)."""


def _reject_symlinks(source: Path) -> None:
    """Reject repository symlinks before copying any PR-controlled content."""

    for root, directories, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        directories[:] = [name for name in directories if name != ".git"]
        for name in (*directories, *files):
            candidate = root_path / name
            if candidate.is_symlink():
                relative = candidate.relative_to(source)
                raise ScratchCopyError(
                    f"scratch source contains unsupported symlink: {relative}"
                )


@contextmanager
def scratch_copy(source: Path, *, prefix: str = "dbt-fixer-scratch-") -> Iterator[Path]:
    """Yield an isolated, writable copy of `source` in a fresh temp directory.

    Raises `ScratchCopyError` immediately (before creating anything) if
    `source` does not exist, is not a directory, or contains a symlink outside
    `.git`. Otherwise always cleans up the entire temp root on exit, regardless
    of how the `with` block exits.
    """

    source = Path(source)
    if not source.exists() or not source.is_dir():
        raise ScratchCopyError(f"scratch source {source} does not exist or is not a directory")
    _reject_symlinks(source)

    tmp_root = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        dest = tmp_root / "repo"
        shutil.copytree(
            source,
            dest,
            # Preserve any symlink introduced in the narrow interval after the
            # validation walk instead of ever dereferencing it.
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        # Close the validation/copy race: a link introduced after the source
        # walk is preserved above, then rejected here before the copy is used.
        _reject_symlinks(dest)
        yield dest
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
