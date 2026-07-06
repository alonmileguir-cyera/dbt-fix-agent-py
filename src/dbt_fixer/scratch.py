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

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ScratchCopyError(RuntimeError):
    """Raised when a scratch copy cannot be created (e.g. bad source path)."""


@contextmanager
def scratch_copy(source: Path, *, prefix: str = "dbt-fixer-scratch-") -> Iterator[Path]:
    """Yield an isolated, writable copy of `source` in a fresh temp directory.

    Raises `ScratchCopyError` immediately (before creating anything) if
    `source` does not exist or is not a directory. Otherwise always cleans
    up the entire temp root on exit, regardless of how the `with` block
    exits.
    """

    source = Path(source)
    if not source.exists() or not source.is_dir():
        raise ScratchCopyError(f"scratch source {source} does not exist or is not a directory")

    tmp_root = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        dest = tmp_root / "repo"
        shutil.copytree(
            source,
            dest,
            symlinks=False,
            ignore=shutil.ignore_patterns(".git"),
        )
        yield dest
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
