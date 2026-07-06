"""Tests for `dbt_fixer.pathsafe.resolve_within_root`.

Covers the success path (a plain in-bounds relative path resolves cleanly)
and the three distinct rejection paths: `..` traversal, absolute paths, and
a symlink planted inside the root that points outside it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dbt_fixer.pathsafe import PathTraversalError, resolve_within_root


def test_resolves_in_bounds_relative_path(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    target = tmp_path / "models" / "stg_customers.sql"
    target.write_text("select 1", encoding="utf-8")

    resolved = resolve_within_root(tmp_path, "models/stg_customers.sql")

    assert resolved == target.resolve()


def test_rejects_dotdot_traversal(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "../etc/passwd")


def test_rejects_dotdot_traversal_embedded_in_middle(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "models/../../etc/passwd")


def test_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "/etc/passwd")


def test_rejects_empty_or_non_string_path(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "")

    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, "   ")

    with pytest.raises(PathTraversalError):
        resolve_within_root(tmp_path, None)  # type: ignore[arg-type]


@pytest.mark.skipif(os.name == "nt", reason="symlinks require elevated privileges on Windows")
def test_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")

    root = tmp_path / "repo"
    root.mkdir()
    escape_link = root / "escape.sql"
    escape_link.symlink_to(outside)

    with pytest.raises(PathTraversalError):
        resolve_within_root(root, "escape.sql")
