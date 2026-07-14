"""Tests for `dbt_fixer.diffing`: unified diff generation, pure `difflib`.

Covers the modified/added/deleted change kinds, the unchanged-file no-op
case, deterministic sorted multi-file output, and offline reproducibility
(byte-identical output across repeated calls over the same trees).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dbt_fixer.diffing import diff_one_file, generate_unified_diff


def test_diff_one_file_returns_none_for_identical_content() -> None:
    assert diff_one_file("select 1\n", "select 1\n", "models/a.sql") is None


def test_diff_one_file_modified_uses_a_and_b_headers() -> None:
    result = diff_one_file("select 1\n", "select 2\n", "models/a.sql")

    assert result is not None
    assert result.change_kind == "modified"
    assert result.unified_diff.startswith("diff --git a/models/a.sql b/models/a.sql\n")
    assert "--- a/models/a.sql\n" in result.unified_diff
    assert "+++ b/models/a.sql\n" in result.unified_diff
    assert "-select 1\n" in result.unified_diff
    assert "+select 2\n" in result.unified_diff


def test_diff_one_file_added_uses_dev_null_from_side() -> None:
    result = diff_one_file(None, "select 1\n", "models/new.sql")

    assert result is not None
    assert result.change_kind == "added"
    assert "--- /dev/null\n" in result.unified_diff
    assert "+++ b/models/new.sql\n" in result.unified_diff
    assert "+select 1\n" in result.unified_diff


def test_diff_one_file_deleted_uses_dev_null_to_side() -> None:
    result = diff_one_file("select 1\n", None, "models/gone.sql")

    assert result is not None
    assert result.change_kind == "deleted"
    assert "--- a/models/gone.sql\n" in result.unified_diff
    assert "+++ /dev/null\n" in result.unified_diff
    assert "-select 1\n" in result.unified_diff


def test_diff_one_file_returns_none_when_both_sides_absent() -> None:
    assert diff_one_file(None, None, "models/never_existed.sql") is None


@pytest.mark.parametrize(
    ("before_text", "after_text"),
    [
        ("select 1", "select 2\n"),
        ("select 1\n", "select 2"),
    ],
)
def test_diff_one_file_preserves_eof_newline_transitions(
    before_text: str, after_text: str
) -> None:
    result = diff_one_file(before_text, after_text, "models/a.sql")

    assert result is not None
    assert result.unified_diff.count("\\ No newline at end of file\n") == 1
    assert "-select 1+select 2" not in result.unified_diff


def _make_tree(root: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        full = root / relative_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")


def test_generate_unified_diff_handles_mixed_add_modify_delete(tmp_path: Path) -> None:
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    _make_tree(
        before_root,
        {
            "models/a.sql": "select 1\n",
            "models/b.sql": "select 2\n",
            "models/unchanged.sql": "select 3\n",
        },
    )
    _make_tree(
        after_root,
        {
            "models/a.sql": "select 999\n",  # modified
            "models/unchanged.sql": "select 3\n",  # unchanged
            "models/new.sql": "select 4\n",  # added
        },
    )
    # models/b.sql exists only "before" -> deleted

    diff_text = generate_unified_diff(
        before_root,
        after_root,
        ["models/a.sql", "models/b.sql", "models/unchanged.sql", "models/new.sql"],
    )

    assert "diff --git a/models/a.sql b/models/a.sql" in diff_text
    assert "diff --git a/models/b.sql b/models/b.sql" in diff_text
    assert "diff --git a/models/new.sql b/models/new.sql" in diff_text
    assert "diff --git a/models/unchanged.sql" not in diff_text
    assert "+++ /dev/null" in diff_text  # b.sql deleted
    assert "--- /dev/null" in diff_text  # new.sql added


def test_generate_unified_diff_orders_paths_deterministically(tmp_path: Path) -> None:
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    _make_tree(before_root, {"z.sql": "1\n", "a.sql": "1\n"})
    _make_tree(after_root, {"z.sql": "2\n", "a.sql": "2\n"})

    diff_text = generate_unified_diff(before_root, after_root, ["z.sql", "a.sql"])

    assert diff_text.index("diff --git a/a.sql") < diff_text.index("diff --git a/z.sql")


def test_generate_unified_diff_deduplicates_repeated_paths(tmp_path: Path) -> None:
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    _make_tree(before_root, {"a.sql": "1\n"})
    _make_tree(after_root, {"a.sql": "2\n"})

    diff_text = generate_unified_diff(before_root, after_root, ["a.sql", "a.sql", "a.sql"])

    assert diff_text.count("diff --git a/a.sql b/a.sql") == 1


def test_generate_unified_diff_empty_when_nothing_changed(tmp_path: Path) -> None:
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    _make_tree(before_root, {"a.sql": "1\n"})
    _make_tree(after_root, {"a.sql": "1\n"})

    assert generate_unified_diff(before_root, after_root, ["a.sql"]) == ""


def test_generate_unified_diff_is_byte_identical_across_repeated_calls(tmp_path: Path) -> None:
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    _make_tree(before_root, {"a.sql": "select 1\n", "b.sql": "select 2\n"})
    _make_tree(after_root, {"a.sql": "select 999\n", "b.sql": "select 2\n"})

    first = generate_unified_diff(before_root, after_root, ["a.sql", "b.sql"])
    second = generate_unified_diff(before_root, after_root, ["a.sql", "b.sql"])

    assert first == second
