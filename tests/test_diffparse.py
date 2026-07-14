"""Tests for `dbt_fixer.diffparse`: parsing and applying this package's diff dialect.

Round-trips every diff kind `dbt_fixer.diffing` can produce (modified,
added, deleted, multi-file) through `parse_diff`/`apply_diff`, plus the
fail-closed error paths (malformed headers, mismatched hunk counts, a
hunk that does not apply cleanly) and the `removed_lines`/`added_lines`
accessors the allowlist gate depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dbt_fixer.diffing import generate_unified_diff
from dbt_fixer.diffparse import (
    DiffParseError,
    PatchApplyError,
    apply_diff,
    apply_file_diff,
    invert_diff,
    parse_diff,
)
from dbt_fixer.pathsafe import PathTraversalError


def test_parse_diff_of_empty_text_is_empty_tuple() -> None:
    assert parse_diff("") == ()
    assert parse_diff("   \n") == ()


def test_round_trip_modified_added_deleted_multi_file(tmp_path: Path) -> None:
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    before_root.mkdir()
    after_root.mkdir()

    (before_root / "a.sql").write_text("select 1\nfrom x\nwhere y = 1\n")
    (after_root / "a.sql").write_text("select 1\nfrom x\nwhere y = 2\n")
    (after_root / "b.sql").write_text("select 2\n")
    (before_root / "c.sql").write_text("to be deleted\n")

    diff_text = generate_unified_diff(before_root, after_root, ["a.sql", "b.sql", "c.sql"])

    blocks = parse_diff(diff_text)
    by_path = {b.path: b for b in blocks}
    assert by_path["a.sql"].change_kind == "modified"
    assert by_path["a.sql"].removed_lines() == ("where y = 1",)
    assert by_path["a.sql"].added_lines() == ("where y = 2",)
    assert by_path["b.sql"].change_kind == "added"
    assert by_path["b.sql"].added_lines() == ("select 2",)
    assert by_path["c.sql"].change_kind == "deleted"
    assert by_path["c.sql"].removed_lines() == ("to be deleted",)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "a.sql").write_text("select 1\nfrom x\nwhere y = 1\n")
    (scratch / "c.sql").write_text("to be deleted\n")

    changed = apply_diff(scratch, diff_text)
    assert changed == ("a.sql", "b.sql", "c.sql")
    assert (scratch / "a.sql").read_text() == "select 1\nfrom x\nwhere y = 2\n"
    assert (scratch / "b.sql").read_text() == "select 2\n"
    assert not (scratch / "c.sql").exists()


def test_apply_file_diff_added_ignores_before_text() -> None:
    diff_text = "diff --git a/x.sql b/x.sql\n--- /dev/null\n+++ b/x.sql\n@@ -0,0 +1 @@\n+select 1\n"
    block = parse_diff(diff_text)[0]
    # Even if (incorrectly) handed stale "before" content, an "added" block
    # is reconstructed purely from its own added lines.
    assert apply_file_diff("stale content that should be ignored\n", block) == "select 1\n"


def test_apply_diff_is_two_phase_and_leaves_scratch_untouched_on_failure(tmp_path: Path) -> None:
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    before_root.mkdir()
    after_root.mkdir()
    (before_root / "a.sql").write_text("select 1\n")
    (after_root / "a.sql").write_text("select 2\n")

    diff_text = generate_unified_diff(before_root, after_root, ["a.sql"])

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # Scratch's actual content diverges from what the diff's context/removed
    # lines expect, so the hunk cannot apply cleanly.
    (scratch / "a.sql").write_text("select 999\n")

    with pytest.raises(PatchApplyError):
        apply_diff(scratch, diff_text)

    # Untouched: apply_diff must not have written anything.
    assert (scratch / "a.sql").read_text() == "select 999\n"


def test_apply_file_diff_raises_when_modified_block_has_no_prior_content() -> None:
    diff_text = "diff --git a/x.sql b/x.sql\n--- a/x.sql\n+++ b/x.sql\n@@ -1 +1 @@\n-a\n+b\n"
    block = parse_diff(diff_text)[0]
    with pytest.raises(PatchApplyError):
        apply_file_diff(None, block)


@pytest.mark.parametrize(
    "bad_text",
    [
        "not a diff at all\n",
        "diff --git a/x b/x\nmissing the --- and +++ lines\n",
        "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ not a hunk header @@\n",
        # Declares 2 removed lines but only provides 1.
        "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,2 +1,1 @@\n-only one line\n",
    ],
)
def test_malformed_diff_text_raises_diff_parse_error(bad_text: str) -> None:
    with pytest.raises(DiffParseError):
        parse_diff(bad_text)


def test_apply_diff_rejects_path_traversal(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    diff_text = (
        "diff --git a/../outside.sql b/../outside.sql\n"
        "--- /dev/null\n"
        "+++ b/../outside.sql\n"
        "@@ -0,0 +1 @@\n"
        "+select 1\n"
    )
    with pytest.raises(PathTraversalError):
        apply_diff(scratch, diff_text)


def test_apply_diff_creates_parent_directories_for_added_files(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    diff_text = (
        "diff --git a/models/new/thing.sql b/models/new/thing.sql\n"
        "--- /dev/null\n"
        "+++ b/models/new/thing.sql\n"
        "@@ -0,0 +1 @@\n"
        "+select 1\n"
    )
    apply_diff(scratch, diff_text)
    assert (scratch / "models" / "new" / "thing.sql").read_text() == "select 1\n"


def test_removed_and_added_lines_are_newline_stripped() -> None:
    diff_text = "diff --git a/x.sql b/x.sql\n--- a/x.sql\n+++ b/x.sql\n@@ -1 +1 @@\n-old\n+new\n"
    block = parse_diff(diff_text)[0]
    assert block.removed_lines() == ("old",)
    assert block.added_lines() == ("new",)


@pytest.mark.parametrize(
    ("before_text", "after_text", "diff_text"),
    [
        (
            "old",
            "new\n",
            "diff --git a/x.sql b/x.sql\n"
            "--- a/x.sql\n"
            "+++ b/x.sql\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "\\ No newline at end of file\n"
            "+new\n",
        ),
        (
            "old\n",
            "new",
            "diff --git a/x.sql b/x.sql\n"
            "--- a/x.sql\n"
            "+++ b/x.sql\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "\\ No newline at end of file\n",
        ),
    ],
)
def test_no_newline_marker_apply_and_invert_round_trip(
    tmp_path: Path, before_text: str, after_text: str, diff_text: str
) -> None:
    block = parse_diff(diff_text)[0]
    removed = next(line for line in block.hunks[0].lines if line.kind == "removed")
    added = next(line for line in block.hunks[0].lines if line.kind == "added")
    assert removed.text.endswith("\n") == before_text.endswith("\n")
    assert added.text.endswith("\n") == after_text.endswith("\n")

    target = tmp_path / "x.sql"
    target.write_bytes(before_text.encode())
    apply_diff(tmp_path, diff_text)
    assert target.read_bytes() == after_text.encode()

    inverse = invert_diff(diff_text)
    assert inverse.count("\\ No newline at end of file\n") == 1
    apply_diff(tmp_path, inverse)
    assert target.read_bytes() == before_text.encode()


def test_parse_diff_tolerates_real_git_metadata_lines():
    """Live finding (bi-dbt #2533 round 5): CI hands the fixer raw `git
    diff` output, which carries index/mode metadata between the header and
    the '---' line. The parser must skip it."""
    from dbt_fixer.diffparse import parse_diff

    diff = (
        "diff --git a/models/x.yml b/models/x.yml\n"
        "index e6a745c2..11a2589e 100644\n"
        "--- a/models/x.yml\n"
        "+++ b/models/x.yml\n"
        "@@ -1 +1,2 @@\n"
        " version: 2\n"
        "+# note\n"
        "diff --git a/models/new.sql b/models/new.sql\n"
        "new file mode 100644\n"
        "index 00000000..59b97e29\n"
        "--- /dev/null\n"
        "+++ b/models/new.sql\n"
        "@@ -0,0 +1 @@\n"
        "+select 1\n"
    )
    blocks = parse_diff(diff)
    assert [b.path for b in blocks] == ["models/x.yml", "models/new.sql"]
    assert blocks[0].change_kind == "modified"
    assert blocks[1].change_kind == "added"


def test_invert_diff_survives_git_metadata_lines():
    from dbt_fixer.diffparse import invert_diff

    diff = (
        "diff --git a/a.yml b/a.yml\n"
        "index abc12345..def67890 100644\n"
        "--- a/a.yml\n+++ b/a.yml\n@@ -1 +1 @@\n-x\n+y\n"
    )
    inverted = invert_diff(diff)
    assert "-y" in inverted and "+x" in inverted
    assert "index " not in inverted  # inverted output is the package dialect
