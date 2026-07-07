"""Tests for `dbt_fixer.applier.apply_proposal`.

Covers: correct whole-file-replace and line-range-edit application
(including multiple non-overlapping line-range edits to one file), original
checkout immutability (edits only ever touch the scratch copy), and every
fail-closed rejection path (missing target, directory target, out-of-bounds
line range, and each conflict shape) -- verifying that a rejected proposal
never partially mutates the scratch copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dbt_fixer.applier import (
    ConflictingEditsError,
    EditTargetIsDirectoryError,
    EditTargetNotFoundError,
    InvalidLineRangeError,
    apply_proposal,
)
from dbt_fixer.pathsafe import PathTraversalError
from dbt_fixer.proposal import Edit, Proposal


def _make_scratch(tmp_path: Path) -> Path:
    root = tmp_path / "scratch"
    (root / "models").mkdir(parents=True)
    (root / "models" / "a.sql").write_text(
        "select\n    id,\n    email\nfrom raw.customers\n", encoding="utf-8"
    )
    (root / "models" / "b.sql").write_text("select 1\n", encoding="utf-8")
    return root


def _proposal(*edits: Edit, rationale: str = "test") -> Proposal:
    return Proposal(edits=tuple(edits), rationale=rationale)


# ---------------------------------------------------------------------------
# success paths
# ---------------------------------------------------------------------------


def test_applies_whole_file_replace(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    edit = Edit(kind="whole_file_replace", path="models/b.sql", content="select 2\n")

    result = apply_proposal(root, _proposal(edit))

    assert result.changed_paths == ("models/b.sql",)
    assert (root / "models" / "b.sql").read_text(encoding="utf-8") == "select 2\n"


def test_applies_line_range_edit(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    edit = Edit(
        kind="line_range_edit",
        path="models/a.sql",
        start_line=2,
        end_line=3,
        replacement="    id,\n    email,\n    created_at\n",
    )

    result = apply_proposal(root, _proposal(edit))

    assert result.changed_paths == ("models/a.sql",)
    assert (root / "models" / "a.sql").read_text(encoding="utf-8") == (
        "select\n    id,\n    email,\n    created_at\nfrom raw.customers\n"
    )


def test_line_range_edit_preserves_indentation_and_line_endings_outside_range(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    (root / "models").mkdir(parents=True)
    # Mixed leading whitespace (tabs and spaces) on lines outside the edit
    # range must survive completely untouched.
    original = "select\n\tid,\n    email\nfrom raw.customers\n"
    (root / "models" / "indented.sql").write_text(original, encoding="utf-8")
    edit = Edit(
        kind="line_range_edit",
        path="models/indented.sql",
        start_line=2,
        end_line=2,
        replacement="\tid,\n\tfull_name,\n",
    )

    apply_proposal(root, _proposal(edit))

    result = (root / "models" / "indented.sql").read_text(encoding="utf-8")
    assert result == "select\n\tid,\n\tfull_name,\n    email\nfrom raw.customers\n"
    # The untouched lines keep their exact original indentation.
    assert "    email\n" in result
    assert "from raw.customers\n" in result


def test_line_range_edit_on_final_line_without_trailing_newline(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    (root / "models").mkdir(parents=True)
    original = "select 1\nselect 2"  # no trailing newline on the last line
    (root / "models" / "no_trailing_newline.sql").write_text(original, encoding="utf-8")
    edit = Edit(
        kind="line_range_edit",
        path="models/no_trailing_newline.sql",
        start_line=2,
        end_line=2,
        replacement="select 999",
    )

    apply_proposal(root, _proposal(edit))

    result = (root / "models" / "no_trailing_newline.sql").read_text(encoding="utf-8")
    assert result == "select 1\nselect 999"


def test_applies_multiple_non_overlapping_line_range_edits_to_same_file(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    edit_top = Edit(
        kind="line_range_edit", path="models/a.sql", start_line=1, end_line=1, replacement="SELECT\n"
    )
    edit_bottom = Edit(
        kind="line_range_edit",
        path="models/a.sql",
        start_line=4,
        end_line=4,
        replacement="FROM raw.customers\n",
    )

    result = apply_proposal(root, _proposal(edit_top, edit_bottom))

    assert result.changed_paths == ("models/a.sql",)
    assert (root / "models" / "a.sql").read_text(encoding="utf-8") == (
        "SELECT\n    id,\n    email\nFROM raw.customers\n"
    )


def test_applies_edits_across_multiple_files(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    edit_a = Edit(kind="whole_file_replace", path="models/a.sql", content="select 42\n")
    edit_b = Edit(kind="whole_file_replace", path="models/b.sql", content="select 43\n")

    result = apply_proposal(root, _proposal(edit_a, edit_b))

    assert result.changed_paths == ("models/a.sql", "models/b.sql")
    assert (root / "models" / "a.sql").read_text(encoding="utf-8") == "select 42\n"
    assert (root / "models" / "b.sql").read_text(encoding="utf-8") == "select 43\n"


# ---------------------------------------------------------------------------
# original-checkout immutability
# ---------------------------------------------------------------------------


def test_never_mutates_a_different_directory_than_scratch_root(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    (original_root / "models").mkdir(parents=True)
    (original_root / "models" / "a.sql").write_text("select 1\n", encoding="utf-8")
    original_snapshot = (original_root / "models" / "a.sql").read_text(encoding="utf-8")

    scratch_root = _make_scratch(tmp_path)
    edit = Edit(kind="whole_file_replace", path="models/a.sql", content="select 999\n")

    apply_proposal(scratch_root, _proposal(edit))

    # The unrelated "original" tree (standing in for the real checkout) is
    # untouched; only the scratch copy passed to apply_proposal changed.
    assert (original_root / "models" / "a.sql").read_text(encoding="utf-8") == original_snapshot
    assert (scratch_root / "models" / "a.sql").read_text(encoding="utf-8") == "select 999\n"


# ---------------------------------------------------------------------------
# fail-closed rejection paths -- no partial application
# ---------------------------------------------------------------------------


def test_rejects_missing_target_and_leaves_scratch_untouched(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    good_edit = Edit(kind="whole_file_replace", path="models/a.sql", content="select 999\n")
    bad_edit = Edit(kind="whole_file_replace", path="models/does_not_exist.sql", content="x")
    original_a = (root / "models" / "a.sql").read_text(encoding="utf-8")

    with pytest.raises(EditTargetNotFoundError):
        apply_proposal(root, _proposal(good_edit, bad_edit))

    assert (root / "models" / "a.sql").read_text(encoding="utf-8") == original_a


def test_rejects_directory_target(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    edit = Edit(kind="whole_file_replace", path="models", content="x")

    with pytest.raises(EditTargetIsDirectoryError):
        apply_proposal(root, _proposal(edit))


def test_rejects_path_traversal_target(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    edit = Edit(kind="whole_file_replace", path="../outside.sql", content="x")

    with pytest.raises(PathTraversalError):
        apply_proposal(root, _proposal(edit))


def test_rejects_out_of_bounds_line_range_and_leaves_scratch_untouched(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    original_a = (root / "models" / "a.sql").read_text(encoding="utf-8")
    edit = Edit(
        kind="line_range_edit", path="models/a.sql", start_line=10, end_line=12, replacement="x\n"
    )

    with pytest.raises(InvalidLineRangeError):
        apply_proposal(root, _proposal(edit))

    assert (root / "models" / "a.sql").read_text(encoding="utf-8") == original_a


def test_rejects_two_whole_file_replace_edits_on_same_path(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    edit_one = Edit(kind="whole_file_replace", path="models/a.sql", content="one\n")
    edit_two = Edit(kind="whole_file_replace", path="models/a.sql", content="two\n")

    with pytest.raises(ConflictingEditsError):
        apply_proposal(root, _proposal(edit_one, edit_two))


def test_rejects_whole_file_replace_and_line_range_edit_on_same_path(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    whole = Edit(kind="whole_file_replace", path="models/a.sql", content="one\n")
    ranged = Edit(
        kind="line_range_edit", path="models/a.sql", start_line=1, end_line=1, replacement="x\n"
    )

    with pytest.raises(ConflictingEditsError):
        apply_proposal(root, _proposal(whole, ranged))


def test_rejects_overlapping_line_range_edits_on_same_path(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    edit_one = Edit(
        kind="line_range_edit", path="models/a.sql", start_line=1, end_line=2, replacement="x\n"
    )
    edit_two = Edit(
        kind="line_range_edit", path="models/a.sql", start_line=2, end_line=3, replacement="y\n"
    )

    with pytest.raises(ConflictingEditsError):
        apply_proposal(root, _proposal(edit_one, edit_two))


def test_conflict_detection_happens_before_any_mutation(tmp_path: Path) -> None:
    root = _make_scratch(tmp_path)
    original_a = (root / "models" / "a.sql").read_text(encoding="utf-8")
    original_b = (root / "models" / "b.sql").read_text(encoding="utf-8")

    good_edit = Edit(kind="whole_file_replace", path="models/b.sql", content="mutated\n")
    conflicting_one = Edit(kind="whole_file_replace", path="models/a.sql", content="one\n")
    conflicting_two = Edit(kind="whole_file_replace", path="models/a.sql", content="two\n")

    with pytest.raises(ConflictingEditsError):
        apply_proposal(root, _proposal(good_edit, conflicting_one, conflicting_two))

    assert (root / "models" / "a.sql").read_text(encoding="utf-8") == original_a
    assert (root / "models" / "b.sql").read_text(encoding="utf-8") == original_b


# ---------------------------------------------------------------------------
# create_file application
# ---------------------------------------------------------------------------


def test_create_file_creates_parents_and_writes(tmp_path):
    from dbt_fixer.applier import apply_proposal
    from dbt_fixer.proposal import Edit, Proposal

    proposal = Proposal(
        edits=(Edit(kind="create_file", path="models/staging/newdir/_m.yml", content="version: 2\n"),),
        rationale="r",
    )
    applied = apply_proposal(tmp_path, proposal)
    assert (tmp_path / "models/staging/newdir/_m.yml").read_text() == "version: 2\n"
    assert applied.changed_paths == ("models/staging/newdir/_m.yml",)


def test_create_file_existing_target_rejected(tmp_path):
    import pytest

    from dbt_fixer.applier import EditTargetAlreadyExistsError, apply_proposal
    from dbt_fixer.proposal import Edit, Proposal

    (tmp_path / "models").mkdir()
    (tmp_path / "models/x.yml").write_text("already here")
    proposal = Proposal(
        edits=(Edit(kind="create_file", path="models/x.yml", content="new"),),
        rationale="r",
    )
    with pytest.raises(EditTargetAlreadyExistsError):
        apply_proposal(tmp_path, proposal)
    assert (tmp_path / "models/x.yml").read_text() == "already here"  # untouched


def test_create_file_traversal_rejected(tmp_path):
    import pytest

    from dbt_fixer.applier import apply_proposal
    from dbt_fixer.pathsafe import PathTraversalError
    from dbt_fixer.proposal import Edit, Proposal

    proposal = Proposal(
        edits=(Edit(kind="create_file", path="../outside.yml", content="x"),),
        rationale="r",
    )
    with pytest.raises(PathTraversalError):
        apply_proposal(tmp_path, proposal)
