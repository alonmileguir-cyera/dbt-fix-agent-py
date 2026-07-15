"""Tests for `dbt_fixer.scratch`: scratch-copy creation and guaranteed cleanup
on every exit path."""

from __future__ import annotations

import pytest

from dbt_fixer.scratch import ScratchCopyError, scratch_copy


def _make_repo(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "model.sql").write_text("select 1")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main")
    return src


def test_scratch_copy_is_isolated_and_matches_source(tmp_path):
    src = _make_repo(tmp_path)
    with scratch_copy(src) as dest:
        assert dest.exists()
        assert dest != src
        assert (dest / "model.sql").read_text() == "select 1"

    # source untouched
    assert (src / "model.sql").read_text() == "select 1"


def test_scratch_copy_excludes_git_directory(tmp_path):
    src = _make_repo(tmp_path)
    with scratch_copy(src) as dest:
        assert not (dest / ".git").exists()


def test_mutating_the_scratch_copy_never_touches_the_source(tmp_path):
    src = _make_repo(tmp_path)
    with scratch_copy(src) as dest:
        (dest / "model.sql").write_text("select 2")
        assert (dest / "model.sql").read_text() == "select 2"
    assert (src / "model.sql").read_text() == "select 1"


def test_cleanup_runs_on_normal_return(tmp_path):
    src = _make_repo(tmp_path)
    with scratch_copy(src) as dest:
        scratch_root = dest.parent
    assert not dest.exists()
    assert not scratch_root.exists()


def test_cleanup_runs_when_an_exception_is_raised_mid_use(tmp_path):
    src = _make_repo(tmp_path)
    captured = {}

    with pytest.raises(RuntimeError, match="boom mid-use"):
        with scratch_copy(src) as dest:
            captured["dest"] = dest
            assert dest.exists()
            raise RuntimeError("boom mid-use")

    assert not captured["dest"].exists()
    assert not captured["dest"].parent.exists()


def test_cleanup_runs_on_early_return_from_enclosing_function(tmp_path):
    src = _make_repo(tmp_path)
    holder: dict = {}

    def _use_it():
        with scratch_copy(src) as dest:
            holder["dest"] = dest
            if dest.exists():
                return "early-return-value"
        return "never-reached"

    result = _use_it()
    assert result == "early-return-value"
    assert not holder["dest"].exists()


def test_rejects_missing_source(tmp_path):
    with pytest.raises(ScratchCopyError):
        with scratch_copy(tmp_path / "does-not-exist"):
            pass  # pragma: no cover - must never get here


def test_rejects_source_that_is_a_file(tmp_path):
    a_file = tmp_path / "not-a-dir.txt"
    a_file.write_text("hello")
    with pytest.raises(ScratchCopyError):
        with scratch_copy(a_file):
            pass  # pragma: no cover - must never get here


def test_two_scratch_copies_of_the_same_source_are_independent(tmp_path):
    src = _make_repo(tmp_path)
    with scratch_copy(src) as dest_a, scratch_copy(src) as dest_b:
        assert dest_a != dest_b
        (dest_a / "model.sql").write_text("select 'a'")
        assert (dest_b / "model.sql").read_text() == "select 1"


def test_rejects_file_symlink_before_copy(tmp_path):
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("must-not-be-copied")
    src = _make_repo(tmp_path)
    (src / "linked-secret.txt").symlink_to(outside)

    with pytest.raises(ScratchCopyError, match=r"unsupported symlink: linked-secret\.txt"):
        with scratch_copy(src):
            pass  # pragma: no cover - must never get here

    assert outside.read_text() == "must-not-be-copied"


def test_rejects_directory_symlink_before_copy(tmp_path):
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "secret.txt").write_text("must-not-be-copied")
    src = _make_repo(tmp_path)
    (src / "linked-directory").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ScratchCopyError, match=r"unsupported symlink: linked-directory"):
        with scratch_copy(src):
            pass  # pragma: no cover - must never get here

    assert (outside / "secret.txt").read_text() == "must-not-be-copied"
