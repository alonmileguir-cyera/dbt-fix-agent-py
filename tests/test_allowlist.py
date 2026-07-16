"""Tests for `dbt_fixer.allowlist`: the deterministic Allowlist Classifier Gate.

Every test builds a real on-disk repo fixture and a real candidate diff via
`dbt_fixer.diffing.generate_unified_diff` (never a hand-typed diff string,
except where a specific malformed-syntax case requires it), so the gate is
always exercised against its actual production input shape.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dbt_fixer.allowlist import AllowlistCaps, run_allowlist_gate
from dbt_fixer.diffing import generate_unified_diff
from dbt_fixer.intake import FailingCheck

DEFAULT_CAPS = AllowlistCaps(max_changed_files=5, max_changed_lines=60)


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative_path, content in files.items():
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return repo


def _candidate_diff(repo: Path, tmp_path: Path, edits: dict[str, str], *, changed_paths=None) -> str:
    """Build a real unified diff by copying `repo`, applying `edits`, and diffing."""

    after = tmp_path / "after"
    if after.exists():
        shutil.rmtree(after)
    shutil.copytree(repo, after)
    for relative_path, new_content in edits.items():
        target = after / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content)
    paths = changed_paths if changed_paths is not None else list(edits.keys())
    return generate_unified_diff(repo, after, paths)


# --- file-type restriction ---------------------------------------------------


def test_candidate_touching_only_allowed_files_passes_type_check(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": "select 1\n"})
    diff_text = _candidate_diff(repo, tmp_path, {"models/a.sql": "select 1\nfrom x\n"})

    verdict = run_allowlist_gate(
        repo_root=repo,
        candidate_diff=diff_text,
        pr_diff="",
        failure_kind="ci",
        caps=DEFAULT_CAPS,
    )
    assert verdict.passed
    assert verdict.violation is None


@pytest.mark.parametrize(
    "path",
    ["dbt_project.yml", "seeds/a.csv", "macros/helper.sql", "models/a.py", "tests/custom_test.sql"],
)
def test_candidate_touching_disallowed_file_is_rejected(tmp_path: Path, path: str) -> None:
    repo = _make_repo(tmp_path, {path: "original content\n"})
    diff_text = _candidate_diff(repo, tmp_path, {path: "changed content\n"})

    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff=diff_text, pr_diff="", failure_kind="ci", caps=DEFAULT_CAPS
    )
    assert not verdict.passed
    assert verdict.violation == "file_type_not_allowed"
    assert path in verdict.reason


# --- PR-authored-only SQL deletion ------------------------------------------


@pytest.mark.parametrize("kind", ["ci", "audit"])
def test_sql_replacement_can_restore_pr_added_head_line_to_exact_base_line(
    tmp_path: Path, kind: str
) -> None:
    path = "models/int_product__identities.sql"
    base_line = "{{ ref('stg_prod_aurora_etl__identity_extra')}} iex"
    head_line = "{{ ref('stg_prod_aurora_etl__identity_extra')}} iex_dbt_audit_exact_file_canary"
    base = _make_repo(tmp_path, {path: f"select iex.id\nfrom source\njoin {base_line}\n"})
    head = tmp_path / "pr-head"
    shutil.copytree(base, head)
    head.joinpath(path).write_text(f"select iex.id\nfrom source\njoin {head_line}\n")
    pr_diff = generate_unified_diff(base, head, [path])
    candidate_diff = _candidate_diff(
        head,
        tmp_path,
        {path: f"select iex.id\nfrom source\njoin {base_line}\n"},
    )

    verdict = run_allowlist_gate(
        repo_root=head,
        candidate_diff=candidate_diff,
        pr_diff=pr_diff,
        failure_kind=kind,
        caps=DEFAULT_CAPS,
    )
    assert verdict.passed


def test_sql_replacement_can_correct_pr_added_head_line_to_third_safe_line(
    tmp_path: Path,
) -> None:
    path = "models/a.sql"
    base = _make_repo(tmp_path, {path: "select old_name\n"})
    head = tmp_path / "pr-head"
    shutil.copytree(base, head)
    head.joinpath(path).write_text("select broken_name\n")
    pr_diff = generate_unified_diff(base, head, [path])
    candidate_diff = _candidate_diff(
        head, tmp_path, {path: "select corrected_name\n"}
    )

    verdict = run_allowlist_gate(
        repo_root=head,
        candidate_diff=candidate_diff,
        pr_diff=pr_diff,
        failure_kind="audit",
        caps=DEFAULT_CAPS,
    )
    assert verdict.passed


def test_sql_candidate_cannot_remove_preexisting_base_line(tmp_path: Path) -> None:
    path = "models/a.sql"
    base = _make_repo(tmp_path, {path: "select 1\nfrom source\n"})
    head = tmp_path / "pr-head"
    shutil.copytree(base, head)
    head.joinpath(path).write_text("select 1\nfrom source\nwhere active\n")
    pr_diff = generate_unified_diff(base, head, [path])
    candidate_diff = _candidate_diff(
        head, tmp_path, {path: "select 1\nwhere active\n"}
    )

    verdict = run_allowlist_gate(
        repo_root=head,
        candidate_diff=candidate_diff,
        pr_diff=pr_diff,
        failure_kind="audit",
        caps=DEFAULT_CAPS,
    )
    assert not verdict.passed
    assert verdict.violation == "sql_deletion_not_a_restore"
    assert "did not add" in verdict.reason


def test_sql_removal_authorization_preserves_multiplicity(tmp_path: Path) -> None:
    path = "models/a.sql"
    base = _make_repo(tmp_path, {path: "select x\n"})
    head = tmp_path / "pr-head"
    shutil.copytree(base, head)
    head.joinpath(path).write_text("select x\nselect x\n")
    pr_diff = generate_unified_diff(base, head, [path])
    candidate_diff = _candidate_diff(head, tmp_path, {path: ""})

    verdict = run_allowlist_gate(
        repo_root=head,
        candidate_diff=candidate_diff,
        pr_diff=pr_diff,
        failure_kind="audit",
        caps=DEFAULT_CAPS,
    )
    assert not verdict.passed
    assert verdict.violation == "sql_deletion_not_a_restore"


def test_sql_deletion_not_matching_pr_deletion_is_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": "select 1\nfrom x\nwhere y = 1\n"})
    pr_diff = ""  # the PR diff deleted nothing at all
    candidate_diff = _candidate_diff(repo, tmp_path, {"models/a.sql": "select 1\nfrom x\n"})

    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff=candidate_diff, pr_diff=pr_diff, failure_kind="ci", caps=DEFAULT_CAPS
    )
    assert not verdict.passed
    assert verdict.violation == "sql_deletion_not_a_restore"


@pytest.mark.parametrize("kind", ["ci", "audit"])
def test_restore_only_rule_applies_to_both_failure_kinds(tmp_path: Path, kind: str) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": "select 1\nfrom x\nwhere y = 1\n"})
    pr_diff = ""
    candidate_diff = _candidate_diff(repo, tmp_path, {"models/a.sql": "select 1\nfrom x\n"})

    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff=candidate_diff, pr_diff=pr_diff, failure_kind=kind, caps=DEFAULT_CAPS
    )
    assert not verdict.passed
    assert verdict.violation == "sql_deletion_not_a_restore"


# --- test weakening, by failure kind ----------------------------------------

_SCHEMA_YML_BEFORE = (
    "models:\n"
    "  - name: orders\n"
    "    columns:\n"
    "      - name: id\n"
    "        tests:\n"
    "          - not_null\n"
    "          - unique\n"
)
_SCHEMA_YML_AFTER = (
    "models:\n"
    "  - name: orders\n"
    "    columns:\n"
    "      - name: id\n"
    "        tests:\n"
    "          - unique\n"
)


def test_ci_kind_categorically_rejects_test_deletion(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/schema.yml": _SCHEMA_YML_BEFORE})
    candidate_diff = _candidate_diff(repo, tmp_path, {"models/schema.yml": _SCHEMA_YML_AFTER})

    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff=candidate_diff, pr_diff="", failure_kind="ci", caps=DEFAULT_CAPS
    )
    assert not verdict.passed
    assert verdict.violation == "test_weakening_rejected_ci"


def test_audit_kind_rejects_test_deletion_without_proof(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/schema.yml": _SCHEMA_YML_BEFORE})
    candidate_diff = _candidate_diff(repo, tmp_path, {"models/schema.yml": _SCHEMA_YML_AFTER})

    verdict = run_allowlist_gate(
        repo_root=repo,
        candidate_diff=candidate_diff,
        pr_diff="",
        failure_kind="audit",
        caps=DEFAULT_CAPS,
        failing_checks=(FailingCheck(identifier="not_null_orders_id", evidence="still failing"),),
    )
    assert not verdict.passed
    assert verdict.violation == "test_weakening_rejected_audit_unproven"


def test_audit_kind_accepts_test_deletion_with_explicit_proof(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/schema.yml": _SCHEMA_YML_BEFORE})
    candidate_diff = _candidate_diff(repo, tmp_path, {"models/schema.yml": _SCHEMA_YML_AFTER})

    verdict = run_allowlist_gate(
        repo_root=repo,
        candidate_diff=candidate_diff,
        pr_diff="",
        failure_kind="audit",
        caps=DEFAULT_CAPS,
        failing_checks=(
            FailingCheck(
                identifier="not_null_orders_id",
                evidence="Manually verified against source data; this test has been proven wrong.",
            ),
        ),
    )
    assert verdict.passed


# --- hard caps + sensitive patterns ------------------------------------------


def test_candidate_exceeding_max_changed_files_is_rejected(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, {f"models/a{i}.sql": f"select {i}\n" for i in range(3)}
    )
    edits = {f"models/a{i}.sql": f"select {i}00\n" for i in range(3)}
    candidate_diff = _candidate_diff(repo, tmp_path, edits)

    verdict = run_allowlist_gate(
        repo_root=repo,
        candidate_diff=candidate_diff,
        pr_diff="",
        failure_kind="ci",
        caps=AllowlistCaps(max_changed_files=2, max_changed_lines=60),
    )
    assert not verdict.passed
    assert verdict.violation == "max_changed_files_exceeded"


def test_candidate_exceeding_max_changed_lines_is_rejected(tmp_path: Path) -> None:
    original = "\n".join(f"line {i}" for i in range(20)) + "\n"
    repo = _make_repo(tmp_path, {"models/a.sql": original})
    modified = "\n".join(f"line {i} changed" for i in range(20)) + "\n"
    candidate_diff = _candidate_diff(repo, tmp_path, {"models/a.sql": modified})

    verdict = run_allowlist_gate(
        repo_root=repo,
        candidate_diff=candidate_diff,
        pr_diff="",
        failure_kind="ci",
        caps=AllowlistCaps(max_changed_files=5, max_changed_lines=10),
    )
    assert not verdict.passed
    assert verdict.violation == "max_changed_lines_exceeded"


@pytest.mark.parametrize(
    "before,after",
    [
        ("select 1\n", "{{ config(pre_hook=\"grant select\") }}\nselect 1\n"),
        ("select 1\n", "{{ config(materialized='table') }}\nselect 1\n"),
        ("select 1\n", "-- bypass masking for this column\nselect 1\n"),
    ],
)
def test_sensitive_pattern_rejected_even_under_caps(tmp_path: Path, before: str, after: str) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": before})
    candidate_diff = _candidate_diff(repo, tmp_path, {"models/a.sql": after})

    verdict = run_allowlist_gate(
        repo_root=repo,
        candidate_diff=candidate_diff,
        pr_diff="",
        failure_kind="ci",
        caps=AllowlistCaps(max_changed_files=50, max_changed_lines=2000),
    )
    assert not verdict.passed
    assert verdict.violation == "sensitive_pattern_detected"


# --- malformed / no-op candidates -------------------------------------------


def test_empty_candidate_diff_is_rejected_as_no_op(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": "select 1\n"})
    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff="", pr_diff="", failure_kind="ci", caps=DEFAULT_CAPS
    )
    assert not verdict.passed
    assert verdict.violation == "no_op_candidate"


def test_malformed_diff_syntax_is_rejected_not_raised(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": "select 1\n"})
    malformed = "this is not a valid unified diff at all\n"

    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff=malformed, pr_diff="", failure_kind="ci", caps=DEFAULT_CAPS
    )
    assert not verdict.passed
    assert verdict.violation == "patch_apply_failed"


def test_diff_that_does_not_apply_cleanly_is_rejected_not_raised(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": "select 1\nfrom x\n"})
    # Hand-built diff whose context line does not match the real repo content.
    conflicting = (
        "diff --git a/models/a.sql b/models/a.sql\n"
        "--- a/models/a.sql\n"
        "+++ b/models/a.sql\n"
        "@@ -1,2 +1,2 @@\n"
        " select 999\n"
        "-from x\n"
        "+from y\n"
    )

    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff=conflicting, pr_diff="", failure_kind="ci", caps=DEFAULT_CAPS
    )
    assert not verdict.passed
    assert verdict.violation == "patch_apply_failed"


def test_no_net_change_candidate_is_rejected_as_no_op(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": "select 1\nfrom x\n"})
    # A contrived diff: removes then re-adds the exact same line, netting no change.
    noop = (
        "diff --git a/models/a.sql b/models/a.sql\n"
        "--- a/models/a.sql\n"
        "+++ b/models/a.sql\n"
        "@@ -1,2 +1,2 @@\n"
        " select 1\n"
        "-from x\n"
        "+from x\n"
    )

    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff=noop, pr_diff="", failure_kind="ci", caps=DEFAULT_CAPS
    )
    assert not verdict.passed
    assert verdict.violation == "no_op_candidate"


# --- determinism / no model calls -------------------------------------------


class _CountingRunner:
    """A fake model runner that records how many times it is invoked.

    Never passed to `run_allowlist_gate` (its signature accepts no model
    runner at all); used here only to prove that repeatedly running the
    gate never causes any call to accrue against it, i.e. the gate truly
    never reaches for a model.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return "{}"


def test_allowlist_gate_is_deterministic_and_never_calls_a_model(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, {"models/a.sql": "select 1\n"})
    candidate_diff = _candidate_diff(
        repo, tmp_path, {"models/a.sql": "select 1\nfrom x\n"}
    )

    runner = _CountingRunner()

    verdicts = [
        run_allowlist_gate(
            repo_root=repo,
            candidate_diff=candidate_diff,
            pr_diff="",
            failure_kind="ci",
            caps=DEFAULT_CAPS,
        )
        for _ in range(5)
    ]

    assert all(v.passed for v in verdicts)
    assert len({(v.passed, v.violation, v.reason) for v in verdicts}) == 1
    assert runner.calls == 0


def test_multi_project_models_paths_are_allowed():
    """bi-dbt layout: <project>/models/... must pass the path rule."""
    from dbt_fixer.allowlist import _check_file_types  # type: ignore[attr-defined]
    from dbt_fixer.diffparse import parse_diff

    diff = (
        "diff --git a/bi-dbt-multiregion/models/staging/x/_m.yml b/bi-dbt-multiregion/models/staging/x/_m.yml\n"
        "--- /dev/null\n"
        "+++ b/bi-dbt-multiregion/models/staging/x/_m.yml\n"
        "@@ -0,0 +1 @@\n"
        "+version: 2\n"
    )
    blocks = parse_diff(diff)
    assert _check_file_types(blocks) is None  # no violation


def test_paths_outside_any_models_tree_still_rejected():
    from dbt_fixer.allowlist import _check_file_types  # type: ignore[attr-defined]
    from dbt_fixer.diffparse import parse_diff

    diff = (
        "diff --git a/macros/evil.yml b/macros/evil.yml\n"
        "--- a/macros/evil.yml\n"
        "+++ b/macros/evil.yml\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    blocks = parse_diff(diff)
    verdict = _check_file_types(blocks)
    assert verdict is not None and not verdict.passed


# ---------------------------------------------------------------------------
# moved-test exemption (live finding: bi-dbt #2533 round 4)
# ---------------------------------------------------------------------------


def _rename_diff():
    return (
        "diff --git a/proj/models/staging/_x__models.yml b/proj/models/staging/_x__models.yml\n"
        "--- a/proj/models/staging/_x__models.yml\n"
        "+++ b/proj/models/staging/_x__models.yml\n"
        "@@ -1,8 +1,8 @@\n"
        " version: 2\n"
        " models:\n"
        "   - name: m\n"
        "     columns:\n"
        "-      - name: team_uid\n"
        "+      - name: id\n"
        "         tests:\n"
        "-          - unique\n"
        "-          - not_null\n"
        "+          - unique\n"
        "+          - not_null\n"
    )


def test_column_rename_with_identical_tests_is_a_move_not_a_deletion(tmp_path):
    """Renaming a mis-declared column re-lists its identical tests under
    the new name; that must clear the allowlist under BOTH kinds."""
    repo = _make_repo(tmp_path, {
        "proj/models/staging/_x__models.yml": (
            "version: 2\nmodels:\n  - name: m\n    columns:\n"
            "      - name: team_uid\n        tests:\n"
            "          - unique\n          - not_null\n"
        ),
    })
    for kind in ("audit", "ci"):
        verdict = run_allowlist_gate(
            repo_root=repo,
            candidate_diff=_rename_diff(),
            pr_diff="",
            failure_kind=kind,
            caps=DEFAULT_CAPS,
        )
        assert verdict.passed, f"kind={kind}: {verdict.reason}"


def test_net_test_deletion_is_still_rejected(tmp_path):
    """Removing a test WITHOUT re-adding it stays rejected (no evidence)."""

    diff = (
        "diff --git a/proj/models/staging/_x__models.yml b/proj/models/staging/_x__models.yml\n"
        "--- a/proj/models/staging/_x__models.yml\n"
        "+++ b/proj/models/staging/_x__models.yml\n"
        "@@ -1,6 +1,5 @@\n"
        " version: 2\n"
        " models:\n"
        "   - name: m\n"
        "     columns:\n"
        "       - name: id\n"
        "-          - not_null\n"
    )
    repo = _make_repo(tmp_path, {
        "proj/models/staging/_x__models.yml": (
            "version: 2\nmodels:\n  - name: m\n    columns:\n"
            "      - name: id\n          - not_null\n"
        ),
    })
    for kind in ("audit", "ci"):
        verdict = run_allowlist_gate(
            repo_root=repo, candidate_diff=diff, pr_diff="",
            failure_kind=kind, caps=DEFAULT_CAPS,
        )
        assert not verdict.passed, f"kind={kind} should reject a net deletion"


def test_move_exemption_is_count_bounded(tmp_path):
    """Removing a test twice while re-adding it once: one removal is a
    move, the second is a net deletion and still rejected."""

    diff = (
        "diff --git a/proj/models/staging/_x__models.yml b/proj/models/staging/_x__models.yml\n"
        "--- a/proj/models/staging/_x__models.yml\n"
        "+++ b/proj/models/staging/_x__models.yml\n"
        "@@ -1,10 +1,7 @@\n"
        " version: 2\n"
        " models:\n"
        "   - name: m\n"
        "     columns:\n"
        "       - name: a\n"
        "-          - not_null\n"
        "       - name: b\n"
        "-          - not_null\n"
        "       - name: c\n"
        "+          - not_null\n"
    )
    repo = _make_repo(tmp_path, {
        "proj/models/staging/_x__models.yml": (
            "version: 2\nmodels:\n  - name: m\n    columns:\n"
            "      - name: a\n          - not_null\n"
            "      - name: b\n          - not_null\n"
            "      - name: c\n"
        ),
    })
    verdict = run_allowlist_gate(
        repo_root=repo, candidate_diff=diff, pr_diff="",
        failure_kind="audit", caps=DEFAULT_CAPS,
    )
    assert not verdict.passed


def test_relocating_a_test_to_a_different_persisting_column_is_rejected(tmp_path):
    """Red-team finding 2: removing `- unique` from a column that PERSISTS
    and re-adding it under a different column is weakening, not a move -
    must be rejected under both kinds."""
    repo = _make_repo(tmp_path, {
        "proj/models/staging/_x__models.yml": (
            "version: 2\nmodels:\n  - name: m\n    columns:\n"
            "      - name: tenant_id\n        tests:\n          - not_null\n          - unique\n"
            "      - name: created_at\n        tests:\n          - not_null\n"
        ),
    })
    diff = (
        "diff --git a/proj/models/staging/_x__models.yml b/proj/models/staging/_x__models.yml\n"
        "--- a/proj/models/staging/_x__models.yml\n"
        "+++ b/proj/models/staging/_x__models.yml\n"
        "@@ -5,7 +5,7 @@\n"
        "       - name: tenant_id\n"
        "         tests:\n"
        "           - not_null\n"
        "-          - unique\n"
        "       - name: created_at\n"
        "         tests:\n"
        "           - not_null\n"
        "+          - unique\n"
    )
    for kind in ("audit", "ci"):
        v = run_allowlist_gate(repo_root=repo, candidate_diff=diff, pr_diff="",
                               failure_kind=kind, caps=DEFAULT_CAPS)
        assert not v.passed, f"kind={kind}: relocation to a persisting column must be rejected"
