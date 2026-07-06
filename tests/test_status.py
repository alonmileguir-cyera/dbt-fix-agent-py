"""Tests for the fixed status vocabulary in `dbt_fixer.status`."""

from __future__ import annotations

import pytest

from dbt_fixer.status import GateResult, RunResult, STATUSES, is_valid_status


def test_exactly_three_statuses_exist():
    assert set(STATUSES) == {"proposed", "no_safe_fix", "failed"}


def test_is_valid_status():
    assert is_valid_status("proposed")
    assert is_valid_status("no_safe_fix")
    assert is_valid_status("failed")
    assert not is_valid_status("blocked")


def test_run_result_rejects_invalid_status():
    with pytest.raises(ValueError):
        RunResult(status="blocked")  # type: ignore[arg-type]


def test_gate_result_glyphs_are_distinct():
    passed = GateResult(name="allowlist", outcome="pass")
    failed = GateResult(name="allowlist", outcome="fail", detail="touched a hook")
    skipped = GateResult(name="dbt parse", outcome="skipped", detail="dbt not on PATH")
    glyphs = {passed.glyph(), failed.glyph(), skipped.glyph()}
    assert len(glyphs) == 3


def test_gate_result_render_includes_detail_when_present():
    gate = GateResult(name="allowlist", outcome="fail", detail="touched a hook")
    rendered = gate.render()
    assert "allowlist" in rendered
    assert "touched a hook" in rendered
