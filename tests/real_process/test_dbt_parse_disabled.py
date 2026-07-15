"""Real-process proof that disabled dbt parse never executes an ambient binary."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from dbt_fixer.dbt_parse import run_dbt_parse_gate
from dbt_fixer.diffing import generate_unified_diff
from dbt_fixer.runners import real_dbt_subprocess_runner

pytestmark = pytest.mark.real_process


def test_disabled_gate_does_not_execute_ambient_dbt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "dbt_project.yml").write_text("name: isolated\n")
    (repo / "models" / "a.sql").write_text("select 1\n")
    after = tmp_path / "after"
    shutil.copytree(repo, after)
    (after / "models" / "a.sql").write_text("select 2\n")
    candidate_diff = generate_unified_diff(repo, after, ["models/a.sql"])

    marker = tmp_path / "AMBIENT_DBT_EXECUTED"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_dbt = bin_dir / "dbt"
    fake_dbt.write_text(f"#!/bin/sh\nprintf invoked > '{marker}'\nexit 0\n")
    fake_dbt.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    verdict = run_dbt_parse_gate(
        repo_root=repo,
        candidate_diff=candidate_diff,
        timeout_seconds=5.0,
        subprocess_runner=real_dbt_subprocess_runner,
    )

    assert verdict.outcome == "skipped"
    assert "disabled" in verdict.reason
    assert not marker.exists()
