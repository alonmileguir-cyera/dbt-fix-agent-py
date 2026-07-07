"""Real-process proof of the README's "Fresh install" contract.

This module is explicitly marked `real_process` (see `tests/conftest.py`'s
offline-only guard and `pyproject.toml`'s default `-m 'not real_process'`
collection filter) because it does the one thing the rest of this fully
offline suite never does: create a brand-new virtualenv, `pip install .`
this package into it with no test extras, and invoke the resulting
`python -m dbt_fixer.entrypoint` as a real subprocess -- proving the package
is actually installable and importable end-to-end from `pyproject.toml`
alone, not just "importable from `src/` because the repo checkout happens to
already be on `sys.path` via an editable install."

Deliberately not part of the default `pytest` run: creating a venv and
installing dependencies (`agno`, `boto3`, `slack_sdk`) from a package index
takes real wall-clock time and needs real network access, which would make
the default suite slow and non-hermetic. Run explicitly with:

    pytest -m real_process tests/real_process/test_fresh_install.py
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.real_process

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":  # pragma: no cover - this suite runs on POSIX CI
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def test_fresh_pip_install_and_entrypoint_invocation_end_to_end(tmp_path: Path) -> None:
    venv_dir = tmp_path / "fresh-venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = _venv_python(venv_dir)
    assert python.exists()

    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", str(_PACKAGE_ROOT)],
        capture_output=True,
        text=True,
        timeout=480,
    )
    assert install.returncode == 0, (
        f"`pip install .` failed:\nstdout:\n{install.stdout}\nstderr:\n{install.stderr}"
    )

    run = subprocess.run(
        [str(python), "-m", "dbt_fixer.entrypoint"],
        capture_output=True,
        text=True,
        timeout=60,
        env={},  # deliberately empty: no DBT_FIXER_* variables configured at all
    )

    assert run.returncode == 0
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    assert lines, f"expected at least one stdout line, got: {run.stdout!r}"
    assert lines[-1] == "dbt-fixer-status: failed"
    assert any(line.startswith("dbt-fixer-reason:") for line in lines)
    # The reason must name at least one of the two required-but-unset
    # variables, not just say something generic -- a deployer reading this
    # from a clean install needs to know exactly what to set.
    reason_line = next(line for line in lines if line.startswith("dbt-fixer-reason:"))
    assert "DBT_FIXER_FAILURE_KIND" in reason_line or "DBT_FIXER_REPO_PATH" in reason_line
