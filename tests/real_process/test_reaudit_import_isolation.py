"""Real-process proof that PR files cannot shadow the sealed auditor."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from dbt_fixer.reaudit import build_auditor_args
from dbt_fixer.runners import real_reaudit_subprocess_runner

pytestmark = pytest.mark.real_process


def test_reaudit_ignores_pr_local_dbt_auditor_package(tmp_path: Path) -> None:
    scratch = tmp_path / "untrusted-pr"
    malicious = scratch / "dbt_auditor"
    malicious.mkdir(parents=True)
    (malicious / "__init__.py").write_text("")
    (malicious / "entrypoint.py").write_text(
        "from pathlib import Path\n"
        "Path('PR_MODULE_EXECUTED').write_text('unsafe')\n"
        "print('malicious auditor executed')\n"
    )

    sealed_root = tmp_path / "sealed-site"
    sealed = sealed_root / "dbt_auditor"
    sealed.mkdir(parents=True)
    (sealed / "__init__.py").write_text("")
    (sealed / "entrypoint.py").write_text("print('sealed auditor executed')\n")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(sealed_root),
        "PYTHONSAFEPATH": "1",
    }
    outcome = real_reaudit_subprocess_runner(
        build_auditor_args(sys.executable),
        env,
        scratch,
        10.0,
    )

    assert outcome.returncode == 0, outcome.stderr
    assert outcome.stdout.strip() == "sealed auditor executed"
    assert not (scratch / "PR_MODULE_EXECUTED").exists()
