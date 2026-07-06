"""Static proof that every Sprint 1 module is free of network/model/subprocess
dependencies.

This complements the runtime guard in `conftest.py` (which would raise if any
test actually *triggered* a real socket or subprocess call): this test proves
the stronger property that the Sprint 1 modules don't even *import* a
network, AWS, Bedrock, or Slack client library, so there is no code path in
this sprint's scope that could make such a call in the first place.
"""

from __future__ import annotations

import ast
from pathlib import Path

import dbt_fixer

SRC_ROOT = Path(dbt_fixer.__file__).resolve().parent

# Every module Sprint 1 introduces. Later sprints will add their own
# necessarily-networked modules (the Bedrock model wrapper, the Slack
# client, the auditor subprocess integration); this list is intentionally
# scoped to Sprint 1's contract, bounds, fencing, and intake modules only.
SPRINT1_MODULES = (
    "__init__.py",
    "_numeric.py",
    "bounds.py",
    "env.py",
    "fencing.py",
    "intake.py",
    "logging_utils.py",
    "pipeline.py",
    "scratch.py",
    "status.py",
)

FORBIDDEN_IMPORT_ROOTS = {
    "boto3",
    "botocore",
    "agno",
    "slack_sdk",
    "requests",
    "urllib",
    "urllib3",
    "http",
    "socket",
    "subprocess",
    "asyncio",
}


def _imported_roots(source: str) -> set[str]:
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_every_sprint1_module_exists():
    for filename in SPRINT1_MODULES:
        assert (SRC_ROOT / filename).exists(), f"expected Sprint 1 module {filename} to exist"


def test_sprint1_modules_import_no_network_or_model_libraries():
    offenders: dict[str, set[str]] = {}
    for filename in SPRINT1_MODULES:
        path = SRC_ROOT / filename
        roots = _imported_roots(path.read_text())
        forbidden = roots & FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            offenders[filename] = forbidden
    assert not offenders, f"forbidden imports found: {offenders}"


def test_sprint1_modules_contain_no_literal_network_or_subprocess_calls():
    # Belt-and-suspenders grep-level check: even a dynamically-constructed
    # import (e.g. `importlib.import_module("boto3")`) would still leave a
    # literal trace of the library name in the source text.
    forbidden_tokens = ("boto3", "botocore", "slack_sdk", "agno.models", "subprocess.")
    offenders: dict[str, list[str]] = {}
    for filename in SPRINT1_MODULES:
        text = (SRC_ROOT / filename).read_text()
        hits = [token for token in forbidden_tokens if token in text]
        if hits:
            offenders[filename] = hits
    assert not offenders, f"forbidden tokens found: {offenders}"
