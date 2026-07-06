"""The `DBT_FIXER_*` environment contract.

A single, fixed set of environment variables fully controls a run. Anything
missing or malformed in a *required* field fails closed -- raises
`EnvValidationError` -- rather than guessing a default. Optional fields have
documented, safe defaults.

This module intentionally does not read the bound-override variables
(`DBT_FIXER_TIMEOUT_SECONDS`, `DBT_FIXER_MAX_TOOL_CALLS`, `DBT_FIXER_MAX_TURNS`);
those are owned by `dbt_fixer.bounds`, which has its own fall-back-to-safe-default
semantics distinct from this module's fail-closed semantics (see that module's
docstring for the full accounting of which module owns which variable).

Environment-contract table (this module's slice of it):

| Variable                      | Required | Default when unset | Malformed handling                     |
|--------------------------------|----------|---------------------|------------------------------------------|
| `DBT_FIXER_FAILURE_KIND`       | yes      | n/a                 | `EnvValidationError` (must be `ci`/`audit`) |
| `DBT_FIXER_REPO_PATH`          | yes      | n/a                 | `EnvValidationError` (must exist, be a dir) |
| `DBT_FIXER_PR_TITLE`           | no       | `""`                | n/a (free text)                          |
| `DBT_FIXER_PR_DESCRIPTION`     | no       | `""`                | n/a (free text)                          |
| `DBT_FIXER_PR_DIFF`            | no       | `""`                | n/a (free text)                          |
| `DBT_FIXER_PR_URL`             | no       | `""`                | n/a (free text)                          |
| `DBT_FIXER_FAILURE_CONTEXT`    | no       | `""`                | n/a; unparseable content is handled by `dbt_fixer.intake`, never here |
| `DBT_FIXER_SLACK_CHANNEL`      | no       | `None`              | n/a (free text; absent channel means Slack delivery is skipped, not an error) |
| `DBT_FIXER_AUDITOR_PYTHON`     | no       | `None`              | n/a (free text path; absence is a hard `no_safe_fix` at re-audit-gate time, not here) |
| `DBT_FIXER_MAX_ROUNDS`         | no       | `3`                 | non-numeric/out-of-`[1, 10]` -> falls back to `3`, records a warning |

`FixerConfig.warnings` carries every fallback-to-default explanation so a
malformed value is never silently substituted -- it is always at least
observable by the caller.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Optional

from ._numeric import parse_bounded_number

FailureKind = Literal["ci", "audit"]
_VALID_FAILURE_KINDS: tuple[str, ...] = ("ci", "audit")

DEFAULT_MAX_ROUNDS = 3
_MAX_ROUNDS_RANGE = (1, 10)  # sanity ceiling; never let a malformed-but-numeric value

ENV_FAILURE_KIND = "DBT_FIXER_FAILURE_KIND"
ENV_REPO_PATH = "DBT_FIXER_REPO_PATH"
ENV_PR_TITLE = "DBT_FIXER_PR_TITLE"
ENV_PR_DESCRIPTION = "DBT_FIXER_PR_DESCRIPTION"
ENV_PR_DIFF = "DBT_FIXER_PR_DIFF"
ENV_PR_URL = "DBT_FIXER_PR_URL"
ENV_FAILURE_CONTEXT = "DBT_FIXER_FAILURE_CONTEXT"
ENV_SLACK_CHANNEL = "DBT_FIXER_SLACK_CHANNEL"
ENV_AUDITOR_PYTHON = "DBT_FIXER_AUDITOR_PYTHON"
ENV_MAX_ROUNDS = "DBT_FIXER_MAX_ROUNDS"


class EnvValidationError(ValueError):
    """Raised when a required environment variable is missing or invalid.

    Callers must treat this as a fail-closed signal (status `failed`), never
    attempt to guess a value for the offending field.
    """


@dataclass(frozen=True)
class FixerConfig:
    """The fully-validated configuration for one dbt_fixer invocation."""

    failure_kind: FailureKind
    repo_path: Path

    pr_title: str = ""
    pr_description: str = ""
    pr_diff: str = ""
    pr_url: str = ""
    failure_context: str = ""
    slack_channel: Optional[str] = None
    auditor_python: Optional[str] = None
    max_rounds: int = DEFAULT_MAX_ROUNDS

    warnings: tuple[str, ...] = field(default_factory=tuple)


def _require_nonempty(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if value is None or value.strip() == "":
        raise EnvValidationError(f"{name} is required and must be non-empty")
    return value


def _parse_max_rounds(env: Mapping[str, str], warnings: list[str]) -> int:
    return parse_bounded_number(
        env,
        ENV_MAX_ROUNDS,
        default=DEFAULT_MAX_ROUNDS,
        min_value=_MAX_ROUNDS_RANGE[0],
        max_value=_MAX_ROUNDS_RANGE[1],
        warnings=warnings,
        caster=int,
    )


def load_config(env: Optional[Mapping[str, str]] = None) -> FixerConfig:
    """Parse and validate the `DBT_FIXER_*` environment contract.

    Raises `EnvValidationError` (fail closed) if a required variable is
    missing or invalid. Never silently substitutes a guessed value for a
    required field.
    """

    if env is None:
        env = os.environ

    failure_kind_raw = _require_nonempty(env, ENV_FAILURE_KIND)
    if failure_kind_raw not in _VALID_FAILURE_KINDS:
        raise EnvValidationError(
            f"{ENV_FAILURE_KIND}={failure_kind_raw!r} is invalid; must be one of "
            f"{_VALID_FAILURE_KINDS}"
        )

    repo_path_raw = _require_nonempty(env, ENV_REPO_PATH)
    repo_path = Path(repo_path_raw)
    if not repo_path.exists():
        raise EnvValidationError(f"{ENV_REPO_PATH}={repo_path_raw!r} does not exist")
    if not repo_path.is_dir():
        raise EnvValidationError(f"{ENV_REPO_PATH}={repo_path_raw!r} is not a directory")

    warnings: list[str] = []
    max_rounds = _parse_max_rounds(env, warnings)

    slack_channel = env.get(ENV_SLACK_CHANNEL) or None
    auditor_python = env.get(ENV_AUDITOR_PYTHON) or None

    return FixerConfig(
        failure_kind=failure_kind_raw,  # type: ignore[arg-type]
        repo_path=repo_path,
        pr_title=env.get(ENV_PR_TITLE, "") or "",
        pr_description=env.get(ENV_PR_DESCRIPTION, "") or "",
        pr_diff=env.get(ENV_PR_DIFF, "") or "",
        pr_url=env.get(ENV_PR_URL, "") or "",
        failure_context=env.get(ENV_FAILURE_CONTEXT, "") or "",
        slack_channel=slack_channel,
        auditor_python=auditor_python,
        max_rounds=max_rounds,
        warnings=tuple(warnings),
    )
