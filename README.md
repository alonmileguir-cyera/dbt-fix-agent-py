# dbt Fix Agent (Shadow Mode)

A sealed, single-purpose Python package that proposes narrowly-scoped,
mechanically-gated repairs for a known-red dbt Cloud CI check or an
auditor-`BLOCKED` PR, proves the fix through independent adversarial gates,
and posts the result to Slack. **It never writes to GitHub** — there is no
write-capable credential in its environment and no code path that pushes,
comments, or opens a PR.

This README documents the package's environment contract as each sprint
lands. Sprint 1 establishes the contract's shape (fail-closed required
variables, fail-safe-default numeric bounds) and the variables owned by that
sprint's modules; later sprints add their own rows to the tables below as
their features (Bedrock model access, Slack delivery, the auditor subprocess
integration) land.

## Running the tests

```
pip install -e ".[test]"
pytest
```

The suite is fully offline: `tests/conftest.py` actively blocks real network
sockets and real subprocess spawns for every test, except a test explicitly
marked `@pytest.mark.real_process` (reserved, starting in a later sprint, for
one clearly-marked real-process integration module).

## Environment contract

### Core run configuration (`dbt_fixer.env`)

| Variable | Required | Default when unset | Malformed-value handling |
|---|---|---|---|
| `DBT_FIXER_FAILURE_KIND` | **yes** | n/a | Missing/blank/invalid (must be `ci` or `audit`) → `EnvValidationError`, run resolves `failed`. |
| `DBT_FIXER_REPO_PATH` | **yes** | n/a | Missing/blank, or path does not exist / is not a directory → `EnvValidationError`, run resolves `failed`. |
| `DBT_FIXER_PR_TITLE` | no | `""` | Free text; no validation. |
| `DBT_FIXER_PR_DESCRIPTION` | no | `""` | Free text; no validation. |
| `DBT_FIXER_PR_DIFF` | no | `""` | Free text; no validation. |
| `DBT_FIXER_PR_URL` | no | `""` | Free text; no validation. |
| `DBT_FIXER_FAILURE_CONTEXT` | no | `""` | Free text; an empty or unparseable value is handled by `dbt_fixer.intake`, resolving the run to `no_safe_fix` with a specific reason — never treated as an environment error. |
| `DBT_FIXER_SLACK_CHANNEL` | no | `None` | Free text; unset means Slack delivery is skipped (a no-op), not an error. |
| `DBT_FIXER_AUDITOR_PYTHON` | no | `None` | Free text path to the sibling auditor's interpreter; unset is a hard `no_safe_fix` at re-audit-gate time (a later sprint), never a skipped gate. |
| `DBT_FIXER_MAX_ROUNDS` | no | `3` | Non-numeric, or outside `[1, 10]` → falls back to `3` and records a warning (never crashes, never clamps to the nearest bound). |

### Bounded-execution primitive (`dbt_fixer.bounds`)

Every model-calling pass in this package runs through the same
`ExecutionBudget`, which enforces these three limits independently and
simultaneously:

| Variable | Required | Default when unset | Valid range | Malformed-value handling |
|---|---|---|---|---|
| `DBT_FIXER_TIMEOUT_SECONDS` | no | `300` | `[1, 3600]` | Falls back to `300` and records a warning. |
| `DBT_FIXER_MAX_TOOL_CALLS` | no | `40` | `[1, 500]` | Falls back to `40` and records a warning. |
| `DBT_FIXER_MAX_TURNS` | no | `8` | `[1, 100]` | Falls back to `8` and records a warning. |

None of these variables ever raise: an out-of-range or non-numeric value
degrades to the documented default rather than crashing the process or
silently clamping to the nearest valid bound.

## Package layout

```
src/dbt_fixer/
  env.py            # DBT_FIXER_* required/optional contract, fail-closed on required
  bounds.py          # timeout/tool-call-cap/turn-limit primitive, fail-safe on malformed
  _numeric.py        # shared fail-safe numeric-bound parsing helper
  scratch.py         # scratch-copy lifecycle (create, use, guaranteed cleanup)
  fencing.py         # untrusted-content fencing + lookalike-marker neutralization
  intake.py          # failure-context -> structured target, or an honest no_safe_fix
  pipeline.py        # stage-1 orchestration: env + intake -> terminal RunResult or continue
  status.py          # the fixed proposed/no_safe_fix/failed vocabulary and glyphs
  logging_utils.py   # stderr-only diagnostic logging (stdout stays a clean machine surface)
```

Later sprints add the path-safe repo tools, the Bedrock-backed proposal
pass, the scratch-copy edit applier and diff generation, the allowlist and
re-audit gates, the fix-refuter and `dbt parse` gates, the bounded retry
loop, and the Slack/stdout delivery contract — each with its own additions
to this README's environment-contract tables.
