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

## Invocation and the stdout contract

```
python -m dbt_fixer.entrypoint
```

reads its entire configuration from the `DBT_FIXER_*` environment
variables documented below, **always exits `0`**, and prints exactly one
line matching `^dbt-fixer-status: (proposed|no_safe_fix|failed)$` as the
*last* line of stdout — this is the only line a caller needs to grep. When
the run has a specific reason to report (every `failed` and `no_safe_fix`
outcome does), a `dbt-fixer-reason: <single-line reason>` line precedes it.

No input — a missing required variable, a malformed value, an empty or
unparseable failure context, or a genuinely unexpected internal error —
ever produces a non-zero exit code or a raw traceback as the last thing
printed; every one of those cases resolves to `failed` (environment/
internal problems) or `no_safe_fix` (an honest "nothing to fix" or
"identified but couldn't act on it" conclusion) with a stated reason.

**The entrypoint wires the full pipeline.** `dbt_fixer.pipeline.run_stage1`
(environment validation + failure-context intake) always runs first; a
terminal Stage 1 result (bad environment, or an unparseable failure
context) is final. Otherwise the bounded propose/apply/gate loop
(`dbt_fixer.retry_loop.run_bounded_fix_attempt`) runs against real,
production seams (`dbt_fixer.runners`: a Bedrock-backed model runner for
the proposal pass, a fresh tool-free finalizer for empty/malformed proposal
output, an independently-constructed runner for the fix-refuter pass, and
real subprocess runners for the sealed-auditor re-audit and `dbt parse`
gates). The dbt parse gate is disabled by default and can run only after an
explicit trusted `DBT_FIXER_DBT_PARSE_MODE=enabled` opt-in. Whatever that
attempt resolves to — `proposed`,
`no_safe_fix`, or `failed` — is reported, unconditionally, to Slack
(`dbt_fixer.slack_delivery.deliver_shadow_report`, which never raises and
never influences the already-computed result) and then to stdout.

## Fresh install

This package installs and runs with nothing beyond what `pyproject.toml`
declares — no vendored state, no implicit external service required to
*start* (a missing Bedrock/Slack/auditor credential fails closed to
`failed`/`no_safe_fix` with a specific reason, never a crash):

```
python -m venv /tmp/dbt-fixer-fresh-venv
source /tmp/dbt-fixer-fresh-venv/bin/activate
pip install .
python -m dbt_fixer.entrypoint
```

The last command runs with an empty environment. Since `DBT_FIXER_FAILURE_KIND`
and `DBT_FIXER_REPO_PATH` are both required and unset, this prints exactly:

```
dbt-fixer-reason: <a message naming the missing required variable(s)>
dbt-fixer-status: failed
```

and exits `0` — proving the package is importable, its console entrypoint
runs end-to-end, and its fail-closed contract holds, all from a clean
install with no test extras and no fixture scaffolding involved.

## Running the tests

```
pip install -e ".[test]"
pytest
```

The suite is fully offline: `tests/conftest.py` actively blocks real network
sockets and real subprocess spawns for every test, except a test explicitly
marked `@pytest.mark.real_process`, which is excluded from collection by
default (see `[tool.pytest.ini_options]` in `pyproject.toml`) and must be
run explicitly:

```
pytest -m real_process tests/real_process
```

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
| `DBT_FIXER_SLACK_USERNAME` | no | `dbt fixer agent` | Sender name shown on Slack posts (`chat.postMessage` `username` override; requires the bi-automations bot's `chat:write.customize` scope). |
| `DBT_FIXER_SLACK_ICON_EMOJI` | no | `:wrench:` | Sender icon emoji for Slack posts (`icon_emoji` override). |
| `DBT_FIXER_AUDITOR_PYTHON` | no | `None` | Free text path to the sibling auditor's interpreter; unset is a hard `no_safe_fix` at re-audit-gate time (a later sprint), never a skipped gate. |
| `DBT_FIXER_MAX_ROUNDS` | no | `3` | Non-numeric, or outside `[1, 10]` → falls back to `3` and records a warning (never crashes, never clamps to the nearest bound). |
| `DBT_FIXER_MAX_CHANGED_FILES` | no | `5` | Non-numeric, or outside `[1, 50]` → falls back to `5` and records a warning. Enforced by the Sprint 3 allowlist gate. |
| `DBT_FIXER_MAX_CHANGED_LINES` | no | `60` | Non-numeric, or outside `[1, 2000]` → falls back to `60` and records a warning. Enforced by the Sprint 3 allowlist gate. |
| `DBT_FIXER_REAUDIT_TIMEOUT_SECONDS` | no | `120` | Non-numeric, or outside `[1, 1800]` → falls back to `120` and records a warning. Bounds the Sprint 3 re-audit gate's subprocess call. |
| `DBT_FIXER_REFUTER_TIMEOUT_SECONDS` | no | `60` | Non-numeric, or outside `[1, 600]` → falls back to `60` and records a warning. Bounds the Sprint 4 fix-refuter gate's model call; a timeout resolves to a fail-closed "refuted" verdict, never a skipped gate. |
| `DBT_FIXER_DBT_PARSE_MODE` | no | `disabled` | Only explicit `enabled` opts in. Unset/blank stays disabled; any other value fails safe to `disabled` and records a warning. The live `pull_request_target` workflow sets `disabled`. |
| `DBT_FIXER_DBT_PARSE_TIMEOUT_SECONDS` | no | `30` | Non-numeric, or outside `[1, 300]` → falls back to `30` and records a warning. Bounds any `dbt parse`-style structural-validation subprocess call. |

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

`DBT_FIXER_TIMEOUT_SECONDS` is float-typed, so it also explicitly rejects
non-finite values that `float()` would otherwise parse successfully --
`nan`, `inf`, `-inf`, and overflow strings like `1e400` all fall back to
`300` with a recorded warning, exactly like any other malformed value. This
is checked ahead of the range comparison because IEEE 754 makes every
ordering comparison against NaN evaluate to `False`, which would otherwise
let a NaN timeout silently pass the `[1, 3600]` range check and permanently
disable the wall-clock timeout it's supposed to enforce.

## Package layout

```
src/dbt_fixer/
  entrypoint.py     # python -m dbt_fixer.entrypoint: always-exit-0, single-status-line CLI contract
  env.py            # DBT_FIXER_* required/optional contract, fail-closed on required
  bounds.py          # timeout/tool-call-cap/turn-limit primitive, fail-safe on malformed
  _numeric.py        # shared fail-safe numeric-bound parsing helper
  scratch.py         # scratch-copy lifecycle (create, use, guaranteed cleanup)
  fencing.py         # untrusted-content fencing + lookalike-marker neutralization
  intake.py          # failure-context -> structured target, or an honest no_safe_fix
  pipeline.py        # stage-1 orchestration: env + intake -> terminal RunResult or continue
  status.py          # the fixed proposed/no_safe_fix/failed vocabulary and glyphs
  logging_utils.py   # stderr-only diagnostic logging (stdout stays a clean machine surface)
  pathsafe.py         # shared path-containment guard (rejects '..', absolute paths, symlink escapes)
  tools/
    repo_tools.py     # RepoTools: rooted, read-only file read/glob-search -- no write method exists
  model_output.py      # strict whole-response JSON-object extraction from raw model text
  proposal.py           # structured fix-proposal schema (whole_file_replace/line_range_edit) + bounded model pass
  agent.py               # Bedrock/agno agent wiring; the only toolkit it builds exposes read/search only
  applier.py              # fail-closed, two-phase application of a Proposal onto an isolated scratch copy
  diffing.py               # pure difflib unified-diff generation, matching real `git diff` semantics
  fix_pipeline.py           # Stage 2 orchestration: read -> propose -> apply -> diff, fully offline-testable
  diffparse.py               # pure-Python parser/applier for the diffing.py unified-diff dialect
  allowlist.py                # deterministic Allowlist Classifier Gate -- no model call, ever
  reaudit.py                   # Re-Audit Gate: injectable-subprocess integration with the sealed sibling auditor
  retry_loop.py                 # Stage 3: bounded propose/apply/allowlist/re-audit loop + terminal status
```

Later sprints add the fix-refuter and `dbt parse` gates and the Slack/stdout
delivery contract — each with its own additions to this README's
environment-contract tables.

## Sprint 2: path-safe repo tools, structured fix proposal, scratch-copy applier

**No write tool is ever exposed to a model.** `dbt_fixer.tools.repo_tools.RepoTools`
exposes exactly `read_file`/`search_files`, both scoped to a fixed repo root
via `dbt_fixer.pathsafe.resolve_within_root` (rejects non-string/empty,
absolute, and `..`-containing paths, and follows symlinks before the final
containment check). `dbt_fixer.agent.build_repo_toolkit` wraps only those two
methods as the `read_repo_file`/`search_repo_files` agno tools; there is no
create/write/delete/rename capability reachable from anything a model can
call. `read_file` raises `PathTraversalError` for a symlink that escapes the
root; `search_files` instead silently excludes individual escaping matches
found during glob enumeration (while still raising if the `pattern`/
`relative_dir` arguments themselves attempt traversal).

**Structured fix proposals are the only way a fix is ever proposed.**
`dbt_fixer.proposal.parse_proposal` enforces a closed JSON schema (exact
top-level and per-edit key sets, no extra fields, only the three edit types
`whole_file_replace`/`line_range_edit`/`create_file`); any mismatch — malformed
or ambiguous JSON, a missing field, an extra key, an unrecognized edit type, a
single bad edit among otherwise-good ones — resolves to `None` ("no proposal"),
never a partial or guessed acceptance. `dbt_fixer.proposal.run_proposal_pass`
records every turn in the shared `ExecutionBudget` and wraps both the primary
and finalizer model boundaries in the remaining hard wall-clock deadline. A
hanging call, or valid-looking output returned after the deadline, is rejected.
If the primary agent returns empty or malformed output, one additional turn
uses a distinct fresh agent with zero tools; omitting that runner fails closed
instead of reusing the tool-enabled primary. It receives the original request
and separately nonce-fenced prior response, then must either emit the same
closed proposal schema or decline with an empty edit list. Preloaded
PR-controlled repository files are also nonce-fenced, and multiple/echoed JSON
objects are treated as ambiguous. The allowlist, re-audit, refuter, and
dbt-parse gates remain unchanged; the fallback can only produce a candidate
that is applied to the isolated scratch copy, never mutate the original
checkout, and never write to GitHub.

**Edits are applied only to an isolated scratch copy.** `dbt_fixer.applier.apply_proposal`
validates every edit in a proposal (target exists, target is a file, every
line range is in bounds, no two edits conflict) *before* mutating anything;
a single invalid or conflicting edit raises a specific `ApplyError` subclass
and leaves the scratch copy completely untouched. The original checkout
(`dbt_fixer.env.FixerConfig.repo_path`) is never passed to the applier.

**Diffs are pure-Python and match real `git diff`.** `dbt_fixer.diffing.generate_unified_diff`
uses only `difflib.unified_diff` — no subprocess, no real git — and is
verified byte-identical (aside from the `diff --git`/`index`/`new file mode`
header lines, which are normalized away in the comparison) to a real `git
diff` for add-only, delete-only, and mixed-change cases in
`tests/real_process/test_diff_matches_git.py`, the one test module in this
package marked `@pytest.mark.real_process`.

`dbt_fixer.fix_pipeline.run_fix_pipeline` wires all of the above into the
full Stage 2 sequence and is proven, offline, to produce byte-identical diff
output across repeated runs of a fixed fake model runner against a fixed
sample repo.

Two additional, unprefixed environment variables (matching the sibling
`dbt-audit-agent-py` package's operator convention, not part of the
`DBT_FIXER_*` contract above) control Bedrock model selection:

| Variable | Required | Default when unset |
|---|---|---|
| `BEDROCK_MODEL_ID` | no | `us.anthropic.claude-sonnet-5` |
| `AWS_REGION` | no | `us-east-1` |

AWS credentials are always resolved via boto3's default credential chain;
no access key, secret key, or profile is ever hardcoded.

## Sprint 3: the deterministic gates and the bounded retry loop

**The Allowlist Classifier Gate (`dbt_fixer.allowlist`) never calls a model.**
`run_allowlist_gate` is ordinary, deterministic Python control flow over an
already-parsed candidate diff (`dbt_fixer.diffparse`): the same candidate
run through it any number of times always produces the same
`AllowlistVerdict`. It rejects, in this fixed order, a malformed/unparseable
or true no-op candidate; any touched path outside `models/*.{yml,yaml,md,sql}`;
exceeding `DBT_FIXER_MAX_CHANGED_FILES`/`DBT_FIXER_MAX_CHANGED_LINES`, or
touching a hook, a `materialized` config change, or a masking/bypass
keyword (checked independently of the caps); a `.sql` candidate removal that
doesn't consume a matching line the original PR diff added to that same file
(so fixes may correct/revert PR-authored SQL but not delete base SQL); and a
removed/weakened dbt schema test in a `.yml`/`.yaml` file
— categorically for `failure_kind=ci`, or for `failure_kind=audit` only
when none of the originally-failing checks both names that test and
explicitly proves it wrong in its evidence text.

**The Re-Audit Gate (`dbt_fixer.reaudit`) re-runs the sealed sibling
`dbt_auditor` package against the candidate.** `run_reaudit_gate` applies
the candidate diff to its own fresh scratch copy of the checkout and
invokes `dbt_auditor.entrypoint` via an injectable `SubprocessRunner`
(a fake in every test — never a real subprocess) in shadow mode, with no
Slack channel ever set. It distinguishes three failure shapes that are
never conflated: a missing or uninvokable `DBT_FIXER_AUDITOR_PYTHON`
interpreter is a `hard_no_safe_fix` (no amount of retrying fixes a missing
interpreter); a nonzero exit, a timeout, or stdout that doesn't parse into
a recognized `completed`/verdict shape is an ordinary gate violation; and a
`BLOCKED` verdict, or (for `failure_kind=audit`) any originally-failing
check not confirmed passing, is also an ordinary gate violation — distinct
from both a pass and the hard interpreter stop.

**The Bounded Retry Loop (`dbt_fixer.retry_loop`) wires it all together.**
`run_bounded_fix_attempt` repeats propose → apply → allowlist → re-audit
for at most `DBT_FIXER_MAX_ROUNDS` independent rounds (never resubmitting
the same candidate), feeding each rejected round's specific violation
reason into the next round's proposal prompt as feedback
(`dbt_fixer.proposal.build_proposal_prompt`'s new optional `feedback`
parameter). It always resolves to exactly one of the closed
`proposed`/`no_safe_fix`/`failed` vocabulary: `proposed` only when the same
round's same candidate passed every gate; `no_safe_fix` for every ordinary
exhaustion path, including the auditor-interpreter hard stop, which ends
the attempt immediately rather than burning the remaining rounds;
`failed` reserved for a genuine bug backstop, never used for an ordinary
rejected candidate or a missing interpreter. The function itself never
raises, and every scratch directory used along the way is created and torn
down by whichever gate/pipeline call owns it — verified in
`tests/test_retry_loop.py` to leave no `dbt-fixer-scratch-*` temp directory
behind across every terminal outcome, including the `failed` backstop.
