"""`python -m dbt_fixer.entrypoint` -- the package's one, always-exit-0 CLI surface.

This module is the outermost boundary of the whole package. No matter what
happens underneath it -- a missing required env var, an unparseable failure
context, a fully-run fix attempt, a Slack outage, or a genuinely unexpected
bug -- the process:

1. exits ``0``, always;
2. emits *exactly one* line matching
   ``^dbt-fixer-status: (proposed|no_safe_fix|failed)$`` as the *last*
   line of stdout;
3. when (and only when) that status is ``proposed``, emits exactly one
   ``dbt-fixer-patch-begin``/``dbt-fixer-patch-end`` pair bracketing the
   winning candidate's unified diff, with any secret-shaped substring
   already scrubbed identically to the copy posted to Slack;
4. never lets a raw traceback become the last thing printed.

**Pipeline wiring.** ``compute_entrypoint_outcome`` runs the full pipeline:
Stage 1 (`dbt_fixer.pipeline.run_stage1`: environment validation + intake) is
always run first; if it resolves to a terminal result (bad environment, or
an unparseable failure context), that result is final. Otherwise the bounded
propose/apply/gate loop (`dbt_fixer.retry_loop.run_bounded_fix_attempt`) is
run against real, production seams (`dbt_fixer.runners`: a real Bedrock-
backed model runner for the proposal pass, a second, independently-
constructed one for the fix-refuter pass, and real subprocess runners for
the sealed-auditor re-audit and ``dbt parse`` gates). Whatever that attempt
resolves to -- ``proposed``, ``no_safe_fix``, or ``failed`` -- is reported,
unconditionally, to Slack (`dbt_fixer.slack_delivery.deliver_shadow_report`,
which never raises and never influences this already-computed result) and
then to stdout.

Every one of these stages already converts its own expected failure modes
into a typed, non-exceptional result; the `except Exception` backstops in
this module exist only so a defect in *this file's own wiring* -- not in any
of the modules it calls -- still resolves to an honest `failed` status
instead of an unhandled crash escaping to a traceback.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from .bounds import ExecutionBudget, load_bounds
from .env import FixerConfig
from .fencing import FencedContext
from .intake import FailureTarget
from .logging_utils import get_logger
from .pipeline import run_stage1
from .redaction import redact_secrets
from .retry_loop import FixAttemptResult, run_bounded_fix_attempt
from .runners import (
    build_real_model_runner,
    build_real_refuter_runner,
    real_dbt_subprocess_runner,
    real_reaudit_subprocess_runner,
)
from .slack_delivery import SlackDeliveryResult, deliver_shadow_report
from .status import (
    STDOUT_PATCH_BEGIN,
    STDOUT_PATCH_END,
    STDOUT_REASON_PREFIX,
    STDOUT_STATUS_PREFIX,
    RunResult,
)

__all__ = [
    "EntrypointOutcome",
    "FixAttemptRunner",
    "SlackDeliverer",
    "compute_entrypoint_outcome",
    "compute_run_result",
    "render_stdout_lines",
    "main",
]

logger = get_logger("entrypoint")

_FALLBACK_STATUS = "failed"
_FALLBACK_REASON = (
    "an unexpected internal error occurred before a result could be computed"
)

# Injectable seam type aliases -- production code passes the real
# implementations below; tests inject fakes so the offline suite never runs
# a real fix attempt or contacts a real Slack workspace.
FixAttemptRunner = Callable[
    [FixerConfig, FailureTarget, FencedContext, "str"], FixAttemptResult
]
SlackDeliverer = Callable[..., SlackDeliveryResult]


@dataclass(frozen=True)
class EntrypointOutcome:
    """The full outcome of one entrypoint run: the terminal `RunResult`, plus
    the winning candidate diff when (and only when) `run_result.status ==
    "proposed"`."""

    run_result: RunResult
    diff: Optional[str] = None


def default_run_fix_attempt(
    config: FixerConfig,
    target: FailureTarget,
    fenced_context: FencedContext,
    repo_root: "str",
) -> FixAttemptResult:
    """The real, production `FixAttemptRunner`: wires real Bedrock-backed
    model runners and real subprocess runners into
    `dbt_fixer.retry_loop.run_bounded_fix_attempt`.

    The proposal pass and the fix-refuter pass each get their own,
    independently-constructed `Agent` and `ExecutionBudget` -- the refuter
    never shares conversation state or remaining tool-call allowance with
    the proposal pass it did not write.
    """

    bounds, _warnings = load_bounds()
    proposal_budget = ExecutionBudget(bounds)
    refuter_budget = ExecutionBudget(bounds)

    model_runner = build_real_model_runner(repo_root, proposal_budget)
    refuter_runner = build_real_refuter_runner(repo_root, refuter_budget)

    return run_bounded_fix_attempt(
        config=config,
        target=target,
        fenced_context=fenced_context,
        repo_root=repo_root,
        model_runner=model_runner,
        subprocess_runner=real_reaudit_subprocess_runner,
        refuter_runner=refuter_runner,
        dbt_subprocess_runner=real_dbt_subprocess_runner,
        budget=proposal_budget,
    )


def _safe_deliver_to_slack(
    deliver_slack: SlackDeliverer,
    *,
    run_result: RunResult,
    config: Optional[FixerConfig],
    diff: Optional[str],
    env: Optional[Mapping[str, str]],
) -> SlackDeliveryResult:
    """Call `deliver_slack` and never let it affect this run's outcome.

    `deliver_shadow_report` (the real, default `deliver_slack`) already
    never raises on its own; this wrapper is a second, defensive backstop
    in case an injected fake in a test -- or a future change to the real
    implementation -- ever does.
    """

    channel = config.slack_channel if config is not None else None
    failure_kind = config.failure_kind if config is not None else "ci"
    pr_url = config.pr_url if config is not None else ""

    try:
        return deliver_slack(
            run_result=run_result,
            failure_kind=failure_kind,
            pr_url=pr_url,
            candidate_diff=diff or "",
            channel=channel,
            token_env=dict(env) if env is not None else None,
        )
    except Exception as exc:  # noqa: BLE001 - Slack delivery must never affect the run's outcome.
        logger.exception("unexpected error delivering the shadow report to Slack: %s", exc)
        return SlackDeliveryResult(
            skipped=True,
            summary_posted=False,
            summary_ts=None,
            detail_chunks_posted=0,
            detail_chunks_total=0,
            reason=f"unexpected error calling Slack delivery: {exc!r}",
        )


def compute_entrypoint_outcome(
    env: Optional[Mapping[str, str]] = None,
    *,
    run_fix_attempt: Optional[FixAttemptRunner] = None,
    deliver_slack: Optional[SlackDeliverer] = None,
) -> EntrypointOutcome:
    """Compute this run's single, terminal `EntrypointOutcome`. Never raises.

    Runs Stage 1 first; a terminal Stage 1 result (bad environment, or an
    unparseable failure context) is final. Otherwise runs the bounded fix
    attempt via `run_fix_attempt` (the real pipeline in production; an
    injected fake in every test). Whatever the final `RunResult` is, it is
    unconditionally handed to `deliver_slack` before this function returns
    -- Slack delivery never influences, and is never allowed to raise past,
    the already-computed result.

    `run_fix_attempt`/`deliver_slack` default to `None` and are resolved to
    this module's own `default_run_fix_attempt`/`deliver_shadow_report`
    *by name, at call time* (deliberately, not as literal default-argument
    values bound once at import time) -- so a test can monkeypatch either
    module-level attribute and have that fake actually take effect here,
    including when the fake is installed via `main`.
    """

    if run_fix_attempt is None:
        run_fix_attempt = default_run_fix_attempt
    if deliver_slack is None:
        deliver_slack = deliver_shadow_report

    try:
        stage1 = run_stage1(env)
    except Exception as exc:  # pragma: no cover - defensive: run_stage1 never raises today
        logger.exception("unexpected error running stage 1: %s", exc)
        run_result = RunResult(
            status="failed", reason=f"unexpected internal error in stage 1: {exc!r}"
        )
        _safe_deliver_to_slack(deliver_slack, run_result=run_result, config=None, diff=None, env=env)
        return EntrypointOutcome(run_result=run_result)

    if stage1.terminal is not None:
        _safe_deliver_to_slack(
            deliver_slack, run_result=stage1.terminal, config=stage1.config, diff=None, env=env
        )
        return EntrypointOutcome(run_result=stage1.terminal)

    # Stage 1 succeeded: a validated config and a concrete `FailureTarget`
    # are both available. `run_stage1`'s own contract guarantees this.
    assert stage1.config is not None and stage1.intake is not None
    config = stage1.config
    target = stage1.intake.target
    fenced_context = stage1.intake.fenced_context
    assert target is not None and fenced_context is not None

    try:
        attempt = run_fix_attempt(config, target, fenced_context, str(config.repo_path))
    except Exception as exc:
        logger.exception("unexpected error running the bounded fix attempt: %s", exc)
        run_result = RunResult(
            status="failed", reason=f"unexpected error running the fix pipeline: {exc!r}"
        )
        _safe_deliver_to_slack(deliver_slack, run_result=run_result, config=config, diff=None, env=env)
        return EntrypointOutcome(run_result=run_result)

    _safe_deliver_to_slack(
        deliver_slack, run_result=attempt.run_result, config=config, diff=attempt.diff, env=env
    )
    return EntrypointOutcome(run_result=attempt.run_result, diff=attempt.diff)


def compute_run_result(env: Optional[Mapping[str, str]] = None) -> RunResult:
    """Backwards-compatible convenience wrapper: just the `RunResult`."""

    return compute_entrypoint_outcome(env).run_result


def _single_line(text: str) -> str:
    """Collapse `text` to one line so it can never masquerade as, or push
    past, the fixed-shape status line that must be the true last line of
    stdout."""

    return " ".join(text.split())


def render_stdout_lines(outcome: "RunResult | EntrypointOutcome") -> list[str]:
    """Render `outcome` as the fixed stdout lines.

    Accepts either a bare `RunResult` (no patch block is ever emitted; the
    legacy shape from Sprint 1) or a full `EntrypointOutcome`. Always
    produces, in order:

    1. an optional reason line;
    2. when `status == "proposed"` and a non-blank diff is present: exactly
       one `dbt-fixer-patch-begin` / `dbt-fixer-patch-end` pair bracketing
       the diff, with any secret-shaped substring already redacted
       identically to the copy delivered to Slack;
    3. the single, line-anchored status line, always last.

    When `status != "proposed"`, no patch block anchor ever appears, even
    if a `diff` happens to be present on the outcome.
    """

    if isinstance(outcome, EntrypointOutcome):
        run_result = outcome.run_result
        diff = outcome.diff
    else:
        run_result = outcome
        diff = None

    lines: list[str] = []
    if run_result.reason:
        lines.append(f"{STDOUT_REASON_PREFIX}: {_single_line(redact_secrets(run_result.reason))}")

    if run_result.status == "proposed" and diff and diff.strip():
        lines.append(STDOUT_PATCH_BEGIN)
        lines.append(redact_secrets(diff).rstrip("\n"))
        lines.append(STDOUT_PATCH_END)

    lines.append(f"{STDOUT_STATUS_PREFIX}: {run_result.status}")
    return lines


def main(argv: Optional[list[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    """Run one dbt_fixer pass and print its fixed stdout contract.

    Always returns `0`. Guarantees the status line is printed exactly once,
    as the last line of stdout, even if computing or rendering the result
    itself fails unexpectedly.
    """

    status = _FALLBACK_STATUS
    lines_before_status: list[str] = []

    try:
        outcome = compute_entrypoint_outcome(env)
        status = outcome.run_result.status
        lines_before_status = render_stdout_lines(outcome)[:-1]
    except Exception as exc:  # pragma: no cover - defensive: compute_entrypoint_outcome never raises
        logger.exception("unhandled exception computing the entrypoint outcome: %s", exc)
        lines_before_status = [f"{STDOUT_REASON_PREFIX}: {_single_line(_FALLBACK_REASON)}"]

    for line in lines_before_status:
        try:
            print(line)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("failed to print a stdout line: %s", exc)

    # The status line is always the true last thing printed, regardless of
    # whether anything above it succeeded.
    try:
        print(f"{STDOUT_STATUS_PREFIX}: {status}")
    except Exception as exc:  # pragma: no cover - stdout itself is broken; nothing left to do
        logger.exception("failed to print status line: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
