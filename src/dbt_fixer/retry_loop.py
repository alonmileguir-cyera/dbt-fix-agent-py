"""Stage 2+3: the bounded retry loop and terminal status resolution.

This is the seam that wires the structured-fix-proposal pass
(`dbt_fixer.fix_pipeline.run_fix_pipeline`) and all four gates
(`dbt_fixer.allowlist.run_allowlist_gate`,
`dbt_fixer.reaudit.run_reaudit_gate`,
`dbt_fixer.refuter.run_fix_refuter_gate`,
`dbt_fixer.dbt_parse.run_dbt_parse_gate`) into a single bounded attempt:

    propose -> apply -> allowlist -> re-audit -> fix-refuter -> dbt parse

repeated for at most `FixerConfig.max_rounds` rounds, in exactly
`dbt_fixer.status.GATE_ORDER`. Each round is independent: a fresh proposal
is generated every time (never the same candidate re-submitted), and a
rejected round's *specific* violation reason is fed back into the next
round's proposal prompt via `dbt_fixer.proposal.build_proposal_prompt`'s
`feedback` parameter, so the model has a concrete reason to change course
rather than repeating itself.

**The dbt parse gate is the one non-authoritative gate.** Allowlist and
re-audit must both pass for a round to proceed; the fix-refuter gate must
also affirmatively pass (an unambiguous "could not refute"). The dbt parse
gate, by contrast, only ever *blocks* a round when it actually ran and
failed (a real nonzero exit or timeout) -- a `"skipped"` outcome (no `dbt`
on PATH, or a scratch/apply-setup problem) never itself blocks a round,
and never itself grants one either: it is simply recorded, visibly, as
skipped in that round's gate list, and the round's outcome is decided
entirely by the three required gates.

Terminal status resolution is exactly the closed vocabulary in
`dbt_fixer.status`:

- **`proposed`** -- reachable *only* when every required gate passed, in
  the same round, for the same candidate diff. A pass on round 2's
  allowlist check never combines with a pass on round 1's re-audit check.
- **`no_safe_fix`** -- every round was tried (or the auditor interpreter is
  missing/uninvokable, which is a hard stop no amount of retrying can fix)
  and none produced a fully-passing candidate. This is the honest "we
  looked, and there is nothing safe to propose" outcome.
- **`failed`** -- reserved for a genuinely unexpected error (a bug in this
  package, not a bad candidate or a bad environment) that none of the
  narrower, already-never-raising layers underneath this one caught. Every
  one of those layers (`run_fix_pipeline`, `run_allowlist_gate`,
  `run_reaudit_gate`, `run_fix_refuter_gate`, `run_dbt_parse_gate`) already
  resolves its own ordinary failure modes to a typed, non-exceptional
  result; this module's own `except Exception` backstop exists only so a
  defect in this wiring itself still resolves to an honest status instead
  of an unhandled crash.

Every scratch directory involved (one per `run_fix_pipeline` call, one per
gate call) is created and torn down entirely inside those callees --
`run_bounded_fix_attempt` itself never opens a scratch copy of its own, and
never raises, so the process this eventually backs always exits cleanly
regardless of which round or gate is the one that ends the attempt.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from .allowlist import AllowlistCaps, AllowlistVerdict, run_allowlist_gate
from .bounds import ExecutionBudget
from .dbt_parse import DbtParseVerdict, DbtSubprocessRunner, WhichFunc, run_dbt_parse_gate
from .env import FixerConfig
from .fencing import FencedContext
from .fix_pipeline import run_fix_pipeline
from .proposal import extract_named_paths, render_preloaded_files
from .intake import FailureTarget
from .proposal import ModelRunner
from .reaudit import ReAuditVerdict, SubprocessRunner, run_reaudit_gate
from .refuter import RefuterRunner, RefuterVerdict, run_fix_refuter_gate
from .status import GateResult, RunResult

__all__ = ["FixAttemptResult", "run_bounded_fix_attempt"]

AllowlistGateRunner = Callable[..., AllowlistVerdict]
ReAuditGateRunner = Callable[..., ReAuditVerdict]
RefuterGateRunner = Callable[..., RefuterVerdict]
DbtParseGateRunner = Callable[..., DbtParseVerdict]


@dataclass(frozen=True)
class FixAttemptResult:
    """The full outcome of one bounded fix attempt.

    `run_result` is the authoritative, closed-vocabulary status (see module
    docstring). `diff` is the winning candidate's unified diff text --
    always `None` unless `run_result.status == "proposed"`, in which case it
    is exactly the diff that passed every gate this same round.
    `rounds_used` counts how many propose/apply/gate rounds were actually
    attempted before this attempt resolved (always `>= 1` unless
    `max_rounds` itself is somehow `<= 0`, which `dbt_fixer.env` never
    produces).
    """

    run_result: RunResult
    diff: Optional[str] = None
    rounds_used: int = 0


def _skip_gate(name: str, reason: str) -> GateResult:
    return GateResult(name=name, outcome="skipped", detail=reason)


def run_bounded_fix_attempt(
    *,
    config: FixerConfig,
    target: FailureTarget,
    fenced_context: FencedContext,
    repo_root: "str | Path",
    model_runner: ModelRunner,
    subprocess_runner: SubprocessRunner,
    refuter_runner: RefuterRunner,
    dbt_subprocess_runner: DbtSubprocessRunner,
    budget: ExecutionBudget,
    allowlist_gate: AllowlistGateRunner = run_allowlist_gate,
    reaudit_gate: ReAuditGateRunner = run_reaudit_gate,
    refuter_gate: RefuterGateRunner = run_fix_refuter_gate,
    dbt_parse_gate: DbtParseGateRunner = run_dbt_parse_gate,
    which: WhichFunc = shutil.which,
) -> FixAttemptResult:
    """Run the bounded propose/apply/allowlist/re-audit/fix-refuter/dbt-parse
    loop for one attempt.

    Args:
        config: The validated run configuration; `max_rounds`,
            `max_changed_files`, `max_changed_lines`, `reaudit_timeout_seconds`,
            `refuter_timeout_seconds`, `dbt_parse_timeout_seconds`,
            `auditor_python`, `failure_kind`, and every `pr_*` field are all
            read from here.
        target: The parsed `FailureTarget` from `dbt_fixer.intake` -- its
            `checks` are passed to the allowlist gate (for kind="audit"
            test-weakening proof) and its `identifiers` to the re-audit gate
            (as the set of originally-failing checks that must now pass).
        fenced_context: The already-fenced failure/PR context rendered into
            every round's proposal prompt, and into every round's
            fix-refuter prompt.
        repo_root: The original, never-mutated checkout root.
        model_runner: The structured-fix-proposal model runner.
        subprocess_runner: The injectable auditor-subprocess runner (a fake
            in every test; never a real subprocess).
        refuter_runner: The injectable fix-refuter model runner (a fake in
            every test). Callers must supply one that starts a fresh,
            isolated context per call -- never the same conversation the
            proposal pass used.
        dbt_subprocess_runner: The injectable `dbt parse` subprocess runner
            (a fake in every test; never a real subprocess).
        budget: The shared `ExecutionBudget` bounding *every* round's model
            call cumulatively -- a slow or stalled model call in round 1
            can still exhaust the budget and end round 2 early.
        allowlist_gate: Injectable seam for the allowlist gate, defaulting
            to the real `run_allowlist_gate`. Overridable in tests to force
            an unexpected exception and exercise the `failed` backstop.
        reaudit_gate: Same, for the re-audit gate.
        refuter_gate: Same, for the fix-refuter gate, defaulting to the real
            `run_fix_refuter_gate`.
        dbt_parse_gate: Same, for the dbt parse gate, defaulting to the real
            `run_dbt_parse_gate`.
        which: Injectable PATH-lookup callable handed through to the dbt
            parse gate, defaulting to `shutil.which`.

    Returns:
        A `FixAttemptResult`. Never raises: any exception escaping the
        per-round gate/pipeline calls below (which themselves already never
        raise for ordinary failure modes) is caught by the outermost
        backstop and resolved to `status="failed"`.
    """

    repo_root = Path(repo_root)
    caps = AllowlistCaps(
        max_changed_files=config.max_changed_files,
        max_changed_lines=config.max_changed_lines,
    )
    # Only checks the fixer is responsible for clearing (excludes known
    # advisory checks - they never block, and the allowlist forbids the
    # doc/style edits that would clear them). See FailureTarget.
    originally_failing_ids: Tuple[str, ...] = target.blocking_identifiers

    # Pre-load the files the findings name, once, so the model doesn't burn
    # its tool-call/wall-clock budget rediscovering what the audit already
    # pointed at. Path-safe and best-effort; "" when nothing resolves.
    _named_paths = extract_named_paths(
        [c.evidence for c in target.checks] + [c.suggestion for c in target.checks]
    )
    preloaded_files = render_preloaded_files(repo_root, _named_paths)

    feedback: Optional[str] = None
    last_reason = "no candidate was proposed"
    last_gates: list[GateResult] = []
    rounds_used = 0

    try:
        for round_num in range(1, config.max_rounds + 1):
            rounds_used = round_num

            pipeline_result = run_fix_pipeline(
                repo_root, fenced_context, model_runner, budget,
                feedback=feedback, preloaded_files=preloaded_files,
                blocking_scope=originally_failing_ids,
            )
            if not pipeline_result.ok:
                last_reason = pipeline_result.reason or "no proposal was produced"
                last_gates = []
                feedback = f"round {round_num}: {last_reason}"
                continue

            candidate_diff = pipeline_result.diff or ""

            allowlist_verdict = allowlist_gate(
                repo_root=repo_root,
                candidate_diff=candidate_diff,
                pr_diff=config.pr_diff,
                failure_kind=config.failure_kind,
                failing_checks=target.checks,
                caps=caps,
            )
            if not allowlist_verdict.passed:
                last_gates = [
                    GateResult(name="allowlist", outcome="fail", detail=allowlist_verdict.reason),
                    _skip_gate("re-audit", "allowlist gate rejected this candidate"),
                    _skip_gate("fix-refuter", "allowlist gate rejected this candidate"),
                    _skip_gate("dbt parse", "allowlist gate rejected this candidate"),
                ]
                last_reason = allowlist_verdict.reason
                feedback = (
                    f"round {round_num} was rejected by the allowlist gate "
                    f"({allowlist_verdict.violation}): {allowlist_verdict.reason}"
                )
                continue

            reaudit_verdict = reaudit_gate(
                repo_root=repo_root,
                candidate_diff=candidate_diff,
                pr_diff=config.pr_diff,
                pr_title=config.pr_title,
                pr_description=config.pr_description,
                pr_url=config.pr_url,
                auditor_python=config.auditor_python,
                failure_kind=config.failure_kind,
                originally_failing_check_ids=originally_failing_ids,
                timeout_seconds=config.reaudit_timeout_seconds,
                subprocess_runner=subprocess_runner,
            )

            allowlist_pass_gate = GateResult(
                name="allowlist", outcome="pass", detail=allowlist_verdict.reason
            )

            if reaudit_verdict.hard_no_safe_fix:
                # A missing/uninvokable auditor interpreter is a structural
                # problem no further round can fix -- stop immediately
                # rather than burning the remaining rounds.
                last_gates = [
                    allowlist_pass_gate,
                    GateResult(name="re-audit", outcome="fail", detail=reaudit_verdict.reason),
                    _skip_gate("fix-refuter", "re-audit gate reported a hard no-safe-fix condition"),
                    _skip_gate("dbt parse", "re-audit gate reported a hard no-safe-fix condition"),
                ]
                return FixAttemptResult(
                    run_result=RunResult(
                        status="no_safe_fix", reason=reaudit_verdict.reason, gates=last_gates
                    ),
                    rounds_used=rounds_used,
                )

            if not reaudit_verdict.passed:
                last_gates = [
                    allowlist_pass_gate,
                    GateResult(name="re-audit", outcome="fail", detail=reaudit_verdict.reason),
                    _skip_gate("fix-refuter", "re-audit gate rejected this candidate"),
                    _skip_gate("dbt parse", "re-audit gate rejected this candidate"),
                ]
                last_reason = reaudit_verdict.reason
                feedback = (
                    f"round {round_num} was rejected by the re-audit gate "
                    f"({reaudit_verdict.violation}): {reaudit_verdict.reason}"
                )
                continue

            reaudit_pass_gate = GateResult(
                name="re-audit", outcome="pass", detail=reaudit_verdict.reason
            )

            refuter_verdict = refuter_gate(
                fenced_context=fenced_context,
                candidate_diff=candidate_diff,
                refuter_runner=refuter_runner,
                timeout_seconds=config.refuter_timeout_seconds,
            )

            if not refuter_verdict.passed:
                last_gates = [
                    allowlist_pass_gate,
                    reaudit_pass_gate,
                    GateResult(name="fix-refuter", outcome="fail", detail=refuter_verdict.reason),
                    _skip_gate("dbt parse", "fix-refuter gate rejected this candidate"),
                ]
                last_reason = refuter_verdict.reason
                feedback = (
                    f"round {round_num} was rejected by the fix-refuter gate: "
                    f"{refuter_verdict.reason}"
                )
                continue

            fix_refuter_pass_gate = GateResult(
                name="fix-refuter", outcome="pass", detail=refuter_verdict.reason
            )

            dbt_parse_verdict = dbt_parse_gate(
                repo_root=repo_root,
                candidate_diff=candidate_diff,
                timeout_seconds=config.dbt_parse_timeout_seconds,
                subprocess_runner=dbt_subprocess_runner,
                which=which,
            )

            if dbt_parse_verdict.outcome == "failed":
                last_gates = [
                    allowlist_pass_gate,
                    reaudit_pass_gate,
                    fix_refuter_pass_gate,
                    GateResult(name="dbt parse", outcome="fail", detail=dbt_parse_verdict.reason),
                ]
                last_reason = dbt_parse_verdict.reason
                feedback = (
                    f"round {round_num} was rejected by the dbt parse gate: "
                    f"{dbt_parse_verdict.reason}"
                )
                continue

            # The dbt parse gate is non-authoritative: a "skipped" outcome
            # here is recorded honestly but never itself blocks or grants
            # `proposed` -- only allowlist, re-audit, and fix-refuter are
            # required, and all three already passed above.
            dbt_parse_result_gate = GateResult(
                name="dbt parse",
                outcome="skipped" if dbt_parse_verdict.outcome == "skipped" else "pass",
                detail=dbt_parse_verdict.reason,
            )

            # Every required gate passed, this same round, for this same
            # candidate diff -- the only shape `proposed` may be reached by.
            last_gates = [
                allowlist_pass_gate,
                reaudit_pass_gate,
                fix_refuter_pass_gate,
                dbt_parse_result_gate,
            ]
            return FixAttemptResult(
                run_result=RunResult(
                    status="proposed",
                    reason=f"round {round_num} produced a candidate that passed every gate",
                    gates=last_gates,
                ),
                diff=candidate_diff,
                rounds_used=rounds_used,
            )

        return FixAttemptResult(
            run_result=RunResult(
                status="no_safe_fix",
                reason=(
                    f"no candidate passed every gate within {config.max_rounds} round(s); "
                    f"last reason: {last_reason}"
                ),
                gates=last_gates,
            ),
            rounds_used=rounds_used,
        )
    except Exception as exc:  # defensive backstop: a bug here must not crash the process
        return FixAttemptResult(
            run_result=RunResult(
                status="failed",
                reason=f"unexpected error during the bounded fix attempt: {exc!r}",
                gates=last_gates,
            ),
            rounds_used=rounds_used,
        )
