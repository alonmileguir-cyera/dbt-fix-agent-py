"""End-to-end offline pipeline: read -> propose -> apply -> diff.

Wires `dbt_fixer.proposal.run_proposal_pass`, `dbt_fixer.scratch.scratch_copy`,
`dbt_fixer.applier.apply_proposal`, and `dbt_fixer.diffing.generate_unified_diff`
into the single Stage 2 sequence a real run performs:

1. build the bounded model prompt from an already-fenced context
   (`dbt_fixer.proposal.build_proposal_prompt`);
2. run the structured-fix-proposal pass through the shared
   `ExecutionBudget`;
3. if -- and only if -- a valid `Proposal` came back, apply it to a fresh,
   isolated scratch copy of the repo (never the original checkout);
4. diff the scratch copy against the original to produce the final
   unified-diff text.

Every failure mode in this sequence (no proposal, a scratch-copy
infrastructure failure, a fail-closed apply rejection) resolves to a
`FixPipelineResult` with `ok=False` and a human-readable `reason` -- this
function never raises and never partially completes a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .applier import ApplyError, apply_proposal
from .bounds import ExecutionBudget
from .diffing import generate_unified_diff
from .fencing import FencedContext
from .pathsafe import PathTraversalError
from .proposal import ModelRunner, ProposalPassResult, build_proposal_prompt, run_proposal_pass
from .scratch import ScratchCopyError, scratch_copy

__all__ = ["FixPipelineResult", "run_fix_pipeline"]


@dataclass(frozen=True)
class FixPipelineResult:
    """The outcome of one full read-propose-apply-diff pipeline run."""

    ok: bool
    diff: Optional[str] = None
    reason: Optional[str] = None
    proposal_pass: Optional[ProposalPassResult] = None


def run_fix_pipeline(
    repo_root: "str | Path",
    fenced_context: FencedContext,
    runner: ModelRunner,
    budget: ExecutionBudget,
    feedback: Optional[str] = None,
) -> FixPipelineResult:
    """Run one full, offline, read-propose-apply-diff pipeline pass.

    Args:
        repo_root: The original checkout root. Never mutated: all edits are
            applied to a throwaway scratch copy (`dbt_fixer.scratch.scratch_copy`).
        fenced_context: The already-fenced (never raw) failure/PR context,
            rendered verbatim into the model prompt.
        runner: The `Callable[[str], str]` model runner (a real agno agent
            runner in production, a fake in every test).
        budget: The shared `ExecutionBudget` bounding this pass's wall-clock
            time, tool calls, and turns.
        feedback: Optional prior-round rejection reason (see
            `dbt_fixer.proposal.build_proposal_prompt`), threaded straight
            through into this round's prompt. `None` (the default) produces
            the exact same prompt as a first-round call always has.

    Returns:
        A `FixPipelineResult`. `ok=True` with `diff` set on success;
        `ok=False` with a `reason` for every failure mode (no proposal,
        scratch-copy infrastructure failure, or a fail-closed apply
        rejection) -- this function never raises.
    """

    repo_root = Path(repo_root)
    prompt = build_proposal_prompt(fenced_context, feedback)
    pass_result = run_proposal_pass(runner, prompt, budget)

    if not pass_result.ok:
        return FixPipelineResult(
            ok=False, reason=pass_result.no_proposal_reason, proposal_pass=pass_result
        )

    assert pass_result.proposal is not None  # guaranteed by ProposalPassResult.ok

    try:
        with scratch_copy(repo_root) as scratch_root:
            try:
                applied = apply_proposal(scratch_root, pass_result.proposal)
            except (ApplyError, PathTraversalError) as exc:
                return FixPipelineResult(
                    ok=False, reason=f"proposal could not be applied: {exc}", proposal_pass=pass_result
                )
            diff_text = generate_unified_diff(repo_root, scratch_root, applied.changed_paths)
    except ScratchCopyError as exc:
        return FixPipelineResult(
            ok=False, reason=f"scratch copy could not be created: {exc}", proposal_pass=pass_result
        )

    return FixPipelineResult(ok=True, diff=diff_text, proposal_pass=pass_result)
