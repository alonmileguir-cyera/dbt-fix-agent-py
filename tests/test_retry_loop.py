"""Tests for `dbt_fixer.retry_loop.run_bounded_fix_attempt`: the bounded
propose/apply/allowlist/re-audit loop and terminal status resolution.

Everything here is offline: the model runner is a fake `Callable[[str], str]`
that records every prompt it was given, and the auditor subprocess is a fake
`SubprocessRunner` -- never a real subprocess (enforced by `conftest.py`'s
always-on network/subprocess guard).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Mapping, Optional

import pytest

from dbt_fixer.allowlist import AllowlistVerdict
from dbt_fixer.bounds import Bounds, ExecutionBudget
from dbt_fixer.env import FixerConfig
from dbt_fixer.fencing import fence_context
from dbt_fixer.intake import FailingCheck, FailureTarget
from dbt_fixer.reaudit import AuditorInvocationError, ProcessOutcome
from dbt_fixer.retry_loop import run_bounded_fix_attempt

# --- fixtures ----------------------------------------------------------------


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "a.sql").write_text("select 1\nfrom x\n", encoding="utf-8")
    (repo / "dbt_project.yml").write_text("name: test\n", encoding="utf-8")
    return repo


def _config(repo_path: Path, **overrides) -> FixerConfig:
    kwargs = dict(
        failure_kind="ci",
        repo_path=repo_path,
        pr_title="Fix broken model",
        pr_description="restores a deleted where-clause",
        pr_diff="",
        pr_url="https://github.com/example/repo/pull/1",
        failure_context="Completed with 1 error\nFailure in test not_null_a (models/a.sql)\n",
        auditor_python="/usr/bin/python3.11",
        max_rounds=3,
        max_changed_files=5,
        max_changed_lines=60,
        reaudit_timeout_seconds=30.0,
    )
    kwargs.update(overrides)
    return FixerConfig(**kwargs)


def _fenced_context():
    return fence_context({"failure_context": "not_null test failing on models/a.sql"})


def _target(**overrides) -> FailureTarget:
    checks = overrides.pop("checks", (FailingCheck(identifier="not_null_a"),))
    return FailureTarget(kind=overrides.pop("kind", "ci"), checks=checks)


def _budget() -> ExecutionBudget:
    return ExecutionBudget(Bounds(max_turns=20))


# --- default fakes for the two Sprint 4 gates --------------------------------
#
# Every existing (pre-Sprint-4) test in this file exercises only the
# allowlist/re-audit gates; it neither needs nor wants a real fix-refuter
# model call or a real `dbt` invocation. `_call` supplies deterministic,
# always-non-blocking fakes for both by default (the fix-refuter gate always
# gives an unambiguous "could not refute" pass, and `which` always reports no
# `dbt` on PATH so the dbt parse gate always resolves to a harmless
# "skipped") so those tests keep exercising exactly what they always have.
# Tests that care about fix-refuter or dbt-parse behavior override these
# explicitly.


def _confident_refuter_pass(prompt: str) -> str:
    return json.dumps({"refuted": False, "could_not_refute": True, "reason": "no flaw found"})


def _no_dbt_on_path(name: str):
    return None


def _passing_dbt_subprocess_runner(argv, cwd, timeout_seconds) -> ProcessOutcome:
    return ProcessOutcome(returncode=0, stdout="", stderr="")


def _call(**kwargs):
    kwargs.setdefault("refuter_runner", _confident_refuter_pass)
    kwargs.setdefault("dbt_subprocess_runner", _passing_dbt_subprocess_runner)
    kwargs.setdefault("which", _no_dbt_on_path)
    return run_bounded_fix_attempt(**kwargs)


def _whole_file_proposal(path: str, content: str) -> str:
    return json.dumps(
        {
            "edits": [{"type": "whole_file_replace", "path": path, "content": content}],
            "rationale": "structured fix",
        }
    )


class _RecordingModelRunner:
    """Records every prompt received; answers via an injected callable."""

    def __init__(self, respond) -> None:
        self.prompts: List[str] = []
        self._respond = respond

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._respond(len(self.prompts))


class _RecordingSubprocessRunner:
    """A fake `SubprocessRunner` that records calls and answers via an injected callable."""

    def __init__(self, respond) -> None:
        self.calls: List[dict] = []
        self._respond = respond

    def __call__(self, args, env: Mapping[str, str], cwd: Path, timeout: float) -> ProcessOutcome:
        self.calls.append({"args": list(args), "env": dict(env), "cwd": cwd, "timeout": timeout})
        return self._respond(len(self.calls))


_PASSED_STDOUT = "dbt-auditor-audit-status: completed\ndbt-auditor verdict: PASSED - looks fixed\n"
_BLOCKED_STDOUT = "dbt-auditor-audit-status: completed\ndbt-auditor verdict: BLOCKED - still broken\n"


def _no_leaked_scratch_dirs() -> bool:
    return not any(Path(tempfile.gettempdir()).glob("dbt-fixer-scratch-*"))


# --- happy path: proposed on the very first round ----------------------------


def test_all_gates_passing_on_round_one_resolves_to_proposed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", "select 1\nfrom x\nwhere y = 1\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    assert _no_leaked_scratch_dirs()
    result = _call(
        config=_config(repo),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
    )
    assert _no_leaked_scratch_dirs()

    assert result.run_result.status == "proposed"
    assert result.rounds_used == 1
    assert len(model_runner.prompts) == 1
    assert result.diff is not None
    assert "where y = 1" in result.diff
    gate_names_outcomes = {g.name: g.outcome for g in result.run_result.gates}
    assert gate_names_outcomes == {
        "allowlist": "pass",
        "re-audit": "pass",
        "fix-refuter": "pass",
        "dbt parse": "skipped",
    }


# --- bounded by max_rounds -----------------------------------------------------


def test_retry_loop_is_bounded_by_max_rounds_when_allowlist_always_rejects(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # Every round proposes an edit to a file outside models/ -- the
    # allowlist gate's file-type/path-prefix rule rejects it every time.
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("dbt_project.yml", f"name: test\nversion: {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    assert _no_leaked_scratch_dirs()
    result = _call(
        config=_config(repo, max_rounds=3),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
    )
    assert _no_leaked_scratch_dirs()

    assert result.run_result.status == "no_safe_fix"
    assert result.rounds_used == 3
    assert len(model_runner.prompts) == 3
    # The re-audit gate is never even reached once the allowlist rejects.
    assert len(subprocess_runner.calls) == 0
    gate_names_outcomes = {g.name: g.outcome for g in result.run_result.gates}
    assert gate_names_outcomes == {
        "allowlist": "fail",
        "re-audit": "skipped",
        "fix-refuter": "skipped",
        "dbt parse": "skipped",
    }


def test_retry_loop_is_bounded_by_max_rounds_when_reaudit_always_blocks(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_BLOCKED_STDOUT))

    result = _call(
        config=_config(repo, max_rounds=2),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
    )

    assert result.run_result.status == "no_safe_fix"
    assert result.rounds_used == 2
    assert len(model_runner.prompts) == 2
    assert len(subprocess_runner.calls) == 2
    gate_names_outcomes = {g.name: g.outcome for g in result.run_result.gates}
    assert gate_names_outcomes == {
        "allowlist": "pass",
        "re-audit": "fail",
        "fix-refuter": "skipped",
        "dbt parse": "skipped",
    }
    assert result.diff is None


# --- feedback threading --------------------------------------------------------


def test_rejected_round_feeds_its_specific_violation_back_as_feedback(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("dbt_project.yml", f"name: test\nversion: {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    _call(
        config=_config(repo, max_rounds=2),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
    )

    assert len(model_runner.prompts) == 2
    first_prompt, second_prompt = model_runner.prompts
    assert "## Previous attempt feedback" not in first_prompt
    assert "## Previous attempt feedback" in second_prompt
    assert "allowlist gate" in second_prompt
    assert "file_type_not_allowed" in second_prompt


# --- proposed requires the SAME round's SAME candidate on every gate ---------


def test_proposed_only_reflects_the_round_that_actually_passed_every_gate(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)

    def _propose(n: int) -> str:
        if n == 1:
            # Round 1: fails the allowlist gate outright.
            return _whole_file_proposal("dbt_project.yml", "name: test\nversion: 2\n")
        if n == 2:
            # Round 2: passes the allowlist gate but the auditor blocks it.
            return _whole_file_proposal("models/a.sql", "select 1\nfrom x\nwhere y = 2\n")
        # Round 3: passes everything.
        return _whole_file_proposal("models/a.sql", "select 1\nfrom x\nwhere y = 3\n")

    def _auditor(n: int) -> ProcessOutcome:
        # Only called for rounds that passed the allowlist gate (round 2, 3).
        if n == 1:
            return ProcessOutcome(returncode=0, stdout=_BLOCKED_STDOUT)
        return ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT)

    model_runner = _RecordingModelRunner(_propose)
    subprocess_runner = _RecordingSubprocessRunner(_auditor)

    result = _call(
        config=_config(repo, max_rounds=3),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
    )

    assert result.run_result.status == "proposed"
    assert result.rounds_used == 3
    assert len(model_runner.prompts) == 3
    assert len(subprocess_runner.calls) == 2  # rounds 2 and 3 only
    assert result.diff is not None
    assert "where y = 3" in result.diff
    assert "where y = 2" not in result.diff
    gate_names_outcomes = {g.name: g.outcome for g in result.run_result.gates}
    assert gate_names_outcomes == {
        "allowlist": "pass",
        "re-audit": "pass",
        "fix-refuter": "pass",
        "dbt parse": "skipped",
    }


# --- fix-refuter gate wiring ---------------------------------------------------


def test_fix_refuter_rejection_is_bounded_by_max_rounds_and_never_reaches_dbt_parse(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))
    dbt_subprocess_runner = _passing_dbt_subprocess_runner
    dbt_calls: List[dict] = []

    def _recording_dbt_subprocess_runner(argv, cwd, timeout_seconds):
        dbt_calls.append({"argv": list(argv), "cwd": cwd, "timeout": timeout_seconds})
        return dbt_subprocess_runner(argv, cwd, timeout_seconds)

    def _always_refutes(prompt: str) -> str:
        return json.dumps(
            {"refuted": True, "could_not_refute": False, "reason": "this diff is a no-op"}
        )

    result = _call(
        config=_config(repo, max_rounds=2),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        refuter_runner=_always_refutes,
        dbt_subprocess_runner=_recording_dbt_subprocess_runner,
        budget=_budget(),
    )

    assert result.run_result.status == "no_safe_fix"
    assert result.rounds_used == 2
    assert result.diff is None
    # The dbt parse gate is never even reached once the fix-refuter rejects.
    assert dbt_calls == []
    gate_names_outcomes = {g.name: g.outcome for g in result.run_result.gates}
    assert gate_names_outcomes == {
        "allowlist": "pass",
        "re-audit": "pass",
        "fix-refuter": "fail",
        "dbt parse": "skipped",
    }
    assert "this diff is a no-op" in result.run_result.reason


def test_fix_refuter_passing_lets_the_round_reach_the_dbt_parse_gate(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", "select 1\nfrom x\nwhere y = 1\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))
    dbt_calls: List[dict] = []

    def _fake_which(name: str):
        return "/usr/local/bin/dbt" if name == "dbt" else None

    def _recording_dbt_subprocess_runner(argv, cwd, timeout_seconds):
        dbt_calls.append({"argv": list(argv), "cwd": cwd, "timeout": timeout_seconds})
        return ProcessOutcome(returncode=0, stdout="", stderr="")

    result = _call(
        config=_config(repo),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        dbt_subprocess_runner=_recording_dbt_subprocess_runner,
        which=_fake_which,
        budget=_budget(),
    )

    assert result.run_result.status == "proposed"
    assert len(dbt_calls) == 1
    gate_names_outcomes = {g.name: g.outcome for g in result.run_result.gates}
    assert gate_names_outcomes == {
        "allowlist": "pass",
        "re-audit": "pass",
        "fix-refuter": "pass",
        "dbt parse": "pass",
    }


def test_refuter_runner_is_a_distinct_isolated_context_from_the_proposal_pass(
    tmp_path: Path,
) -> None:
    """refuter_fresh_context_invocation: the fix-refuter gate must run as a
    genuinely fresh, isolated model pass -- never a continuation of the
    proposal pass's own conversation/context. Each fake here stands in for
    a real agno Agent session with its own private internal state; this
    test proves the two are never the same object and that neither fake's
    prompts ever leak into the other's, across multiple rounds."""

    repo = _make_repo(tmp_path)

    class _FakeProposalContext:
        """Stands in for the proposal pass's own model session/context."""

        def __init__(self) -> None:
            self.marker = "proposal-context-marker"
            self.prompts: List[str] = []

        def __call__(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return _whole_file_proposal("models/a.sql", "select 1\nfrom x\nwhere y = 1\n")

    class _FakeRefuterContext:
        """Stands in for the fix-refuter gate's own, separate model session."""

        def __init__(self) -> None:
            self.marker = "refuter-context-marker"
            self.prompts: List[str] = []

        def __call__(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return _confident_refuter_pass(prompt)

    proposal_context = _FakeProposalContext()
    refuter_context = _FakeRefuterContext()

    # The two contexts must be genuinely distinct objects, never the same
    # one reused for both roles.
    assert proposal_context is not refuter_context

    subprocess_runner = _RecordingSubprocessRunner(
        lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT)
    )

    result = _call(
        config=_config(repo),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=proposal_context,
        subprocess_runner=subprocess_runner,
        refuter_runner=refuter_context,
        budget=_budget(),
    )

    assert result.run_result.status == "proposed"
    assert proposal_context.prompts, "the proposal context was never invoked"
    assert refuter_context.prompts, "the refuter context was never invoked"

    # Isolation, proven both directions: nothing unique to one context's
    # marker or transcript ever appears inside the other's prompts.
    for refuter_prompt in refuter_context.prompts:
        assert proposal_context.marker not in refuter_prompt
        for proposal_prompt in proposal_context.prompts:
            assert proposal_prompt not in refuter_prompt

    for proposal_prompt in proposal_context.prompts:
        assert refuter_context.marker not in proposal_prompt
        for refuter_prompt in refuter_context.prompts:
            assert refuter_prompt not in proposal_prompt


def test_dbt_parse_gate_failure_rejects_the_round_but_stays_non_authoritative_on_skip(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    def _fake_which(name: str):
        return "/usr/local/bin/dbt" if name == "dbt" else None

    # Differential contract: the CANDIDATE parse fails while the BASELINE
    # parses clean (calls alternate candidate, baseline per round), so the
    # failure is genuinely the patch's fault and must reject the round.
    dbt_calls = {"n": 0}

    def _failing_dbt_subprocess_runner(argv, cwd, timeout_seconds):
        dbt_calls["n"] += 1
        if dbt_calls["n"] % 2 == 1:  # candidate scratch
            return ProcessOutcome(returncode=1, stdout="", stderr="Compilation Error: bad ref()")
        return ProcessOutcome(returncode=0, stdout="", stderr="")  # baseline

    result = _call(
        config=_config(repo, max_rounds=2),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        dbt_subprocess_runner=_failing_dbt_subprocess_runner,
        which=_fake_which,
        budget=_budget(),
    )

    assert result.run_result.status == "no_safe_fix"
    assert result.rounds_used == 2
    assert result.diff is None
    gate_names_outcomes = {g.name: g.outcome for g in result.run_result.gates}
    assert gate_names_outcomes == {
        "allowlist": "pass",
        "re-audit": "pass",
        "fix-refuter": "pass",
        "dbt parse": "fail",
    }
    assert "bad ref" in result.run_result.reason


# --- distinct failed vs. no_safe_fix semantics --------------------------------


def test_missing_auditor_interpreter_is_no_safe_fix_and_stops_immediately(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    result = _call(
        config=_config(repo, max_rounds=5, auditor_python=None),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
    )

    assert result.run_result.status == "no_safe_fix"
    # Never a silently-skipped gate and never `proposed`: the reason string
    # explicitly names the missing auditor, not a generic/opaque failure.
    assert result.run_result.status != "proposed"
    assert "auditor" in result.run_result.reason.lower()
    assert "DBT_FIXER_AUDITOR_PYTHON" in result.run_result.reason
    # A missing interpreter can't be fixed by retrying -- the loop stops
    # after the very first round instead of burning all 5.
    assert result.rounds_used == 1
    assert len(subprocess_runner.calls) == 0


def test_uninvokable_auditor_interpreter_is_also_a_no_safe_fix_naming_the_auditor(
    tmp_path: Path,
) -> None:
    """Distinct from an *unconfigured* interpreter: here `auditor_python` is
    set, but invoking it raises (e.g. the path doesn't actually exist on
    this host). This must fail closed exactly the same way -- `no_safe_fix`,
    never `proposed`, never a silently-skipped gate -- with a reason that
    still names the auditor, not a raw stack trace.
    """

    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )

    def _raise_invocation_error(args, env, cwd, timeout):
        raise AuditorInvocationError(
            "could not start the sealed auditor: "
            "[Errno 2] No such file or directory: '/does/not/exist/python3.11'"
        )

    result = _call(
        config=_config(repo, max_rounds=5, auditor_python="/does/not/exist/python3.11"),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=_raise_invocation_error,
        budget=_budget(),
    )

    assert result.run_result.status == "no_safe_fix"
    assert "auditor" in result.run_result.reason.lower()
    assert "Exception:" not in result.run_result.reason
    assert "Traceback" not in result.run_result.reason
    assert result.rounds_used == 1


def test_unexpected_exception_in_a_gate_resolves_to_failed_not_no_safe_fix(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    def _broken_allowlist_gate(**kwargs) -> AllowlistVerdict:
        raise RuntimeError("simulated bug in the allowlist gate")

    assert _no_leaked_scratch_dirs()
    result = _call(
        config=_config(repo, max_rounds=3),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
        allowlist_gate=_broken_allowlist_gate,
    )
    assert _no_leaked_scratch_dirs()

    assert result.run_result.status == "failed"
    assert "simulated bug in the allowlist gate" in result.run_result.reason
    assert result.diff is None


def test_failed_and_no_safe_fix_are_never_the_same_status(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)

    # no_safe_fix: every round is honestly tried and none passes.
    model_runner_a = _RecordingModelRunner(
        lambda n: _whole_file_proposal("dbt_project.yml", f"name: test\nversion: {n}\n")
    )
    subprocess_runner_a = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))
    result_a = _call(
        config=_config(repo, max_rounds=2),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner_a,
        subprocess_runner=subprocess_runner_a,
        budget=_budget(),
    )

    # failed: a genuine bug in the wiring.
    model_runner_b = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", "select 1\nfrom x\nwhere y = 1\n")
    )
    subprocess_runner_b = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    def _broken_reaudit_gate(**kwargs):
        raise ValueError("boom")

    result_b = _call(
        config=_config(repo, max_rounds=2),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner_b,
        subprocess_runner=subprocess_runner_b,
        budget=_budget(),
        reaudit_gate=_broken_reaudit_gate,
    )

    assert result_a.run_result.status == "no_safe_fix"
    assert result_b.run_result.status == "failed"
    assert result_a.run_result.status != result_b.run_result.status


# --- scratch-dir cleanup across every terminal outcome ------------------------


@pytest.mark.parametrize(
    "scenario",
    ["proposed", "no_safe_fix_allowlist", "no_safe_fix_reaudit", "no_safe_fix_missing_interpreter", "failed"],
)
def test_scratch_dirs_never_leak_regardless_of_terminal_outcome(tmp_path: Path, scenario: str) -> None:
    repo = _make_repo(tmp_path)
    allowlist_gate = None
    auditor_python: Optional[str] = "/usr/bin/python3.11"

    if scenario == "proposed":
        model_runner = _RecordingModelRunner(
            lambda n: _whole_file_proposal("models/a.sql", "select 1\nfrom x\nwhere y = 1\n")
        )
        subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))
    elif scenario == "no_safe_fix_allowlist":
        model_runner = _RecordingModelRunner(
            lambda n: _whole_file_proposal("dbt_project.yml", f"name: test\nversion: {n}\n")
        )
        subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))
    elif scenario == "no_safe_fix_reaudit":
        model_runner = _RecordingModelRunner(
            lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
        )
        subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_BLOCKED_STDOUT))
    elif scenario == "no_safe_fix_missing_interpreter":
        auditor_python = None
        model_runner = _RecordingModelRunner(
            lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
        )
        subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))
    else:  # failed
        model_runner = _RecordingModelRunner(
            lambda n: _whole_file_proposal("models/a.sql", "select 1\nfrom x\nwhere y = 1\n")
        )
        subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

        def allowlist_gate(**kwargs):
            raise RuntimeError("simulated failure")

    assert _no_leaked_scratch_dirs()
    kwargs = dict(
        config=_config(repo, max_rounds=2, auditor_python=auditor_python),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
    )
    if allowlist_gate is not None:
        kwargs["allowlist_gate"] = allowlist_gate
    result = _call(**kwargs)
    assert _no_leaked_scratch_dirs()

    expected_status = {
        "proposed": "proposed",
        "no_safe_fix_allowlist": "no_safe_fix",
        "no_safe_fix_reaudit": "no_safe_fix",
        "no_safe_fix_missing_interpreter": "no_safe_fix",
        "failed": "failed",
    }[scenario]
    assert result.run_result.status == expected_status
