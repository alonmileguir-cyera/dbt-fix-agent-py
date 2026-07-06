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
from dbt_fixer.reaudit import ProcessOutcome
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
    result = run_bounded_fix_attempt(
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
    assert gate_names_outcomes == {"allowlist": "pass", "re-audit": "pass"}


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
    result = run_bounded_fix_attempt(
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
    assert gate_names_outcomes == {"allowlist": "fail", "re-audit": "skipped"}


def test_retry_loop_is_bounded_by_max_rounds_when_reaudit_always_blocks(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_BLOCKED_STDOUT))

    result = run_bounded_fix_attempt(
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
    assert gate_names_outcomes == {"allowlist": "pass", "re-audit": "fail"}
    assert result.diff is None


# --- feedback threading --------------------------------------------------------


def test_rejected_round_feeds_its_specific_violation_back_as_feedback(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("dbt_project.yml", f"name: test\nversion: {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    run_bounded_fix_attempt(
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

    result = run_bounded_fix_attempt(
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
    assert gate_names_outcomes == {"allowlist": "pass", "re-audit": "pass"}


# --- distinct failed vs. no_safe_fix semantics --------------------------------


def test_missing_auditor_interpreter_is_no_safe_fix_and_stops_immediately(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    result = run_bounded_fix_attempt(
        config=_config(repo, max_rounds=5, auditor_python=None),
        target=_target(),
        fenced_context=_fenced_context(),
        repo_root=repo,
        model_runner=model_runner,
        subprocess_runner=subprocess_runner,
        budget=_budget(),
    )

    assert result.run_result.status == "no_safe_fix"
    # A missing interpreter can't be fixed by retrying -- the loop stops
    # after the very first round instead of burning all 5.
    assert result.rounds_used == 1
    assert len(subprocess_runner.calls) == 0


def test_unexpected_exception_in_a_gate_resolves_to_failed_not_no_safe_fix(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    model_runner = _RecordingModelRunner(
        lambda n: _whole_file_proposal("models/a.sql", f"select 1\nfrom x\nwhere y = {n}\n")
    )
    subprocess_runner = _RecordingSubprocessRunner(lambda n: ProcessOutcome(returncode=0, stdout=_PASSED_STDOUT))

    def _broken_allowlist_gate(**kwargs) -> AllowlistVerdict:
        raise RuntimeError("simulated bug in the allowlist gate")

    assert _no_leaked_scratch_dirs()
    result = run_bounded_fix_attempt(
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
    result_a = run_bounded_fix_attempt(
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

    result_b = run_bounded_fix_attempt(
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
    result = run_bounded_fix_attempt(**kwargs)
    assert _no_leaked_scratch_dirs()

    expected_status = {
        "proposed": "proposed",
        "no_safe_fix_allowlist": "no_safe_fix",
        "no_safe_fix_reaudit": "no_safe_fix",
        "no_safe_fix_missing_interpreter": "no_safe_fix",
        "failed": "failed",
    }[scenario]
    assert result.run_result.status == expected_status
