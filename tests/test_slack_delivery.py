"""Tests for shadow-mode Slack delivery: fixed-format summary + threaded detail.

Every scenario here is fully offline: `_FakeSlackClient` never touches a
real network socket (blocked globally by `conftest.py` in any case), and
every Secrets Manager path is stubbed via an injectable `client_factory`.
"""

from __future__ import annotations

from typing import Optional

from dbt_fixer.slack_delivery import SlackDeliveryResult, deliver_shadow_report
from dbt_fixer.status import GateResult, RunResult

# Built by concatenation so no secret-shaped literal lands in this blob
# (GitHub push protection flags the joined form).
FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"


class _FakeSlackClient:
    """Records every `chat.postMessage` call; simulates configurable failures.

    Calls are 0-indexed in the order they happen: index 0 is always the
    summary post attempt; every later index is a detail-chunk post
    attempt. `fail_at`/`not_ok_at` let a test target either independently.
    """

    def __init__(
        self,
        *,
        fail_at: "set[int] | None" = None,
        error: "Exception | None" = None,
        not_ok_at: "set[int] | None" = None,
    ):
        self.calls: "list[dict]" = []
        self._ts_counter = 0
        self._fail_at = fail_at or set()
        self._error = error
        self._not_ok_at = not_ok_at or set()

    def chat_postMessage(self, *, channel, text, thread_ts=None):
        index = len(self.calls)
        self.calls.append({"channel": channel, "text": text, "thread_ts": thread_ts})
        if index in self._fail_at:
            raise (self._error or RuntimeError("simulated Slack API failure"))
        if index in self._not_ok_at:
            return {"ok": False, "error": "channel_not_found"}
        self._ts_counter += 1
        return {"ok": True, "ts": f"1234.{self._ts_counter:04d}"}


def _summary_call(client: _FakeSlackClient) -> "dict | None":
    return client.calls[0] if client.calls else None


def _detail_calls(client: _FakeSlackClient) -> "list[dict]":
    return client.calls[1:]


def _run_result(
    *,
    status: str = "proposed",
    reason: str = "restored the deleted CTE alias; re-audit and refuter both cleared it",
    gates: "list[GateResult] | None" = None,
) -> RunResult:
    if gates is None:
        gates = [
            GateResult(name="allowlist", outcome="pass", detail="1 file, 3 lines changed"),
            GateResult(name="re-audit", outcome="pass", detail="auditor verdict: PASSED"),
            GateResult(name="fix-refuter", outcome="pass", detail="could not refute"),
            GateResult(name="dbt parse", outcome="skipped", detail="dbt not on PATH"),
        ]
    return RunResult(status=status, reason=reason, gates=gates)


def _boom_secrets_client():
    raise RuntimeError("no AWS credentials available in test environment")


# ---------------------------------------------------------------------------
# Delivery happens for every status, unconditionally (shadow mode is the
# only mode; there is no clean-outcome suppression concept here).
# ---------------------------------------------------------------------------


def test_delivers_summary_and_detail_for_every_status():
    for status in ("proposed", "no_safe_fix", "failed"):
        client = _FakeSlackClient()
        run_result = _run_result(status=status)
        result = deliver_shadow_report(
            run_result=run_result,
            failure_kind="ci",
            channel="#dbt-fixer",
            token="xoxb-test",
            client_factory=lambda t, c=client: c,
        )
        assert result.skipped is False
        assert result.summary_posted is True
        summary = _summary_call(client)
        assert summary is not None
        assert run_result.glyph() in summary["text"]
        assert status in summary["text"]
        assert len(_detail_calls(client)) >= 1


def test_summary_contains_failure_kind_reason_and_gate_scoreboard():
    client = _FakeSlackClient()
    run_result = _run_result()
    deliver_shadow_report(
        run_result=run_result,
        failure_kind="audit",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    summary_text = _summary_call(client)["text"]
    assert "`audit`" in summary_text
    assert run_result.reason in summary_text
    for gate in run_result.gates:
        assert gate.name in summary_text
        assert gate.glyph() in summary_text


def test_summary_includes_pr_url_line_when_provided():
    client = _FakeSlackClient()
    deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        pr_url="https://github.com/cyeragit/bi-dbt/pull/2497",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    summary_text = _summary_call(client)["text"]
    assert "https://github.com/cyeragit/bi-dbt/pull/2497" in summary_text
    assert "*PR:*" in summary_text


def test_summary_field_order_is_glyph_pr_failure_kind_reason_gates():
    """Spec's fixed shape: status glyph, PR link, failure kind, reason, then
    the gate scoreboard -- in that order, every time, for every status."""

    for status in ("proposed", "no_safe_fix", "failed"):
        client = _FakeSlackClient()
        run_result = _run_result(status=status)
        deliver_shadow_report(
            run_result=run_result,
            failure_kind="ci",
            pr_url="https://github.com/cyeragit/bi-dbt/pull/2497",
            channel="#chan",
            token="xoxb-test",
            client_factory=lambda t, c=client: c,
        )
        summary_text = _summary_call(client)["text"]

        glyph_index = summary_text.index(run_result.glyph())
        pr_index = summary_text.index("*PR:*")
        failure_kind_index = summary_text.index("*Failure kind:*")
        reason_index = summary_text.index(run_result.reason)
        gates_index = summary_text.index("*Gates:*")

        assert glyph_index < pr_index < failure_kind_index < reason_index < gates_index, (
            f"summary field order violated for status={status!r}: {summary_text!r}"
        )


def test_summary_omits_pr_line_when_url_absent():
    client = _FakeSlackClient()
    deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    assert "*PR:*" not in _summary_call(client)["text"]


# ---------------------------------------------------------------------------
# Threaded detail structure: diff, then rationale, then gate detail; diffs
# and paths always fenced.
# ---------------------------------------------------------------------------


def test_detail_thread_posted_under_the_summary_ts():
    client = _FakeSlackClient()
    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        candidate_diff="--- a/models/foo.sql\n+++ b/models/foo.sql\n@@ -1 +1 @@\n-old\n+new\n",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    details = _detail_calls(client)
    assert details
    for call in details:
        assert call["thread_ts"] == result.summary_ts


def test_detail_sections_appear_in_fixed_order_diff_rationale_gate_detail():
    client = _FakeSlackClient()
    deliver_shadow_report(
        run_result=_run_result(reason="a short, specific rationale"),
        failure_kind="ci",
        candidate_diff="--- a/models/foo.sql\n+++ b/models/foo.sql\n@@ -1 +1 @@\n-old\n+new\n",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    full_detail = "\n".join(call["text"] for call in _detail_calls(client))
    diff_pos = full_detail.index("*Proposed patch*")
    rationale_pos = full_detail.index("*Rationale*")
    gate_pos = full_detail.index("*Gate detail*")
    assert diff_pos < rationale_pos < gate_pos


def test_diff_is_rendered_in_a_fenced_unified_diff_block():
    client = _FakeSlackClient()
    diff_text = "--- a/models/foo.sql\n+++ b/models/foo.sql\n@@ -1 +1 @@\n-old\n+new\n"
    deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        candidate_diff=diff_text,
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    full_detail = "\n".join(call["text"] for call in _detail_calls(client))
    assert "```diff" in full_detail
    assert "models/foo.sql" in full_detail
    # The diff body must appear inside the fence, not as unfenced prose:
    fence_start = full_detail.index("```diff")
    fence_end = full_detail.index("```", fence_start + len("```diff"))
    assert "models/foo.sql" in full_detail[fence_start:fence_end]


def test_no_candidate_diff_still_produces_a_labeled_diff_section():
    client = _FakeSlackClient()
    deliver_shadow_report(
        run_result=_run_result(status="no_safe_fix", reason="no safe restore-only patch found"),
        failure_kind="ci",
        candidate_diff="",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    full_detail = "\n".join(call["text"] for call in _detail_calls(client))
    assert "*Proposed patch*" in full_detail
    assert "No candidate diff" in full_detail


# ---------------------------------------------------------------------------
# Chunking respects Slack's platform limits.
# ---------------------------------------------------------------------------


def test_large_detail_content_is_chunked_into_multiple_replies_under_the_limit():
    client = _FakeSlackClient()
    big_diff = "\n".join(f"+row_{i} INT," for i in range(2000))
    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        candidate_diff=big_diff,
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    details = _detail_calls(client)
    assert len(details) > 1
    assert result.detail_chunks_total == len(details)
    for call in details:
        assert len(call["text"]) < 3_000


# ---------------------------------------------------------------------------
# Failure isolation: summary and detail posts are attempted and caught
# fully independently, in both failure orderings.
# ---------------------------------------------------------------------------


def test_summary_post_failure_still_attempts_the_detail_post():
    # Index 0 (the summary) raises; the detail chunk(s) at index >= 1 must
    # still be attempted, posted as standalone messages (no summary_ts to
    # thread under).
    client = _FakeSlackClient(fail_at={0})
    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        candidate_diff="--- a/x.sql\n+++ b/x.sql\n@@ -1 +1 @@\n-a\n+b\n",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    assert result.skipped is False
    assert result.summary_posted is False
    assert result.summary_ts is None
    assert result.detail_chunks_posted == result.detail_chunks_total
    assert result.detail_chunks_posted >= 1
    details = _detail_calls(client)
    assert details
    for call in details:
        assert call["thread_ts"] is None


def test_summary_post_not_ok_response_still_attempts_the_detail_post():
    client = _FakeSlackClient(not_ok_at={0})
    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    assert result.summary_posted is False
    assert result.detail_chunks_posted >= 1


def test_detail_post_failure_after_summary_success_still_reports_summary_delivered():
    # Summary (index 0) succeeds; every detail chunk (index >= 1) fails.
    client = _FakeSlackClient(fail_at={1})
    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        candidate_diff="--- a/x.sql\n+++ b/x.sql\n@@ -1 +1 @@\n-a\n+b\n",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    assert result.skipped is False
    assert result.summary_posted is True
    assert result.summary_ts is not None
    assert result.detail_chunks_posted == 0
    assert result.detail_chunks_total >= 1
    # The one detail attempt still happened (and was caught), it just failed.
    assert len(_detail_calls(client)) == 1


def test_detail_post_partial_failure_partway_through_reports_partial_count():
    client = _FakeSlackClient(fail_at={2})  # summary + first chunk ok, second chunk fails
    big_diff = "\n".join(f"+row_{i} INT," for i in range(2000))
    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        candidate_diff=big_diff,
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    assert result.summary_posted is True
    assert 0 < result.detail_chunks_posted < result.detail_chunks_total


# ---------------------------------------------------------------------------
# Never raises; degrades to a no-op (or partial delivery) on every failure
# mode, and computation is unaffected either way.
# ---------------------------------------------------------------------------


def test_no_channel_configured_skips_without_constructing_a_client():
    def _boom_factory(token):
        raise AssertionError("client must never be constructed with no channel configured")

    for blank_channel in (None, "", "   "):
        result = deliver_shadow_report(
            run_result=_run_result(),
            failure_kind="ci",
            channel=blank_channel,
            token="xoxb-test",
            client_factory=_boom_factory,
        )
        assert isinstance(result, SlackDeliveryResult)
        assert result.skipped is True
        assert result.summary_posted is False
        assert result.detail_chunks_posted == 0
        assert "channel" in result.reason.lower()


def test_no_token_available_skips_without_raising():
    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        channel="#chan",
        token=None,
        token_env={},
        secrets_client_factory=_boom_secrets_client,
        client_factory=lambda t: (_ for _ in ()).throw(
            AssertionError("client must never be constructed without a token")
        ),
    )
    assert result.skipped is True
    assert "token" in result.reason.lower()


def test_client_construction_failure_degrades_to_noop():
    def _boom(token):
        raise RuntimeError("cannot construct client")

    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        channel="#chan",
        token="xoxb-test",
        client_factory=_boom,
    )
    assert result.skipped is True
    assert "construction" in result.reason.lower()


def test_summary_and_detail_exceptions_never_propagate():
    class RateLimitError(Exception):
        pass

    client = _FakeSlackClient(fail_at={0, 1, 2, 3}, error=RateLimitError("rate_limited"))
    result = deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        candidate_diff="--- a/x.sql\n+++ b/x.sql\n@@ -1 +1 @@\n-a\n+b\n",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    assert result.skipped is False
    assert result.summary_posted is False
    assert result.detail_chunks_posted == 0


def test_computation_completes_regardless_of_slack_outcome():
    # The caller (entrypoint) computes run_result *before* calling Slack
    # delivery; this asserts that a totally-failing Slack layer never
    # raises back into that already-completed computation.
    def _boom(token):
        raise RuntimeError("network unreachable")

    run_result = _run_result(status="failed", reason="model runner raised an unexpected error")
    result = deliver_shadow_report(
        run_result=run_result,
        failure_kind="ci",
        channel="#chan",
        token="xoxb-test",
        client_factory=_boom,
    )
    assert result.skipped is True
    # The RunResult itself is completely untouched by the Slack failure.
    assert run_result.status == "failed"


# ---------------------------------------------------------------------------
# Redaction of secret-shaped content before it ever reaches Slack.
# ---------------------------------------------------------------------------


def test_secret_like_content_in_diff_is_redacted_before_posting():
    client = _FakeSlackClient()
    diff_text = f"--- a/config.sql\n+++ b/config.sql\n@@ -1 +1 @@\n-old\n+key={FAKE_AWS_KEY}\n"
    deliver_shadow_report(
        run_result=_run_result(),
        failure_kind="ci",
        candidate_diff=diff_text,
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    for call in _detail_calls(client):
        assert FAKE_AWS_KEY not in call["text"]


def test_secret_like_content_in_reason_is_redacted_in_summary_and_detail():
    client = _FakeSlackClient()
    run_result = _run_result(reason="leaked password: SuperSecretValue123")
    deliver_shadow_report(
        run_result=run_result,
        failure_kind="ci",
        channel="#chan",
        token="xoxb-test",
        client_factory=lambda t, c=client: c,
    )
    assert "SuperSecretValue123" not in _summary_call(client)["text"]
    for call in _detail_calls(client):
        assert "SuperSecretValue123" not in call["text"]


# ---------------------------------------------------------------------------
# plain-English change summary (readability: what the patch DOES)
# ---------------------------------------------------------------------------


def test_summary_states_what_the_patch_does():
    from dbt_fixer.slack_delivery import _build_summary_text
    from dbt_fixer.status import GateResult, RunResult

    diff = (
        "diff --git a/proj/models/staging/_x__models.yml b/proj/models/staging/_x__models.yml\n"
        "--- /dev/null\n"
        "+++ b/proj/models/staging/_x__models.yml\n"
        "@@ -0,0 +1,2 @@\n"
        "+version: 2\n"
        "+models: []\n"
    )
    text = _build_summary_text(
        RunResult(status="proposed", reason="ok", gates=[GateResult("allowlist", "pass")]),
        failure_kind="audit",
        pr_url="https://example.com/pr/1",
        candidate_diff=diff,
    )
    assert "*Proposed change:* creates `proj/models/staging/_x__models.yml` (+2 lines)" in text


def test_diff_file_summary_covers_create_modify_delete():
    from dbt_fixer.slack_delivery import _summarize_diff_files

    diff = (
        "diff --git a/a.yml b/a.yml\n--- /dev/null\n+++ b/a.yml\n@@ -0,0 +1 @@\n+x\n"
        "diff --git a/b.sql b/b.sql\n--- a/b.sql\n+++ b/b.sql\n@@ -1,2 +1,2 @@\n-old\n+new\n"
        "diff --git a/c.md b/c.md\n--- a/c.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n"
    )
    assert _summarize_diff_files(diff) == [
        "creates `a.yml` (+1 lines)",
        "modifies `b.sql` (+1/-1 lines)",
        "deletes `c.md` (-1 lines)",
    ]


def test_detail_thread_has_breakdown_and_apply_hint():
    from dbt_fixer.slack_delivery import _build_detail_text
    from dbt_fixer.status import RunResult

    diff = (
        "diff --git a/a.yml b/a.yml\n--- /dev/null\n+++ b/a.yml\n@@ -0,0 +1 @@\n+x\n"
    )
    text = _build_detail_text(
        RunResult(status="proposed", reason="ok", gates=[]), candidate_diff=diff
    )
    assert "*Proposed patch*" in text
    assert "git apply" in text
    assert "creates `a.yml` (+1 lines)" in text
    assert "```diff" in text


def test_malformed_diff_degrades_to_no_change_line():
    from dbt_fixer.slack_delivery import _build_summary_text, _summarize_diff_files
    from dbt_fixer.status import RunResult

    assert _summarize_diff_files("complete garbage, not a diff") == []
    text = _build_summary_text(
        RunResult(status="proposed", reason="ok", gates=[]),
        failure_kind="audit",
        pr_url="",
        candidate_diff="complete garbage, not a diff",
    )
    assert "*Proposed change:*" not in text  # presentation never fabricates
