"""Shadow-mode Slack delivery: fixed-format summary + threaded, chunked detail.

Delivery shape (spec "Structure over prose" / "Spacing and composition as
information hierarchy"):

1. A short summary message is posted first: the run's status glyph
   (:data:`dbt_fixer.status.STATUS_GLYPH`), a PR link, the failure kind, a
   one-line reason, and a compact gate scoreboard.
2. The full detail -- the candidate diff (fenced, unified-diff syntax), the
   rationale, and per-gate detail, in that fixed order -- is posted as
   threaded replies beneath the summary, chunked via
   :func:`dbt_fixer.delivery_common.chunk_markdown` so each reply stays
   strictly under Slack's message-size limit.

Delivery happens for *every* run, unconditionally: unlike the sibling
`dbt_auditor` package (which suppresses a clean ``PASSED`` verdict in
normal mode), shadow mode is the only mode dbt_fixer has, so there is no
"suppress a clean outcome" rule to apply here.

**Failure isolation is the load-bearing property of this module.** Unlike
the auditor's ``deliver_slack_report`` -- which early-returns the moment the
summary post fails or raises, never even attempting the threaded detail
post -- this module always attempts the detail post independently of
whether the summary post succeeded, and always attempts the summary post
independently of whatever the detail post's eventual outcome will be. A
failure in one is caught, logged, and never prevents an attempt at the
other:

- if the summary post fails, the detail chunks are still posted, as
  top-level messages (there being no summary ``ts`` to thread them under);
- if the summary post succeeds but the detail post fails partway through,
  the summary is left standing as posted and the partial detail delivery
  is reported honestly.

Every failure mode here -- no channel configured, no bot token available, a
Slack client that can't be constructed, or any exception/not-ok response
from the Slack Web API itself -- is caught, logged with the specific
failure mode named, and degrades to a no-op (or a partial delivery, when
only one of the two posts failed). Slack delivery never raises, and it
never influences the already-computed :class:`dbt_fixer.status.RunResult`;
the full pipeline runs and stdout is produced identically regardless of
whether Slack is configured or reachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol

from .delivery_common import SLACK_TEXT_CHUNK_LIMIT, chunk_markdown
from .redaction import redact_secrets
from .secrets import SecretsManagerClient, get_slack_bot_token
from .status import RunResult

logger = logging.getLogger(__name__)

__all__ = [
    "SlackClient",
    "SlackClientFactory",
    "SlackDeliveryResult",
    "default_slack_client_factory",
    "deliver_shadow_report",
]


class SlackClient(Protocol):
    """The minimal Slack Web API surface this module depends on.

    A real implementation (:func:`default_slack_client_factory`) is a thin
    wrapper over ``slack_sdk.WebClient`` -- its methods already match this
    shape exactly, so no adapter layer is needed for the real path. Tests
    inject a fake implementing this same protocol so no test ever makes a
    live network call.
    """

    def chat_postMessage(
        self, *, channel: str, text: str, thread_ts: "str | None" = None
    ) -> Mapping[str, Any]: ...


SlackClientFactory = Callable[[str], "SlackClient"]


def default_slack_client_factory(token: str) -> SlackClient:
    """Build a real ``slack_sdk.WebClient`` bound to ``token``."""

    from slack_sdk import WebClient  # local import: optional cost only paid
    # when Slack delivery is actually attempted with a real token.

    return WebClient(token=token)


@dataclass(frozen=True)
class SlackDeliveryResult:
    """Outcome of one :func:`deliver_shadow_report` call.

    ``skipped`` is ``True`` only when delivery never even attempted to
    reach the Slack Web API at all (no channel configured, no token
    resolvable, or the client itself couldn't be constructed) -- in every
    other case at least one ``chat.postMessage`` call was actually made.
    ``summary_posted`` and ``detail_chunks_posted``/``detail_chunks_total``
    report the two post attempts independently, since either may succeed
    or fail without regard to the other.
    """

    skipped: bool
    summary_posted: bool
    summary_ts: Optional[str]
    detail_chunks_posted: int
    detail_chunks_total: int
    reason: str


def _summarize_diff_files(diff_text: str) -> "list[str]":
    """Plain-English per-file lines for a unified diff: what the patch DOES.

    Pure text scan (no diff library): tolerant of any diff dialect the
    pipeline emits, and a malformed diff simply yields fewer/no lines --
    this is presentation, never validation (the applier owns that).
    """

    files: "list[dict]" = []
    current: "Optional[dict]" = None
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            path = parts[-1][2:] if parts[-1].startswith("b/") else parts[-1]
            current = {"path": path, "created": False, "deleted": False, "plus": 0, "minus": 0}
            files.append(current)
        elif current is not None and line.startswith("--- "):
            if line[4:].strip() == "/dev/null":
                current["created"] = True
        elif current is not None and line.startswith("+++ "):
            if line[4:].strip() == "/dev/null":
                current["deleted"] = True
        elif current is not None and line.startswith("+") and not line.startswith("+++"):
            current["plus"] += 1
        elif current is not None and line.startswith("-") and not line.startswith("---"):
            current["minus"] += 1

    lines = []
    for f in files:
        if f["created"]:
            action = "creates"
            counts = f"+{f['plus']} lines"
        elif f["deleted"]:
            action = "deletes"
            counts = f"-{f['minus']} lines"
        else:
            action = "modifies"
            counts = f"+{f['plus']}/-{f['minus']} lines"
        lines.append(f"{action} `{f['path']}` ({counts})")
    return lines


def _build_summary_text(
    run_result: RunResult, *, failure_kind: str, pr_url: str, candidate_diff: str = ""
) -> str:
    glyph = run_result.glyph()
    pr_line = f"*PR:* {pr_url.strip()}\n" if pr_url and pr_url.strip() else ""
    reason = run_result.reason.strip() if run_result.reason and run_result.reason.strip() else (
        "no additional detail provided"
    )
    scoreboard = " | ".join(f"{gate.glyph()} {gate.name}" for gate in run_result.gates) or (
        "no gates recorded"
    )
    file_lines = _summarize_diff_files(candidate_diff)
    if file_lines:
        shown = file_lines[:5]
        if len(file_lines) > 5:
            shown.append(f"...and {len(file_lines) - 5} more file(s)")
        change_lines = "*Proposed change:* " + "; ".join(shown) + "\n"
    else:
        change_lines = ""
    return (
        f"{glyph} *Status:* `{run_result.status}`\n"
        f"{pr_line}"
        f"*Failure kind:* `{failure_kind}`\n"
        f"{reason}\n"
        f"{change_lines}"
        f"*Gates:* {scoreboard}\n"
        "\U0001f9f5 Full patch and gate detail posted as threaded replies below."
    )


def _build_detail_text(run_result: RunResult, *, candidate_diff: str) -> str:
    diff_body = candidate_diff.strip() if candidate_diff else ""
    if diff_body:
        file_lines = _summarize_diff_files(diff_body)
        breakdown = ("\n".join(f"\u2022 {line}" for line in file_lines) + "\n") if file_lines else ""
        diff_section = (
            "*Proposed patch* -- verified by the gate stack above; a human"
            " must still review and apply it (`git apply` on the PR branch):\n"
            f"{breakdown}"
            f"```diff\n{diff_body}\n```"
        )
    else:
        diff_section = "*Proposed patch*\nNo candidate diff was produced for this run."

    reason = run_result.reason.strip() if run_result.reason and run_result.reason.strip() else (
        "no additional detail provided"
    )
    rationale_section = f"*Rationale*\n{reason}"

    gate_lines = "\n".join(gate.render() for gate in run_result.gates) or "no gates recorded"
    gate_section = f"*Gate detail*\n{gate_lines}"

    # Fixed order, always: diff, then rationale, then gate detail.
    return "\n\n".join([diff_section, rationale_section, gate_section])


def _post_summary(
    client: SlackClient, *, channel: str, text: str
) -> "tuple[str | None, bool, str]":
    """Best-effort summary post. Never raises; returns ``(ts, posted, reason)``."""

    try:
        response = client.chat_postMessage(channel=channel, text=text)
    except Exception as exc:  # noqa: BLE001 - Slack API failure must never crash the run.
        logger.error(
            "slack_delivery: chat.postMessage (summary) raised (%s: %s) - "
            "still attempting the threaded detail post independently",
            type(exc).__name__,
            exc,
        )
        return None, False, f"summary post raised: {type(exc).__name__}: {exc}"

    if not response or not response.get("ok", False):
        error_detail = response.get("error") if response else "empty response"
        logger.error(
            "slack_delivery: chat.postMessage (summary) returned not-ok "
            "(%s) - still attempting the threaded detail post independently",
            error_detail,
        )
        return None, False, f"summary post returned not-ok: {error_detail}"

    return response.get("ts"), True, "summary delivered"


def _post_detail_chunks(
    client: SlackClient, *, channel: str, chunks: "list[str]", thread_ts: "str | None"
) -> "tuple[int, str]":
    """Best-effort sequential chunk posting. Never raises; returns ``(posted, reason)``.

    When ``thread_ts`` is ``None`` (the summary post failed or never
    happened), chunks are posted as standalone top-level messages rather
    than thread replies -- there being no summary message to thread under
    -- so the detail is still delivered instead of silently dropped.
    """

    if not chunks:
        return 0, "no detail content to post"

    posted = 0
    for index, chunk in enumerate(chunks):
        try:
            response = client.chat_postMessage(channel=channel, text=chunk, thread_ts=thread_ts)
        except Exception as exc:  # noqa: BLE001 - Slack API failure must never crash the run.
            logger.error(
                "slack_delivery: chat.postMessage (detail chunk %d/%d) "
                "raised (%s: %s) - remaining detail chunks skipped, %d of "
                "%d already delivered",
                index + 1,
                len(chunks),
                type(exc).__name__,
                exc,
                posted,
                len(chunks),
            )
            return posted, f"detail chunk {index + 1}/{len(chunks)} raised: {type(exc).__name__}: {exc}"

        if not response or not response.get("ok", False):
            error_detail = response.get("error") if response else "empty response"
            logger.error(
                "slack_delivery: chat.postMessage (detail chunk %d/%d) "
                "returned not-ok (%s) - remaining detail chunks skipped",
                index + 1,
                len(chunks),
                error_detail,
            )
            return posted, f"detail chunk {index + 1}/{len(chunks)} returned not-ok: {error_detail}"

        posted += 1

    return posted, "all detail chunks delivered"


def _combine_reason(
    *, summary_posted: bool, summary_reason: str, posted: int, total: int, detail_reason: str
) -> str:
    if summary_posted and posted == total:
        return "summary and full detail delivered"
    return f"summary: {summary_reason}; detail: {detail_reason} ({posted}/{total} chunks)"


def deliver_shadow_report(
    *,
    run_result: RunResult,
    failure_kind: str,
    pr_url: str = "",
    candidate_diff: str = "",
    channel: Optional[str],
    token: Optional[str] = None,
    client_factory: SlackClientFactory = default_slack_client_factory,
    token_env: "dict[str, str] | None" = None,
    secrets_client_factory: "Callable[[], SecretsManagerClient] | None" = None,
) -> SlackDeliveryResult:
    """Deliver ``run_result`` to ``channel`` on Slack: summary + threaded detail.

    Args:
        run_result: The already-computed, terminal :class:`RunResult` for
            this invocation. Never mutated or reinterpreted here.
        failure_kind: ``"ci"`` or ``"audit"``, shown in the summary line.
        pr_url: Optional PR link shown in the summary.
        candidate_diff: The winning candidate diff (empty for `no_safe_fix`
            / `failed` runs), rendered fenced as unified diff in the thread.
        channel: Destination Slack channel (id or ``#name``). ``None`` or
            blank means Slack delivery is skipped entirely -- a documented
            no-op, not an error -- and the caller's pipeline/stdout output
            is unaffected either way.
        token: An already-resolved Slack bot token. If omitted, resolved
            via :func:`dbt_fixer.secrets.get_slack_bot_token`.
        client_factory: Builds the Slack client from a resolved token.
            Defaults to a real ``slack_sdk.WebClient``; tests inject a fake
            implementing :class:`SlackClient`.
        token_env: Environment mapping forwarded to
            :func:`dbt_fixer.secrets.get_slack_bot_token` when ``token`` is
            not supplied directly. Defaults to the real process
            environment.
        secrets_client_factory: AWS Secrets Manager client factory
            forwarded to :func:`dbt_fixer.secrets.get_slack_bot_token` when
            ``token`` is not supplied directly. Tests that want to
            exercise the "no token available" path deterministically must
            inject a fake factory here.

    Never raises. Every failure path returns a :class:`SlackDeliveryResult`
    describing exactly what was (and wasn't) delivered, having already
    logged the specific failure mode. The summary post and the threaded
    detail post are each attempted and caught independently: a failure in
    one never prevents an attempt at the other, in either ordering.
    """

    if not channel or not channel.strip():
        logger.warning(
            "slack_delivery: no Slack channel configured - skipping Slack "
            "delivery (no-op); computation is unaffected"
        )
        return SlackDeliveryResult(
            skipped=True,
            summary_posted=False,
            summary_ts=None,
            detail_chunks_posted=0,
            detail_chunks_total=0,
            reason="no Slack channel configured",
        )

    resolved_token = (
        token
        if token is not None
        else get_slack_bot_token(env=token_env, client_factory=secrets_client_factory)
    )
    if not resolved_token:
        logger.warning(
            "slack_delivery: no Slack bot token available - skipping Slack "
            "delivery (no-op); run result is unaffected"
        )
        return SlackDeliveryResult(
            skipped=True,
            summary_posted=False,
            summary_ts=None,
            detail_chunks_posted=0,
            detail_chunks_total=0,
            reason="no Slack bot token available",
        )

    try:
        client = client_factory(resolved_token)
    except Exception as exc:  # noqa: BLE001 - fail-closed-to-no-op, by design.
        logger.error(
            "slack_delivery: failed to construct Slack client (%s: %s) - "
            "skipping delivery",
            type(exc).__name__,
            exc,
        )
        return SlackDeliveryResult(
            skipped=True,
            summary_posted=False,
            summary_ts=None,
            detail_chunks_posted=0,
            detail_chunks_total=0,
            reason=f"Slack client construction failed: {type(exc).__name__}: {exc}",
        )

    summary_text = redact_secrets(
        _build_summary_text(
            run_result, failure_kind=failure_kind, pr_url=pr_url, candidate_diff=candidate_diff
        )
    )
    summary_ts, summary_posted, summary_reason = _post_summary(
        client, channel=channel, text=summary_text
    )

    detail_text = redact_secrets(_build_detail_text(run_result, candidate_diff=candidate_diff))
    chunks = chunk_markdown(detail_text, max_chars=SLACK_TEXT_CHUNK_LIMIT)
    posted, detail_reason = _post_detail_chunks(
        client, channel=channel, chunks=chunks, thread_ts=summary_ts
    )

    return SlackDeliveryResult(
        skipped=False,
        summary_posted=summary_posted,
        summary_ts=summary_ts,
        detail_chunks_posted=posted,
        detail_chunks_total=len(chunks),
        reason=_combine_reason(
            summary_posted=summary_posted,
            summary_reason=summary_reason,
            posted=posted,
            total=len(chunks),
            detail_reason=detail_reason,
        ),
    )
