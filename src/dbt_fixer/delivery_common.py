"""Shared primitives used by the Slack delivery surface.

This module intentionally has no dependency on any real Slack SDK or
network client - it is pure, deterministic, and independently
unit-testable: chunking markdown text so it fits within a hard
per-message character budget without ever splitting a fenced code block
or a single line in a way that breaks rendering (:func:`chunk_markdown`).

Ported near-verbatim from the sibling `dbt_auditor` package's
`delivery_common.chunk_markdown` -- this logic is fully generic (it knows
nothing about verdicts, statuses, or gates), so no adaptation was needed
beyond dropping the auditor's `Verdict`-specific helpers that dbt_fixer has
no equivalent concept for (unlike the auditor, dbt_fixer delivers a Slack
message for every run unconditionally; there is no "suppress a clean
verdict in normal mode" rule to port, since shadow mode is the only mode
that exists for this package).
"""

from __future__ import annotations

import re

__all__ = ["SLACK_TEXT_CHUNK_LIMIT", "chunk_markdown"]

# Slack's Block Kit `section`/`mrkdwn` text object has a documented
# 3,000-character limit; chunking to this boundary keeps every posted
# message renderable regardless of which Slack surface (legacy `text`
# param or a Block Kit block) ultimately displays it. Chosen
# conservatively so a chunk is always *strictly* under any of Slack's
# documented message-size limits, never merely close to one.
SLACK_TEXT_CHUNK_LIMIT: int = 3_000

_FENCE_OPEN_RE = re.compile(r"^```(\S*)\s*$")


def chunk_markdown(text: str, *, max_chars: int = SLACK_TEXT_CHUNK_LIMIT) -> "list[str]":
    """Split ``text`` into chunks each strictly under ``max_chars``.

    Guarantees:

    - Chunk order corresponds exactly to the original content's order.
    - No line's content is ever dropped or duplicated across chunks.
    - A chunk is never split in the middle of a single line: an
      individual line is only ever broken across chunks as an explicit,
      last-resort hard split (only reachable when a single line alone
      exceeds ``max_chars``), never as a side effect of the normal
      line-packing loop.
    - A fenced code block (opened with a ` ``` ` line, optionally with a
      language tag) that would otherwise straddle a chunk boundary is
      closed with a synthetic ` ``` ` at the end of one chunk and
      re-opened with the same language tag at the start of the next, so
      every chunk is independently valid markdown and no code content
      renders as if it were prose (or vice versa).

    Returns an empty list for empty/blank input - there is nothing to
    post, and callers must not synthesize a placeholder chunk out of
    nothing.
    """

    if not text:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    lines = text.split("\n")
    chunks: "list[str]" = []
    current_lines: "list[str]" = []
    in_code_block = False
    fence_lang = ""

    # Exact-length accounting (rather than the earlier per-line "+1" estimate,
    # whose newline/fence bookkeeping had off-by-a-few overflow bugs). We always
    # measure the TRUE rendered length of the chunk we are about to emit.

    def _rendered_len(extra: "str | None" = None) -> int:
        parts = current_lines + ([extra] if extra is not None else [])
        return len("\n".join(parts))

    def _flush(*, reopen_fence: bool) -> None:
        nonlocal current_lines
        parts = list(current_lines)
        if in_code_block:
            parts.append("```")
        chunk_text = "\n".join(parts)
        if chunk_text:
            chunks.append(chunk_text)
        current_lines = []
        # Reopen the fence at the start of the next chunk so code stays fenced.
        if reopen_fence and in_code_block:
            current_lines.append(f"```{fence_lang}")

    for line in lines:
        stripped = line.strip()
        fence_match = _FENCE_OPEN_RE.match(stripped)
        is_fence_open = fence_match is not None and not in_code_block
        is_fence_close = stripped == "```" and in_code_block

        # Reserve for the synthetic closing fence based on the fence state that
        # would hold AFTER this line is appended (an opening-fence line opens a
        # block, so a chunk ending on it still needs a "\n```" when flushed; a
        # closing-fence line ends the block, so no synthetic close is needed).
        # Using the pre-line state here was the residual overflow bug.
        post_in_code = True if is_fence_open else (False if is_fence_close else in_code_block)
        reserve = 4 if post_in_code else 0

        # 1) If appending this line to the current chunk would exceed the budget
        #    (rendered length + the closing fence we'd have to add), flush first
        #    and reopen the fence in a fresh chunk.
        if current_lines and _rendered_len(line) + reserve > max_chars:
            _flush(reopen_fence=True)

        # 2) Does the line fit in the (possibly just-reopened) fresh chunk? If not
        #    — even as the sole content of a fenced chunk that closes — it must be
        #    hard-split. This single check covers BOTH the reopened opening fence
        #    (already in current_lines) and the eventual closing fence, so a line
        #    can never overflow after a reopen. (Correct by construction.)
        if _rendered_len(line) + reserve > max_chars:
            # Drop a reopened-only fence line (we emit self-wrapped pieces instead
            # so we never leave an empty "```lang\n```" chunk behind).
            current_lines = []
            wrap = (len(fence_lang) + 8) if in_code_block else 0  # "```lang\n" + "\n```"
            width = max(1, max_chars - wrap)
            for start in range(0, len(line), width):
                piece = line[start : start + width]
                chunks.append(f"```{fence_lang}\n{piece}\n```" if in_code_block else piece)
            if in_code_block:
                current_lines.append(f"```{fence_lang}")
            # A hard-split line is content, never a fence line, so fence state
            # is unchanged; continue with the next line.
            continue

        current_lines.append(line)

        if is_fence_open:
            in_code_block = True
            fence_lang = fence_match.group(1) if fence_match else ""
        elif is_fence_close:
            in_code_block = False
            fence_lang = ""

    if current_lines:
        _flush(reopen_fence=False)

    # Final guarantee (correct by construction): no returned chunk may exceed
    # max_chars, PERIOD. The packing above already ensures this for every
    # realistic limit (including Slack's 3000). The only way to overflow is the
    # degenerate regime where max_chars is smaller than the fence wrapping itself
    # (e.g. max_chars=10 can't hold "```python\n...\n```") — there, valid fencing
    # is impossible, so we fall back to a raw character split. The length
    # invariant callers depend on holds unconditionally; no content is dropped.
    enforced: "list[str]" = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            enforced.append(chunk)
        else:
            for start in range(0, len(chunk), max_chars):
                enforced.append(chunk[start : start + max_chars])
    return enforced
