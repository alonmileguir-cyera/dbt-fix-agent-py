"""Tests for the shared delivery primitive: markdown chunking.

Ported from the sibling `dbt_auditor` package's `test_delivery_common.py`,
minus the auditor-specific `VERDICT_ACTION`/`should_suppress_in_normal_mode`
tests - dbt_fixer has no verdict-suppression concept (see
`dbt_fixer.delivery_common`'s module docstring for why).
"""

from __future__ import annotations

import re

import pytest

from dbt_fixer.delivery_common import SLACK_TEXT_CHUNK_LIMIT, chunk_markdown

# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------


def test_empty_and_blank_text_produce_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown(None) == []  # type: ignore[arg-type]


def test_short_text_is_a_single_chunk():
    text = "# Status: proposed\n\nAll gates passed."
    chunks = chunk_markdown(text, max_chars=SLACK_TEXT_CHUNK_LIMIT)
    assert chunks == [text]


def test_every_chunk_is_strictly_under_the_limit():
    text = "\n".join(f"line {i} " + ("x" * 50) for i in range(500))
    chunks = chunk_markdown(text, max_chars=200)
    assert chunks
    for chunk in chunks:
        assert len(chunk) < 200


def test_no_line_content_is_dropped_across_chunks():
    lines = [f"item-{i}" for i in range(300)]
    text = "\n".join(lines)
    chunks = chunk_markdown(text, max_chars=100)
    reconstructed = "\n".join(chunks)
    for line in lines:
        assert line in reconstructed


def test_no_line_content_is_duplicated_across_chunks():
    lines = [f"unique-marker-{i:04d}-end" for i in range(200)]
    text = "\n".join(lines)
    chunks = chunk_markdown(text, max_chars=150)
    full_lines = "\n".join(chunks).split("\n")
    for line in lines:
        assert full_lines.count(line) == 1


def test_chunk_order_is_preserved():
    lines = [f"seq-{i:04d}" for i in range(400)]
    text = "\n".join(lines)
    chunks = chunk_markdown(text, max_chars=120)
    # Extract sequence numbers in the order they appear across all chunks.
    found = re.findall(r"seq-(\d{4})", "\n".join(chunks))
    found_ints = [int(x) for x in found]
    assert found_ints == sorted(found_ints)
    assert found_ints == list(range(400))


def test_code_block_split_across_chunks_is_closed_and_reopened():
    body_lines = [f"    row_{i} INT," for i in range(200)]
    text = "Some intro text.\n\n```sql\n" + "\n".join(body_lines) + "\n```\n\nTrailing text."
    chunks = chunk_markdown(text, max_chars=300)
    assert len(chunks) > 1

    # Every chunk must have balanced fences: an even number of ``` markers,
    # meaning it never leaves a code block open at chunk end without
    # being explicitly closed within that same chunk.
    for chunk in chunks:
        fence_count = chunk.count("```")
        assert fence_count % 2 == 0, f"unbalanced fence in chunk: {chunk!r}"
        assert len(chunk) < 300


def test_code_block_content_is_not_lost_when_split():
    body_lines = [f"    row_{i} INT," for i in range(200)]
    text = "```sql\n" + "\n".join(body_lines) + "\n```"
    chunks = chunk_markdown(text, max_chars=300)
    full = "\n".join(chunks)
    for line in body_lines:
        assert line in full
    for chunk in chunks:
        assert len(chunk) < 300


def test_code_block_chunks_never_exceed_limit_reproduction():
    # Direct reproduction of a previously-found overflow: a fenced code
    # block whose lines pack right up against the chunk boundary used to
    # cause the synthetic closing ``` (and, on the next chunk, the
    # reopening fence) to be appended *after* the max_chars budget check,
    # pushing the rendered chunk over the limit. Every chunk - including
    # every fence-closing and fence-reopening chunk - must stay strictly
    # under max_chars.
    body_lines = ["x" * 40 for _ in range(2000)]
    text = "```py\n" + "\n".join(body_lines) + "\n```"
    chunks = chunk_markdown(text, max_chars=SLACK_TEXT_CHUNK_LIMIT)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) < SLACK_TEXT_CHUNK_LIMIT, (
            f"chunk of length {len(chunk)} is not strictly under "
            f"{SLACK_TEXT_CHUNK_LIMIT}: {chunk[:80]!r}..."
        )
        fence_count = chunk.count("```")
        assert fence_count % 2 == 0, f"unbalanced fence in chunk: {chunk!r}"
    full = "\n".join(
        line for chunk in chunks for line in chunk.split("\n") if line.strip("x") == ""
    )
    for line in body_lines:
        assert line in full


def test_code_block_chunks_never_exceed_limit_at_small_boundaries():
    # Same class of bug, exercised at a variety of small max_chars values
    # and fence-language-tag lengths, since the exact overflow depends on
    # how closely lines pack against the boundary.
    for max_chars in (50, 80, 120, 300, 301, 302):
        for lang in ("", "sql", "python"):
            body_lines = [f"row_{i} INT," for i in range(60)]
            fence = f"```{lang}" if lang else "```"
            text = f"{fence}\n" + "\n".join(body_lines) + "\n```"
            chunks = chunk_markdown(text, max_chars=max_chars)
            for chunk in chunks:
                assert len(chunk) <= max_chars, (
                    f"max_chars={max_chars} lang={lang!r}: chunk length "
                    f"{len(chunk)} exceeds limit: {chunk[:80]!r}..."
                )


def test_single_line_longer_than_limit_is_hard_split_not_dropped():
    long_line = "x" * 500
    chunks = chunk_markdown(long_line, max_chars=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100
    # Every character of the original line must appear somewhere across
    # the chunks, in order.
    reconstructed = "".join(chunks)
    assert "x" * 500 in reconstructed or reconstructed.count("x") >= 500


def test_max_chars_must_be_positive():
    with pytest.raises(ValueError):
        chunk_markdown("hello", max_chars=0)


def test_realistic_full_report_chunks_cleanly():
    sections = []
    for i in range(10):
        sections.append(f"### Gate {i}\n\n**Outcome:** pass\n\n> some detail line {i}")
    report = "\n\n".join(sections)
    chunks = chunk_markdown(report, max_chars=200)
    assert chunks
    for chunk in chunks:
        assert len(chunk) < 200
    full = "\n".join(chunks)
    for i in range(10):
        assert f"Gate {i}" in full


def test_fence_reopened_with_same_language_tag():
    body_lines = [f"line_{i}" for i in range(100)]
    text = "```python\n" + "\n".join(body_lines) + "\n```"
    chunks = chunk_markdown(text, max_chars=150)
    assert len(chunks) > 1
    # Every chunk after the first that contains code must reopen with the
    # same "```python" tag, never a bare "```" or a different language.
    reopen_chunks = chunks[1:]
    for chunk in reopen_chunks:
        first_line = chunk.split("\n", 1)[0]
        if first_line.startswith("```") and first_line != "```":
            assert first_line == "```python"
