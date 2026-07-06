"""Untrusted-content fencing and lookalike-marker neutralization.

Every piece of content that originates from a PR author or a CI/audit tool
output -- the CI failure log or audit report, the PR diff, title,
description, and URL -- is untrusted input. None of it is ever allowed to
look, to a model reading the assembled prompt, indistinguishable from this
package's own instructions. This module is the single choke point that
content passes through before it is allowed anywhere near a model-bound
payload.

**Fence grammar.** Each field is wrapped as:

    <<<UNTRUSTED:{field}:{nonce}>>>
    {neutralized content}
    <<<END_UNTRUSTED:{field}:{nonce}>>>

`nonce` is a fresh, cryptographically random token (`secrets.token_hex`)
generated per fenced payload, so an attacker cannot pre-compute a valid
close marker for a field they don't control the nonce of.

**Lookalike-marker neutralization.** Defense-in-depth on top of the nonce:
before content is placed inside a fence, every run of three or more
consecutive `<` or `>` characters -- the only characters that can ever form
one of our marker strings, regardless of field name or nonce -- has
zero-width spaces spliced into it. This breaks *any* string that would
otherwise read as `<<<...>>>`, including a lookalike marker that guesses (or
is even handed) the correct field name and nonce. The content is still
fully present and human-readable; only its ability to form a matching fence
boundary is destroyed.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

_ZERO_WIDTH = "​"

_OPEN_TEMPLATE = "<<<UNTRUSTED:{field}:{nonce}>>>"
_CLOSE_TEMPLATE = "<<<END_UNTRUSTED:{field}:{nonce}>>>"

# Matches any run of 3+ '<' or 3+ '>' -- the only substrings that could ever
# form part of a real or forged fence marker.
_MARKERLIKE_RUN_RE = re.compile(r"<{3,}|>{3,}")

# The fixed, canonical field order every fenced context renders in, matching
# the order a reviewer expects (see `dbt_fixer.env.FixerConfig` field order).
FIELD_ORDER: Tuple[str, ...] = (
    "pr_url",
    "pr_title",
    "pr_description",
    "pr_diff",
    "failure_context",
)


def generate_nonce() -> str:
    """Return a fresh, unpredictable per-payload nonce."""

    return secrets.token_hex(8)


def neutralize_lookalikes(text: Optional[str]) -> str:
    """Break any run of 3+ `<` or `>` characters in `text`.

    This guarantees no substring of `text` can ever equal (or be mistaken
    for) one of this module's fence markers, regardless of whether the
    attacker happens to guess the field name and/or nonce correctly.
    """

    if not text:
        return ""

    def _break(match: "re.Match[str]") -> str:
        return _ZERO_WIDTH.join(match.group(0))

    return _MARKERLIKE_RUN_RE.sub(_break, text)


@dataclass(frozen=True)
class FenceBlock:
    """One rendered `<<<UNTRUSTED:...>>> ... <<<END_UNTRUSTED:...>>>` block."""

    field: str
    nonce: str
    rendered: str


def fence_field(field: str, content: Optional[str], nonce: str) -> FenceBlock:
    """Wrap `content` for `field` in a nonce-bound, lookalike-neutralized fence."""

    safe_content = neutralize_lookalikes(content)
    open_marker = _OPEN_TEMPLATE.format(field=field, nonce=nonce)
    close_marker = _CLOSE_TEMPLATE.format(field=field, nonce=nonce)
    rendered = f"{open_marker}\n{safe_content}\n{close_marker}"
    return FenceBlock(field=field, nonce=nonce, rendered=rendered)


@dataclass(frozen=True)
class FencedContext:
    """A full set of fenced fields sharing one nonce, ready to render."""

    nonce: str
    blocks: Dict[str, FenceBlock]

    def render(self, order: Tuple[str, ...] = FIELD_ORDER) -> str:
        """Render every block present in `order`, in that fixed order.

        Fields not present in this context (not passed to `fence_context`)
        are silently skipped -- a run only fences the fields it actually
        has content for.
        """

        return "\n\n".join(
            self.blocks[name].rendered for name in order if name in self.blocks
        )


def fence_context(fields: Mapping[str, str], *, nonce: Optional[str] = None) -> FencedContext:
    """Fence every field in `fields` under one shared nonce.

    If `nonce` is not supplied, a fresh one is generated. Passing an
    explicit `nonce` is supported only for testing the neutralization
    behavior against a known/attacker-guessed nonce; production call sites
    should always let this generate its own.
    """

    resolved_nonce = nonce or generate_nonce()
    blocks = {
        name: fence_field(name, value, resolved_nonce) for name, value in fields.items()
    }
    return FencedContext(nonce=resolved_nonce, blocks=blocks)
