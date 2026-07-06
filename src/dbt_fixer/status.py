"""Fixed status vocabulary and design-language constants shared by every surface
(stdout, Slack, the rendered report).

The status vocabulary is intentionally small and closed: exactly three outcomes,
each with one consistent glyph, mirroring the sibling `dbt_auditor` package's
`VERDICT_GLYPH` convention so an engineer who already reads auditor output
recognizes the Fix Agent's output instantly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["proposed", "no_safe_fix", "failed"]

STATUSES: tuple[Status, ...] = ("proposed", "no_safe_fix", "failed")

STATUS_GLYPH: dict[Status, str] = {
    "proposed": "✅",  # checkmark
    "no_safe_fix": "\U0001F6AB",  # no-entry
    "failed": "⚠️",  # warning
}

# Fixed gate checklist order -- a reader must be able to tell which gate killed a
# candidate in under two seconds, always in this order.
GATE_ORDER: tuple[str, ...] = ("allowlist", "re-audit", "fix-refuter", "dbt parse")

GATE_GLYPH_PASS = "✓"
GATE_GLYPH_FAIL = "✗"
GATE_GLYPH_SKIPPED = "SKIPPED"

# stdout machine-surface anchors (Sprint 6 formalizes full usage; the status line
# contract is established now so later sprints only ever add to it, never change
# its shape).
STDOUT_STATUS_PREFIX = "dbt-fixer-status"
STDOUT_REASON_PREFIX = "dbt-fixer-reason"
STDOUT_PATCH_BEGIN = "dbt-fixer-patch-begin"
STDOUT_PATCH_END = "dbt-fixer-patch-end"


def is_valid_status(value: str) -> bool:
    return value in STATUSES


@dataclass
class GateResult:
    """A single gate's outcome, always carrying a human-readable reason -- a
    skipped gate is *labeled* SKIPPED with a reason, never simply omitted."""

    name: str
    outcome: Literal["pass", "fail", "skipped"]
    detail: str = ""

    def glyph(self) -> str:
        if self.outcome == "pass":
            return GATE_GLYPH_PASS
        if self.outcome == "fail":
            return GATE_GLYPH_FAIL
        return GATE_GLYPH_SKIPPED

    def render(self) -> str:
        line = f"{self.glyph()} {self.name}"
        if self.detail:
            line += f" -- {self.detail}"
        return line


@dataclass
class RunResult:
    """The single, authoritative result of one dbt_fixer invocation."""

    status: Status
    reason: str = ""
    gates: list[GateResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not is_valid_status(self.status):
            raise ValueError(f"invalid status: {self.status!r}; must be one of {STATUSES}")

    def glyph(self) -> str:
        return STATUS_GLYPH[self.status]
