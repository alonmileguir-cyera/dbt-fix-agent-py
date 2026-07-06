"""Shared fail-safe numeric environment parsing.

Every numeric-bound env var in this package (wall-clock timeout, tool-call
cap, turn limit, max retry rounds, ...) uses exactly the same fallback
contract, defined once here so it can never drift between call sites:

- A missing or empty value silently uses the documented default.
- A present-but-malformed value (non-numeric, or numeric but outside the
  documented ``[min_value, max_value]`` range -- which also rejects negative
  and zero values whenever the range's floor is >= 1) falls back to the
  *same* documented default.
- A malformed value is never clamped to the nearest bound. Clamping would
  turn a wildly-wrong operator input into a different, silently-chosen
  number; the only safe recovery from "we don't trust this value" is the
  known-good default, plus a recorded warning so the substitution is never
  silent.
- This function never raises. Numeric bounds are optional-by-design: a
  malformed bound degrades gracefully instead of failing the whole run.
"""

from __future__ import annotations

from typing import Callable, List, Mapping, Optional, TypeVar, Union

Number = Union[int, float]
T = TypeVar("T", int, float)


def parse_bounded_number(
    env: Mapping[str, str],
    name: str,
    *,
    default: T,
    min_value: T,
    max_value: T,
    warnings: List[str],
    caster: Callable[[str], T] = int,  # type: ignore[assignment]
) -> T:
    """Parse ``env[name]`` as a bounded number, falling back to ``default``.

    Appends a human-readable explanation to ``warnings`` whenever the
    fallback is used for a *present* (but malformed) value. A value that is
    simply absent or blank is not considered a warning-worthy event.
    """

    raw: Optional[str] = env.get(name)
    if raw is None or raw.strip() == "":
        return default

    text = raw.strip()
    try:
        value = caster(text)
    except (TypeError, ValueError):
        warnings.append(
            f"{name}={raw!r} is not a valid number; falling back to default {default!r}"
        )
        return default

    if value < min_value or value > max_value:
        warnings.append(
            f"{name}={raw!r} is outside the valid range [{min_value!r}, {max_value!r}]; "
            f"falling back to default {default!r}"
        )
        return default

    return value
