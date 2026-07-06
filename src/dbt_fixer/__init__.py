"""dbt_fixer: shadow-mode dbt Fix Agent.

Sealed, single-purpose package that proposes narrowly-scoped, mechanically-gated
repairs for a known, red dbt Cloud CI failure or an auditor `BLOCKED` verdict.

This package never writes to a git checkout it did not create itself (a scratch
copy), never pushes, and is architecturally incapable of touching GitHub. Every
proposal is delivered as a Slack message in shadow mode only -- a human always
clicks merge.
"""

__version__ = "0.1.0"
