"""Tests for `dbt_fixer.intake`: structured target parsing across CI and audit
formats, and the honest `no_safe_fix` path for unparseable input."""

from __future__ import annotations

from dbt_fixer.env import FixerConfig
from dbt_fixer.intake import parse_failure_target, resolve_intake

CI_FIXTURE = """
Completed with 1 error and 0 warnings:

Failure in test not_null_orders_order_id (models/staging/schema.yml)
  Got 3 results, configured to fail if != 0

Done. PASS=10 WARN=0 ERROR=1 SKIP=0 TOTAL=11
""".strip()

CI_MULTI_FIXTURE = """
Completed with 2 errors and 0 warnings:

Failure in test not_null_orders_order_id (models/staging/schema.yml)
  Got 3 results, configured to fail if != 0

Failure in test unique_orders_order_id (models/staging/schema.yml)
  Got 2 results, configured to fail if != 0

Done. PASS=8 WARN=0 ERROR=2 SKIP=0 TOTAL=10
""".strip()

AUDIT_SINGLE_FIXTURE = """
dbt-auditor-status: verdict=BLOCKED

## Audit Report
### Failing Checks
- check: schema_test:not_null_orders_customer_id
  status: FAIL
  evidence: column customer_id contains 3 null values in `orders`, all pre-existing guest checkouts
  suggestion: remove the not_null test; the audit proved the nulls are expected
""".strip()

AUDIT_MULTI_FIXTURE = """
dbt-auditor-status: verdict=BLOCKED

## Audit Report
### Failing Checks
- check: schema_test:not_null_orders_customer_id
  status: FAIL
  evidence: column customer_id contains 3 null values
  suggestion: remove the not_null test
- check: schema_test:unique_orders_order_id
  status: FAIL
  evidence: 2 duplicate order_id values found
  suggestion: investigate upstream dedup logic
""".strip()

AUDIT_PASSED_FIXTURE = "dbt-auditor-status: verdict=PASSED\n\nNo failing checks."


# --- CI fixtures -------------------------------------------------------------


def test_parses_ci_failure_single_check():
    target, reason = parse_failure_target("ci", CI_FIXTURE)
    assert reason is None
    assert target is not None
    assert target.kind == "ci"
    assert target.identifiers == ("not_null_orders_order_id",)
    assert "3 results" in target.checks[0].evidence


def test_parses_ci_failure_multiple_checks():
    target, reason = parse_failure_target("ci", CI_MULTI_FIXTURE)
    assert reason is None
    assert target.identifiers == ("not_null_orders_order_id", "unique_orders_order_id")


# --- audit fixtures -----------------------------------------------------------


def test_parses_audit_report_single_failing_check():
    target, reason = parse_failure_target("audit", AUDIT_SINGLE_FIXTURE)
    assert reason is None
    assert target.kind == "audit"
    assert target.identifiers == ("schema_test:not_null_orders_customer_id",)
    check = target.checks[0]
    assert "null values" in check.evidence
    assert "remove the not_null test" in check.suggestion


def test_parses_audit_report_multiple_failing_checks():
    target, reason = parse_failure_target("audit", AUDIT_MULTI_FIXTURE)
    assert reason is None
    assert len(target.checks) == 2
    assert target.identifiers == (
        "schema_test:not_null_orders_customer_id",
        "schema_test:unique_orders_order_id",
    )
    assert "duplicate" in target.checks[1].evidence


def test_audit_passed_verdict_is_not_a_target():
    target, reason = parse_failure_target("audit", AUDIT_PASSED_FIXTURE)
    assert target is None
    assert reason is not None
    assert "PASSED" in reason
    assert "BLOCKED" in reason


# --- unparseable / no_safe_fix -------------------------------------------------


def test_empty_string_is_unparseable():
    target, reason = parse_failure_target("ci", "")
    assert target is None
    assert reason


def test_truncated_ci_context_is_unparseable():
    target, reason = parse_failure_target("ci", "Failure in te")
    assert target is None
    assert reason


def test_unrecognized_freeform_text_is_unparseable():
    target, reason = parse_failure_target("ci", "this is just some random unrelated log output")
    assert target is None
    assert "recognized" in reason


def test_ci_header_present_but_no_extractable_check_is_unparseable():
    target, reason = parse_failure_target("ci", "Completed with 1 error and 0 warnings")
    assert target is None
    assert "no individual" in reason


def test_audit_shape_present_but_no_extractable_check_is_unparseable():
    target, reason = parse_failure_target("audit", "dbt-auditor-status: verdict=BLOCKED\n\n(no detail)")
    assert target is None
    assert "no failing" in reason


def test_unrecognized_failure_kind_is_unparseable():
    target, reason = parse_failure_target("bogus", CI_FIXTURE)  # type: ignore[arg-type]
    assert target is None
    assert "bogus" in reason


# --- resolve_intake: full config -> IntakeResult, including fencing ----------


def test_resolve_intake_empty_context_is_no_safe_fix(tmp_path):
    config = FixerConfig(failure_kind="ci", repo_path=tmp_path, failure_context="")
    result = resolve_intake(config)
    assert not result.ok
    assert "empty" in result.no_safe_fix_reason
    assert result.fenced_context is not None


def test_resolve_intake_whitespace_only_context_is_no_safe_fix(tmp_path):
    config = FixerConfig(failure_kind="ci", repo_path=tmp_path, failure_context="   \n\t  ")
    result = resolve_intake(config)
    assert not result.ok
    assert "empty" in result.no_safe_fix_reason


def test_resolve_intake_truncated_context_is_no_safe_fix(tmp_path):
    config = FixerConfig(failure_kind="ci", repo_path=tmp_path, failure_context="Failure in te")
    result = resolve_intake(config)
    assert not result.ok
    assert result.no_safe_fix_reason


def test_resolve_intake_success_carries_target_and_fenced_context(tmp_path):
    config = FixerConfig(
        failure_kind="ci",
        repo_path=tmp_path,
        failure_context=CI_FIXTURE,
        pr_title="restore the accidentally deleted not_null test",
    )
    result = resolve_intake(config)
    assert result.ok
    assert result.no_safe_fix_reason is None
    assert result.target.identifiers == ("not_null_orders_order_id",)

    rendered = result.fenced_context.render(("pr_title", "failure_context"))
    assert "restore the accidentally deleted not_null test" in rendered
    assert "Failure in test" in rendered


def test_resolve_intake_audit_success(tmp_path):
    config = FixerConfig(failure_kind="audit", repo_path=tmp_path, failure_context=AUDIT_SINGLE_FIXTURE)
    result = resolve_intake(config)
    assert result.ok
    assert result.target.kind == "audit"
    assert result.target.identifiers == ("schema_test:not_null_orders_customer_id",)


# ---------------------------------------------------------------------------
# the REAL rendered auditor report format (dbt_auditor.report.render_report)
# ---------------------------------------------------------------------------

REAL_BLOCKED_REPORT = """<!-- dbt-auditor-report-id:kind=dbt-adversarial-audit|repo=cyeragit/bi-dbt|pr=2500 -->

# \U0001f6d1 Verdict: **BLOCKED**

## \U0001f6a8 Critical Issues

- **Schema Contract Verification** (`schema_contract_verification`) - **FAILED** (score: 0/100)

> schema.yml declares id with unique+not_null but the SQL unions 4 regions.

## Checks

### Tenant Isolation Integrity (`tenant_isolation_integrity`)

**Severity:** Critical

**Score:** 95/100 &nbsp;&nbsp; **State:** **PASS**

**Evidence:**

> tenantId is carried through unchanged.

### Schema Contract Verification (`schema_contract_verification`)

**Severity:** Critical

**Score:** 0/100 &nbsp;&nbsp; **State:** **FAIL**

**Evidence:**

> _raw_etl_history__models.yml declares id with both unique and not_null tests.
> The model SQL sets id from uid independently in 4 UNION ALL branches.

**Reasoning:** The declared contract overpromises uniqueness.
"""

REAL_ARTIFACT_REPORT = """<!-- dbt-auditor-report-id:kind=dbt-adversarial-audit|repo=cyeragit/bi-dbt|pr=2500 -->

# \U0001f6d1 Verdict: **BLOCKED**

## Checks

### Tenant Isolation Integrity (`tenant_isolation_integrity`)

**Severity:** Critical

**Status:** ⚠️ **Incomplete data for this check** (no result was reported for this check)

**Score:** N/A &nbsp;&nbsp; **State:** **UNCONFIRMED**

**Evidence:**

> _No evidence was provided for this check._
"""


def test_real_blocked_report_parses_failing_checks_with_evidence():
    target, reason = parse_failure_target("audit", REAL_BLOCKED_REPORT)
    assert reason is None and target is not None
    assert [c.identifier for c in target.checks] == ["schema_contract_verification"]
    assert "unique and not_null" in target.checks[0].evidence


def test_real_failed_banner_report_is_still_fixable():
    """Cross-package contract: the auditor's report-only banner reads
    '**FAILED**' (not '**BLOCKED**') because it never gates a merge, while the
    Verdict enum stays BLOCKED. The fixer must still recognize such a report as
    a fixable target - it keys only on 'not PASSED' + a FAIL check, and the
    all-caps 'FAILED' parses cleanly under the [A-Z_]+ verdict regex."""
    failed_banner = REAL_BLOCKED_REPORT.replace("**BLOCKED**", "**FAILED**")
    target, reason = parse_failure_target("audit", failed_banner)
    assert reason is None and target is not None
    assert [c.identifier for c in target.checks] == ["schema_contract_verification"]


def test_real_artifact_report_is_rejected_not_fixed():
    target, reason = parse_failure_target("audit", REAL_ARTIFACT_REPORT)
    assert target is None
    assert "artifact" in reason


def test_real_passed_report_has_nothing_to_fix():
    passed = REAL_BLOCKED_REPORT.replace("**BLOCKED**", "**PASSED**").replace(
        "**State:** **FAIL**", "**State:** **PASS**"
    )
    target, reason = parse_failure_target("audit", passed)
    assert target is None
    assert "nothing to fix" in reason


def test_real_report_captures_severity_and_blocking_ids_excludes_advisory():
    """Live finding (bi-dbt #2533 round 8): the efficacy gate must not
    require an advisory check to pass - the fixer isn't responsible for
    (or allowed to make) doc/style fixes. blocking_identifiers drops
    known-advisory checks; unknown severity stays blocking."""
    from dbt_fixer.intake import parse_failure_target

    report = (
        "# Verdict: **BLOCKED**\n\n"
        "### Schema Contract Verification (`schema_contract_verification`)\n\n"
        "**Severity:** Critical &nbsp; **Score:** 0/100 &nbsp; **State:** **FAIL**\n\n"
        "**Evidence:**\n\n> mismatch\n\n"
        "### SQL Style and Testability (`sql_style_and_testability`)\n\n"
        "**Severity:** Advisory &nbsp; **Score:** 25/100 &nbsp; **State:** **FAIL**\n\n"
        "**Evidence:**\n\n> undocumented\n"
    )
    target, reason = parse_failure_target("audit", report)
    assert target is not None, reason
    ids = set(target.identifiers)
    assert ids == {"schema_contract_verification", "sql_style_and_testability"}
    # advisory dropped from the efficacy requirement; critical kept
    assert target.blocking_identifiers == ("schema_contract_verification",)


def test_unknown_severity_stays_blocking():
    """CI/legacy checks have no severity - they must remain required."""
    from dbt_fixer.intake import FailingCheck, FailureTarget

    target = FailureTarget(
        kind="ci",
        checks=(
            FailingCheck(identifier="a"),                       # unknown
            FailingCheck(identifier="b", severity="critical"),
            FailingCheck(identifier="c", severity="advisory"),
        ),
    )
    assert target.blocking_identifiers == ("a", "b")


def test_problem_summary_names_checks_with_evidence_and_leads_with_blocking():
    from dbt_fixer.intake import FailingCheck, FailureTarget

    target = FailureTarget(
        kind="audit",
        checks=(
            FailingCheck(identifier="sql_style", severity="advisory", evidence="minor nit"),
            FailingCheck(
                identifier="schema_contract_verification",
                severity="critical",
                evidence="yml   declares\ncol `x` the model omits",
            ),
        ),
    )
    summary = target.problem_summary
    lines = summary.splitlines()
    # One `- ` bullet per check; blocking (critical) check leads, advisory after.
    assert all(l.startswith("- ") for l in lines)
    assert summary.index("schema_contract_verification") < summary.index("sql_style")
    # Evidence reduced to its first point, marker-free, on that check's line.
    assert "yml declares" in summary


def test_problem_summary_truncates_long_evidence_and_caps_checks():
    from dbt_fixer.intake import FailingCheck, FailureTarget

    target = FailureTarget(
        kind="ci",
        checks=tuple(
            FailingCheck(identifier=f"check_{i}", evidence="e" * 300) for i in range(5)
        ),
    )
    summary = target.problem_summary
    assert "..." in summary  # long evidence truncated
    assert "(+2 more)" in summary  # only 3 checks shown, 2 summarized
    assert "check_4" not in summary


def test_problem_summary_handles_no_evidence():
    from dbt_fixer.intake import FailingCheck, FailureTarget

    target = FailureTarget(kind="ci", checks=(FailingCheck(identifier="my_test"),))
    assert target.problem_summary == "- `my_test`"
