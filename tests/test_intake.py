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
