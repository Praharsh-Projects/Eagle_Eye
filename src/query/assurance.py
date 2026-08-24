"""Compatibility assurance metadata for canonical Eagle Eye answers.

Structured analytics are published when the canonical service has validated
finite rows and scope reconciliation.  Method diagnostics remain trace-only;
they are not a second publication gate.  Source-grounded research retains its
separate evidence and citation requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_LIVE_OPERATIONS = {
    "live_port_arrivals",
    "vessel_eta",
    "vessel_delay",
    "eta_comparison",
}
_FORECAST_OPERATIONS = {
    "forecast_arrivals",
    "forecast_congestion",
    "forecast_comparison",
}
_NON_NUMERIC_MODES = {"app_help"}
_UNAVAILABLE_STATES = {"NO_DATA", "NO_CURRENT_DATA", "ERROR", "ASSURANCE_UNAVAILABLE"}
_NOT_APPLICABLE_STATES = {"CLARIFICATION_REQUIRED", "UNSUPPORTED"}
_AVAILABILITY_CODES = {
    "available",
    "no_data",
    "source_unavailable",
    "source_stale",
    "coverage_unavailable",
    "ambiguous_match",
    "not_applicable",
}


@dataclass(frozen=True)
class AssuranceDecision:
    """Publication decision represented independently of the public schema."""

    status: str
    level: str
    basis: str
    reason: str
    checks: tuple[str, ...]
    availability_code: str
    publish_numeric: bool


def _confidence_token(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if "high" in normalized:
        return "high"
    if "medium" in normalized or "moderate" in normalized:
        return "medium"
    if "low" in normalized:
        return "low"
    return "not_applicable"


def evaluate_assurance(
    *,
    mode: str,
    operation: str,
    state: str,
    confidence: str,
    caveats: Iterable[str] = (),
    evidence_count: int = 0,
    retrieval_status: str = "not_applicable",
    availability_code: str | None = None,
) -> AssuranceDecision:
    """Return wire-compatible assurance metadata for one execution outcome."""

    confidence_token = _confidence_token(confidence)
    checks = [
        f"result_state={state}",
        f"legacy_confidence={confidence_token}",
    ]

    if state in _NOT_APPLICABLE_STATES or mode in _NON_NUMERIC_MODES:
        return AssuranceDecision(
            status="not_applicable",
            level="not_applicable",
            basis="system_response",
            reason="This response does not make a scored numeric or sourced factual claim.",
            checks=tuple(checks),
            availability_code="not_applicable",
            publish_numeric=False,
        )

    requested_availability = (
        availability_code
        if availability_code in _AVAILABILITY_CODES
        else None
    )

    if state in _UNAVAILABLE_STATES:
        return AssuranceDecision(
            status="unavailable",
            level="not_applicable",
            basis="system_response",
            reason="No validated result is available for publication.",
            checks=tuple(checks),
            availability_code=requested_availability
            or ("source_unavailable" if state == "NO_CURRENT_DATA" else "no_data"),
            publish_numeric=False,
        )

    if state == "GENERAL" and confidence_token == "not_applicable":
        return AssuranceDecision(
            status="not_applicable",
            level="not_applicable",
            basis="system_response",
            reason="This general response does not claim a verified analytics value.",
            checks=tuple(checks),
            availability_code="not_applicable",
            publish_numeric=False,
        )

    if mode == "maritime_research":
        checks.extend(
            (
                f"evidence_count={max(0, int(evidence_count))}",
                f"retrieval_status={retrieval_status}",
            )
        )
        if confidence_token != "high" or evidence_count <= 0 or retrieval_status != "ok":
            return AssuranceDecision(
                status="unavailable",
                level="not_applicable",
                basis="source_grounded_research",
                reason="The research response lacks fully validated authoritative claim coverage.",
                checks=tuple(checks),
                availability_code="coverage_unavailable",
                publish_numeric=False,
            )

    if mode == "general_chat" and confidence_token != "not_applicable":
        checks.extend(
            (
                f"evidence_count={max(0, int(evidence_count))}",
                f"retrieval_status={retrieval_status}",
            )
        )
        if (
            confidence_token != "high"
            or evidence_count <= 0
            or retrieval_status != "ok"
        ):
            return AssuranceDecision(
                status="unavailable",
                level="not_applicable",
                basis="source_grounded_research",
                reason="The current factual response lacks fully validated source coverage.",
                checks=tuple(checks),
                availability_code="coverage_unavailable",
                publish_numeric=False,
            )

    # Current-source operations remain bounded by the upstream freshness and
    # identity checks represented by the high source-validation label.  The
    # direct-result rule applies after those checks, never instead of them.
    if operation in _LIVE_OPERATIONS and confidence_token != "high":
        checks.append("live_source_validation=failed")
        return AssuranceDecision(
            status="unavailable",
            level="not_applicable",
            basis="official_live_source",
            reason="The current-source freshness or identity checks did not pass.",
            checks=tuple(checks),
            availability_code=requested_availability or "source_unavailable",
            publish_numeric=False,
        )

    if state not in {"COMPUTED", "PARTIAL", "RETRIEVED", "GENERAL"}:
        return AssuranceDecision(
            status="unavailable",
            level="not_applicable",
            basis="system_response",
            reason="No result rows are available.",
            checks=tuple(checks),
            availability_code=requested_availability or "no_data",
            publish_numeric=False,
        )

    checks.append(
        "live_source_validation=passed"
        if operation in _LIVE_OPERATIONS
        else "canonical_result_validation=passed"
    )
    return AssuranceDecision(
        status="verified",
        level="high",
        basis=(
            "official_live_source"
            if operation in _LIVE_OPERATIONS
            else "validated_model"
            if operation in _FORECAST_OPERATIONS
            else "source_grounded_research"
            if mode in {"maritime_research", "general_chat"}
            else "direct_computation"
        ),
        reason="Canonical result validation passed.",
        checks=tuple(checks),
        availability_code="available",
        publish_numeric=True,
    )
