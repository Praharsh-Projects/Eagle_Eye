from __future__ import annotations

import pytest

from src.query.assurance import AssuranceDecision, evaluate_assurance


def _evaluate(**overrides: object) -> AssuranceDecision:
    values: dict[str, object] = {
        "mode": "analytics",
        "operation": "arrivals",
        "state": "COMPUTED",
        "confidence": "high",
    }
    values.update(overrides)
    return evaluate_assurance(**values)


def test_validated_direct_analytics_is_published_with_compatible_metadata() -> None:
    decision = _evaluate()

    assert decision.status == "verified"
    assert decision.level == "high"
    assert decision.basis == "direct_computation"
    assert decision.availability_code == "available"
    assert decision.publish_numeric is True
    assert "canonical_result_validation=passed" in decision.checks


@pytest.mark.parametrize(
    ("confidence", "normalized"),
    [
        ("medium", "medium"),
        ("Moderate confidence", "medium"),
        ("low", "low"),
        ("unknown", "not_applicable"),
    ],
)
def test_legacy_confidence_labels_do_not_block_validated_structured_rows(
    confidence: str,
    normalized: str,
) -> None:
    decision = _evaluate(confidence=confidence)

    assert decision.status == "verified"
    assert decision.level == "high"
    assert decision.basis == "direct_computation"
    assert decision.availability_code == "available"
    assert decision.publish_numeric is True
    assert f"legacy_confidence={normalized}" in decision.checks
    assert "canonical_result_validation=passed" in decision.checks


def test_usable_partial_result_is_published_directly() -> None:
    decision = _evaluate(state="PARTIAL")

    assert decision.status == "verified"
    assert decision.level == "high"
    assert decision.availability_code == "available"
    assert decision.publish_numeric is True
    assert "result_state=PARTIAL" in decision.checks


@pytest.mark.parametrize(
    "caveat, expected_token",
    [
        ("Only partial coverage was available.", "partial coverage"),
        ("This is an arrivals-pressure proxy.", "proxy"),
        ("A heuristic match was used.", "heuristic"),
        ("The route was reconstructed from events.", "reconstructed"),
        ("The calculation used a fallback factor.", "fallback"),
    ],
)
def test_method_classification_caveats_do_not_block_validated_rows(
    caveat: str,
    expected_token: str,
) -> None:
    decision = _evaluate(caveats=[caveat])

    assert decision.status == "verified"
    assert decision.level == "high"
    assert decision.availability_code == "available"
    assert decision.publish_numeric is True
    assert all(expected_token not in check for check in decision.checks)


def test_carbon_publication_depends_on_canonical_row_validation_not_legacy_labels() -> None:
    passed = _evaluate(operation="carbon", confidence="High confidence")
    modeled_inventory = _evaluate(
        operation="carbon",
        confidence="High confidence",
        caveats=[
            "Carbon estimates are deterministic but proxy-based inventory outputs, not direct stack measurements."
        ],
    )
    failed = _evaluate(operation="carbon", confidence="low")
    proxy = _evaluate(
        operation="carbon",
        confidence="high",
        caveats=["Daily proxy downgrade applies to this estimate."],
    )

    assert (passed.status, passed.level, passed.publish_numeric) == ("verified", "high", True)
    assert (modeled_inventory.status, modeled_inventory.level, modeled_inventory.publish_numeric) == (
        "verified",
        "high",
        True,
    )
    assert (failed.status, failed.level, failed.publish_numeric) == (
        "verified",
        "high",
        True,
    )
    assert (proxy.status, proxy.level, proxy.publish_numeric) == (
        "verified",
        "high",
        True,
    )


@pytest.mark.parametrize(
    "operation",
    ["forecast_arrivals", "forecast_congestion", "forecast_comparison"],
)
def test_finite_forecasts_publish_regardless_of_legacy_quality_label(operation: str) -> None:
    verified = _evaluate(operation=operation, confidence="high")
    unvalidated = _evaluate(operation=operation, confidence="medium")
    fallback = _evaluate(
        operation=operation,
        confidence="high",
        caveats=["Seasonal fallback was used."],
    )

    assert verified.status == "verified"
    assert verified.basis == "validated_model"
    assert verified.publish_numeric is True
    assert unvalidated.status == "verified"
    assert unvalidated.basis == "validated_model"
    assert unvalidated.publish_numeric is True
    assert fallback.status == "verified"
    assert fallback.basis == "validated_model"
    assert fallback.publish_numeric is True


def test_research_requires_high_confidence_evidence_and_successful_retrieval() -> None:
    verified = _evaluate(
        mode="maritime_research",
        operation="research",
        state="RETRIEVED",
        confidence="high",
        evidence_count=2,
        retrieval_status="ok",
    )

    assert verified.status == "verified"
    assert verified.level == "high"
    assert verified.basis == "source_grounded_research"
    assert verified.availability_code == "available"
    assert verified.publish_numeric is True

    for overrides in (
        {"confidence": "medium", "evidence_count": 2, "retrieval_status": "ok"},
        {"confidence": "high", "evidence_count": 0, "retrieval_status": "ok"},
        {"confidence": "high", "evidence_count": 2, "retrieval_status": "unavailable"},
    ):
        decision = _evaluate(
            mode="maritime_research",
            operation="research",
            state="RETRIEVED",
            **overrides,
        )
        assert decision.status == "unavailable"
        assert decision.level == "not_applicable"
        assert decision.basis == "source_grounded_research"
        assert decision.publish_numeric is False


def test_current_factual_general_response_requires_high_confidence_and_evidence() -> None:
    verified = _evaluate(
        mode="general_chat",
        operation="general_response",
        state="GENERAL",
        confidence="high",
        evidence_count=1,
        retrieval_status="ok",
    )
    missing_evidence = _evaluate(
        mode="general_chat",
        operation="general_response",
        state="GENERAL",
        confidence="high",
        evidence_count=0,
        retrieval_status="ok",
    )
    unverified = _evaluate(
        mode="general_chat",
        operation="general_response",
        state="GENERAL",
        confidence="medium",
        evidence_count=1,
        retrieval_status="ok",
    )
    failed_retrieval = _evaluate(
        mode="general_chat",
        operation="general_response",
        state="GENERAL",
        confidence="high",
        evidence_count=1,
        retrieval_status="error",
    )

    assert verified.status == "verified"
    assert verified.basis == "source_grounded_research"
    assert verified.publish_numeric is True
    assert missing_evidence.status == "unavailable"
    assert missing_evidence.basis == "source_grounded_research"
    assert missing_evidence.publish_numeric is False
    assert unverified.status == "unavailable"
    assert unverified.basis == "source_grounded_research"
    assert unverified.publish_numeric is False
    assert failed_retrieval.status == "unavailable"
    assert failed_retrieval.publish_numeric is False


@pytest.mark.parametrize(
    "mode, state",
    [
        ("app_help", "GENERAL"),
        ("clarification", "CLARIFICATION_REQUIRED"),
        ("unsupported", "UNSUPPORTED"),
    ],
)
def test_help_clarification_and_unsupported_are_not_applicable(
    mode: str,
    state: str,
) -> None:
    decision = _evaluate(
        mode=mode,
        operation="help" if mode == "app_help" else "unsupported",
        state=state,
        confidence="high",
    )

    assert decision.status == "not_applicable"
    assert decision.level == "not_applicable"
    assert decision.basis == "system_response"
    assert decision.availability_code == "not_applicable"
    assert decision.publish_numeric is False


@pytest.mark.parametrize(
    "state, availability_code",
    [
        ("NO_DATA", "no_data"),
        ("NO_CURRENT_DATA", "source_unavailable"),
        ("ERROR", "no_data"),
        ("ASSURANCE_UNAVAILABLE", "no_data"),
    ],
)
def test_no_data_and_unavailable_states_never_publish_values(
    state: str,
    availability_code: str,
) -> None:
    decision = _evaluate(state=state, confidence="high")

    assert decision.status == "unavailable"
    assert decision.level == "not_applicable"
    assert decision.basis == "system_response"
    assert decision.availability_code == availability_code
    assert decision.publish_numeric is False


def test_specific_upstream_availability_reason_is_preserved() -> None:
    decision = _evaluate(
        state="NO_CURRENT_DATA",
        confidence="not_applicable",
        availability_code="source_stale",
    )

    assert decision.availability_code == "source_stale"


def test_unscored_general_response_is_not_applicable() -> None:
    decision = _evaluate(
        mode="general_chat",
        operation="general_response",
        state="GENERAL",
        confidence="not_applicable",
    )

    assert decision.status == "not_applicable"
    assert decision.level == "not_applicable"
    assert decision.availability_code == "not_applicable"
    assert decision.publish_numeric is False


@pytest.mark.parametrize(
    "operation",
    ["live_port_arrivals", "vessel_eta", "vessel_delay", "eta_comparison"],
)
def test_live_eta_requires_upstream_source_validation(operation: str) -> None:
    verified = _evaluate(operation=operation, confidence="high")
    unavailable = _evaluate(operation=operation, confidence="medium")

    assert verified.status == "verified"
    assert verified.level == "high"
    assert verified.basis == "official_live_source"
    assert verified.availability_code == "available"
    assert verified.publish_numeric is True
    assert "live_source_validation=passed" in verified.checks
    assert unavailable.status == "unavailable"
    assert unavailable.level == "not_applicable"
    assert unavailable.basis == "official_live_source"
    assert unavailable.availability_code == "source_unavailable"
    assert unavailable.publish_numeric is False
    assert "live_source_validation=failed" in unavailable.checks


@pytest.mark.parametrize("state", ["NO_CURRENT_DATA", "NO_DATA", "ERROR"])
def test_live_eta_upstream_unavailable_states_remain_unavailable(state: str) -> None:
    decision = _evaluate(
        operation="vessel_eta",
        state=state,
        confidence="high",
        availability_code="source_stale" if state == "NO_CURRENT_DATA" else "no_data",
    )

    assert decision.status == "unavailable"
    assert decision.level == "not_applicable"
    assert decision.publish_numeric is False
