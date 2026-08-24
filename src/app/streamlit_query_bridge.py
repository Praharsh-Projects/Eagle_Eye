"""Presentation bridge from the canonical query service to the legacy Streamlit shell.

The Streamlit page is intentionally a view only: planning, validation, analytics,
context, and evidence all come from :class:`QueryService`.  The small legacy
result objects produced here exist solely because the restored report layout
already knows how to present those shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd

from src.carbon.query import CarbonResult
from src.forecast.forecast import ForecastResult
from src.kpi.query import AnalyticsResult
from src.query.models import (
    AnswerEnvelope,
    AnswerState,
    QueryFiltersPayload,
    QueryMode,
    QueryOperation,
    QueryRequest,
)
from src.query.service import QueryService


LegacyResult = Union[AnalyticsResult, ForecastResult, CarbonResult]


@dataclass(frozen=True)
class StreamlitEvidence:
    lines: List[str]
    rows: List[Dict[str, Any]]
    trace: Dict[str, Any]


@dataclass(frozen=True)
class StreamlitQueryResult:
    envelope: AnswerEnvelope
    result: LegacyResult
    evidence: StreamlitEvidence


def distinct_visual_summary(reason: str, summary: str) -> Optional[str]:
    """Return an accessibility caption only when it adds information."""

    normalized_reason = " ".join(str(reason).split()).casefold()
    normalized_summary = " ".join(str(summary).split()).casefold()
    if not normalized_summary or normalized_summary == normalized_reason:
        return None
    return str(summary).strip()


@dataclass(frozen=True)
class CanonicalPresentation:
    """Truthful labels for rendering an :class:`AnswerEnvelope` in Streamlit."""

    answer: str
    source_label: str
    source_detail: str
    confidence_label: str
    state_label: str
    method_steps: List[str]
    applied_scope: List[str]
    freshness: List[str]


def canonical_presentation(envelope: AnswerEnvelope) -> CanonicalPresentation:
    """Build display copy only from canonical envelope fields.

    The restored Streamlit shell still accepts legacy result shapes for charts and
    tables.  It must not infer result semantics from those compatibility objects.
    """

    state = envelope.state
    assurance = envelope.assurance
    availability = envelope.availability
    if state == AnswerState.ASSURANCE_UNAVAILABLE:
        source_label = "Source-grounded answer unavailable"
        source_detail = "The attached sources do not support a complete factual answer."
    elif (
        state == AnswerState.COMPUTED
        and assurance is not None
        and assurance.basis == "official_live_source"
    ):
        source_label = "Official live source"
        source_detail = "Validated current or future information from the named official source."
    elif state == AnswerState.COMPUTED:
        source_label = "Structured data"
        source_detail = "Values from the structured datasets for the applied scope."
    elif state == AnswerState.PARTIAL:
        source_label = "Structured data"
        source_detail = "Available values are shown for the applied scope."
    elif state == AnswerState.RETRIEVED:
        source_label = "Source-grounded research"
        source_detail = "Answer grounded in the evidence sources attached to this result."
    elif state == AnswerState.GENERAL:
        source_label = "General assistance"
        source_detail = "No historical analytics computation is claimed for this response."
    elif state == AnswerState.CLARIFICATION_REQUIRED:
        source_label = "Clarification required"
        source_detail = "No analytics computation ran because the requested scope is ambiguous."
    elif state == AnswerState.UNSUPPORTED:
        source_label = "Unsupported request"
        source_detail = "No analytics computation ran because no supported operation matched."
    elif state == AnswerState.NO_CURRENT_DATA:
        source_label = "Unavailable — historical data only"
        source_detail = "The validated historical datasets cannot establish a current or live value."
    elif state == AnswerState.NO_DATA:
        source_label = "Unavailable — no matching data"
        source_detail = "No validated result was available for the applied query scope."
    else:
        source_label = "Unavailable — query error"
        source_detail = "No validated result was produced."

    confidence_label = "Available" if state in {
        AnswerState.COMPUTED,
        AnswerState.PARTIAL,
        AnswerState.RETRIEVED,
        AnswerState.GENERAL,
    } else "Not applicable"

    if (
        availability is not None
        and availability.code not in {"available", "not_applicable"}
    ):
        source_detail = (
            f"{source_detail} Availability: {availability.code.replace('_', ' ')}."
        )

    method_steps: List[str] = []

    scope = envelope.applied_scope
    applied_scope: List[str] = []
    if scope.ports:
        applied_scope.append(f"Ports: {', '.join(scope.ports)}.")
    if scope.origin_port or scope.destination_port:
        applied_scope.append(
            f"Route: {scope.origin_port or 'unspecified'} to {scope.destination_port or 'unspecified'}."
        )
    if scope.date_from or scope.date_to:
        applied_scope.append(
            f"Requested dates: {scope.date_from or 'unspecified'} to {scope.date_to or 'unspecified'}."
        )
    if scope.target_date:
        applied_scope.append(f"Target date: {scope.target_date}.")
    if scope.vessel_type:
        applied_scope.append(f"Vessel type: {scope.vessel_type}.")
    if scope.vessel_name:
        applied_scope.append(f"Vessel: {scope.vessel_name}.")
    if scope.mmsi:
        applied_scope.append(f"MMSI: {scope.mmsi}.")
    if scope.imo:
        applied_scope.append(f"IMO: {scope.imo}.")
    if scope.horizon_hours:
        applied_scope.append(f"Live horizon: {scope.horizon_hours} hours.")
    if scope.source_scope:
        applied_scope.append(f"Source scope: {scope.source_scope}.")
    if not applied_scope:
        applied_scope.append(
            "No explicit port, route, date, vessel identity, or vessel-type scope was applied."
        )

    freshness = [envelope.freshness.message]
    if envelope.freshness.as_of:
        as_of = f"Source snapshot as of: {envelope.freshness.as_of}."
        if as_of.casefold() != envelope.freshness.message.casefold():
            freshness.append(as_of)
    coverage = (
        f"Global dataset coverage: {envelope.freshness.data_from or 'unavailable'} to "
        f"{envelope.freshness.data_to or 'unavailable'}."
    )
    if coverage.casefold() != envelope.freshness.message.casefold():
        freshness.append(coverage)

    return CanonicalPresentation(
        answer=envelope.answer,
        source_label=source_label,
        source_detail=source_detail,
        confidence_label=confidence_label,
        state_label=(
            "Analysis result"
            if state in {AnswerState.COMPUTED, AnswerState.PARTIAL, AnswerState.RETRIEVED, AnswerState.GENERAL}
            else state.value.replace("_", " ").title()
        ),
        method_steps=method_steps,
        applied_scope=applied_scope,
        freshness=freshness,
    )


def dataset_frame(envelope: AnswerEnvelope, dataset_id: str) -> pd.DataFrame:
    """Return one validated envelope dataset as a presentation dataframe."""

    dataset = next((item for item in envelope.datasets if item.id == dataset_id), None)
    if dataset is None:
        return pd.DataFrame()
    frame = pd.DataFrame(dataset.rows)
    for column in dataset.columns:
        if column.field not in frame.columns:
            continue
        if column.data_type == "datetime":
            frame[column.field] = pd.to_datetime(frame[column.field], errors="coerce", utc=True)
        elif column.data_type in {"number", "integer"}:
            frame[column.field] = pd.to_numeric(frame[column.field], errors="coerce")
    return frame


def _coverage_notes(envelope: AnswerEnvelope) -> List[str]:
    prefix = "Global data freshness" if envelope.freshness.historical else "Live source freshness"
    notes = [f"{prefix}: {envelope.freshness.message}"]
    if envelope.freshness.as_of:
        notes.append(f"Source snapshot as of: {envelope.freshness.as_of}.")
    if envelope.freshness.data_from or envelope.freshness.data_to:
        coverage_prefix = (
            "Global dataset coverage"
            if envelope.freshness.historical
            else "Live source window"
        )
        notes.append(
            f"{coverage_prefix}: {envelope.freshness.data_from or 'unavailable'} to "
            f"{envelope.freshness.data_to or 'unavailable'}."
        )
    return notes


def _legacy_result(envelope: AnswerEnvelope) -> LegacyResult:
    table = dataset_frame(envelope, "table")
    chart = dataset_frame(envelope, "chart")
    table_or_none = table if not table.empty else None
    chart_or_none = chart if not chart.empty else None
    coverage_notes = _coverage_notes(envelope)
    caveats: List[str] = []
    status = envelope.state.value

    if envelope.plan.operation in {
        QueryOperation.FORECAST_ARRIVALS,
        QueryOperation.FORECAST_CONGESTION,
        QueryOperation.FORECAST_COMPARISON,
    }:
        history = pd.DataFrame()
        forecast = pd.DataFrame()
        if not chart.empty:
            if "actual" in chart.columns:
                history_cols = [field for field in ("date", "actual") if field in chart.columns]
                history = chart.loc[chart["actual"].notna(), history_cols].copy()
            if "predicted" in chart.columns:
                forecast_cols = [
                    field
                    for field in ("date", "predicted", "lower", "upper")
                    if field in chart.columns
                ]
                forecast = chart.loc[chart["predicted"].notna(), forecast_cols].copy()
        return ForecastResult(
            status=status,
            answer=envelope.answer,
            history=history if not history.empty else None,
            forecast=forecast if not forecast.empty else table_or_none,
            coverage_notes=coverage_notes,
            caveats=caveats,
        )

    if envelope.plan.operation == QueryOperation.CARBON:
        evidence_ids = [item.id for item in envelope.evidence]
        presentation = canonical_presentation(envelope)
        return CarbonResult(
            status=status,
            answer=envelope.answer,
            table=table_or_none,
            chart=chart_or_none,
            coverage_notes=coverage_notes,
            caveats=caveats,
            boundary=envelope.plan.carbon_boundary,
            pollutants=list(envelope.plan.pollutants),
            source_label=presentation.source_label,
            confidence_label="high",
            confidence_reason="Canonical result available.",
            uncertainty_interval={},
            params_version=envelope.trace.data_manifest_version or "unavailable",
            evidence_ids=evidence_ids,
            segment_ids=[],
            result_state=status,
            diagnostics={
                "trace_id": envelope.trace.trace_id,
                "result_hash": envelope.trace.result_hash,
                "visualization_decision": envelope.trace.visualization_decision,
            },
        )

    return AnalyticsResult(
        status=status,
        answer=envelope.answer,
        table=table_or_none,
        chart=chart_or_none,
        coverage_notes=coverage_notes,
        caveats=caveats,
    )


def _evidence(envelope: AnswerEnvelope) -> StreamlitEvidence:
    lines: List[str] = []
    rows: List[Dict[str, Any]] = []
    for item in envelope.evidence:
        detail = item.excerpt or item.title
        lines.append(f"{item.source_type} | {item.title} | {detail}")
        rows.append(
            {
                "evidence_id": item.id,
                "source_type": item.source_type,
                "title": item.title,
                "excerpt": item.excerpt,
                "url": item.url,
                **dict(item.metadata),
            }
        )
    trace = envelope.trace.model_dump(mode="json")
    canonical_retrieval_status = envelope.trace.retrieval_status
    trace.update(
        {
            "retrieval_status": (
                canonical_retrieval_status
                if canonical_retrieval_status
                else ("ok" if envelope.evidence else "not_required")
            ),
            "reason": (
                envelope.assurance.reason
                if envelope.assurance is not None
                else "Canonical evidence is attached to this answer."
                if envelope.evidence
                else "No external evidence was required for this legacy canonical result."
            ),
            "mode": envelope.mode.value,
            "query_latency_ms": envelope.trace.latency_ms,
            "returned_items": len(rows),
            "assurance": (
                envelope.assurance.model_dump(mode="json")
                if envelope.assurance is not None
                else None
            ),
            "availability": (
                envelope.availability.model_dump(mode="json")
                if envelope.availability is not None
                else None
            ),
        }
    )
    return StreamlitEvidence(lines=lines, rows=rows, trace=trace)


def run_canonical_query(
    service: QueryService,
    *,
    question: str,
    conversation_id: str,
    top_k_evidence: int,
    user_filters: Optional[Mapping[str, Any]] = None,
) -> StreamlitQueryResult:
    """Execute one Streamlit request through the sole canonical pipeline."""

    filters = dict(user_filters or {})
    allowed_filters = {
        key: filters.get(key)
        for key in (
            "port",
            "date_from",
            "date_to",
            "vessel_type",
            "vessel_name",
            "mmsi",
            "imo",
            "anomaly",
        )
    }
    envelope = service.query(
        QueryRequest(
            question=question,
            conversation_id=conversation_id,
            top_k_evidence=min(10, max(0, int(top_k_evidence))),
            filters=QueryFiltersPayload(**allowed_filters),
        )
    )
    return StreamlitQueryResult(
        envelope=envelope,
        result=_legacy_result(envelope),
        evidence=_evidence(envelope),
    )
