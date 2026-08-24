from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import AnswerState, QueryOperation, QueryRequest
from src.query.service import QueryService
from src.utils.ais_anomaly import detect_sudden_jump_events_from_parquet


EXACT_PORT_SCOPED_PROMPT = "Investigate unusual AIS jumps near SEGVX in March 2022."


def _jump_rows(*, include_observed_locode: bool = True) -> pd.DataFrame:
    rows = [
        {
            "stable_id": "segvx_1",
            "event_kind": "ais_position",
            "mmsi": "111111111",
            "timestamp_full": "2022-03-10T00:00:00Z",
            "latitude": 60.0,
            "longitude": 17.0,
            "locode_norm": "SEGVX",
            "destination_norm": "SEGOT",
        },
        {
            "stable_id": "segvx_2",
            "event_kind": "ais_position",
            "mmsi": "111111111",
            "timestamp_full": "2022-03-10T00:10:00Z",
            "latitude": 60.5,
            "longitude": 17.5,
            "locode_norm": "SEGVX",
            "destination_norm": "SEGOT",
        },
        {
            "stable_id": "segot_1",
            "event_kind": "ais_position",
            "mmsi": "222222222",
            "timestamp_full": "2022-03-10T01:00:00Z",
            "latitude": 57.0,
            "longitude": 11.0,
            "locode_norm": "SEGOT",
            "destination_norm": "SEGVX",
        },
        {
            "stable_id": "segot_2",
            "event_kind": "ais_position",
            "mmsi": "222222222",
            "timestamp_full": "2022-03-10T01:10:00Z",
            "latitude": 57.5,
            "longitude": 11.5,
            "locode_norm": "SEGOT",
            "destination_norm": "SEGVX",
        },
    ]
    frame = pd.DataFrame(rows)
    if not include_observed_locode:
        frame = frame.drop(columns=["locode_norm"])
    return frame


def _service(tmp_path: Path, events_path: Path) -> QueryService:
    processed = Path("data/processed")
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / "ais_scope.sqlite3"),
        events_path=events_path,
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


def test_row_level_detector_applies_exact_observed_locode_and_excludes_mismatches(tmp_path: Path) -> None:
    events_path = tmp_path / "events.parquet"
    _jump_rows().to_parquet(events_path, index=False)

    payload = detect_sudden_jump_events_from_parquet(
        events_path,
        locode="SEGVX",
        date_from="2022-03-01",
        date_to="2022-03-31",
    )

    assert payload["scope_status"] == "applied"
    assert payload["scope_applied"] is True
    assert payload["scope_field"] == "locode_norm"
    assert payload["count"] == 1
    assert {event["stable_id"] for event in payload["events"]} == {"segvx_2"}
    assert {event["port"] for event in payload["events"]} == {"SEGVX"}


def test_exact_canonical_port_scoped_prompt_passes_filter_and_excludes_other_ports(tmp_path: Path) -> None:
    events_path = tmp_path / "events.parquet"
    _jump_rows().to_parquet(events_path, index=False)

    envelope = _service(tmp_path, events_path).query(
        QueryRequest(question=EXACT_PORT_SCOPED_PROMPT, conversation_id="exact_ais_port_scope")
    )

    assert envelope.plan.operation == QueryOperation.AIS_JUMP
    assert envelope.applied_scope.ports == ["SEGVX"]
    assert envelope.state == AnswerState.COMPUTED
    assert envelope.confidence == "high"
    assert envelope.assurance is not None
    assert envelope.assurance.status == "verified"
    assert envelope.availability is not None
    assert envelope.availability.code == "available"
    assert "Detected 1 potential sudden AIS coordinate jumps" in envelope.answer
    assert next(dataset for dataset in envelope.datasets if dataset.id == "table").row_count == 1
    assert {visual.kind for visual in envelope.visualizations} == {"map", "cartesian"}
    assert all(visual.kind != "omitted" for visual in envelope.visualizations)


def test_port_scoped_ais_request_is_explicitly_unsupported_without_observed_locode(tmp_path: Path) -> None:
    events_path = tmp_path / "events_without_observed_port.parquet"
    _jump_rows(include_observed_locode=False).to_parquet(events_path, index=False)

    envelope = _service(tmp_path, events_path).query(
        QueryRequest(question=EXACT_PORT_SCOPED_PROMPT, conversation_id="unsupported_ais_port_scope")
    )

    assert envelope.plan.operation == QueryOperation.AIS_JUMP
    assert envelope.applied_scope.ports == ["SEGVX"]
    assert envelope.state == AnswerState.UNSUPPORTED
    assert "Port-scoped AIS jump analysis is unavailable for SEGVX" in envelope.answer
    assert "destination text cannot establish an observed port location" in envelope.answer.lower()
    assert not envelope.datasets
    assert envelope.visualizations[0].kind == "omitted"


class _BrokenJumpRetriever:
    retrieval_backend = "broken-test-index"

    @staticmethod
    def detect_sudden_jumps(*, filters):
        raise RuntimeError("HNSW collection is unavailable")


def test_broken_jump_index_falls_back_to_row_level_events(tmp_path: Path) -> None:
    events_path = tmp_path / "events.parquet"
    _jump_rows().to_parquet(events_path, index=False)
    service = _service(tmp_path, events_path)
    service.retriever = _BrokenJumpRetriever()  # type: ignore[assignment]

    envelope = service.query(
        QueryRequest(
            question=EXACT_PORT_SCOPED_PROMPT,
            conversation_id="broken_index_row_fallback",
            top_k_evidence=0,
        )
    )

    assert envelope.state == AnswerState.COMPUTED
    assert "Detected 1 potential sudden AIS coordinate jumps" in envelope.answer
    assert next(dataset for dataset in envelope.datasets if dataset.id == "table").row_count == 1
    assert any("AIS jump index query failed" in warning for warning in envelope.trace.warnings)


def test_broken_jump_index_without_event_rows_returns_canonical_no_data(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, tmp_path / "missing-events.parquet")
    service.retriever = _BrokenJumpRetriever()  # type: ignore[assignment]

    envelope = service.query(
        QueryRequest(
            question="Show suspicious AIS jumps for MMSI 246521000 on 2022-03-10.",
            conversation_id="broken_index_no_rows",
            top_k_evidence=0,
        )
    )

    assert envelope.state == AnswerState.NO_DATA
    assert not envelope.datasets
    assert [visual.kind for visual in envelope.visualizations] == ["omitted"]
    assert any("AIS jump index query failed" in warning for warning in envelope.trace.warnings)
