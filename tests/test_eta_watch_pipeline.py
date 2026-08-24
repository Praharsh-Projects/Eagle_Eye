from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import AnswerState, ETAWatchIntent, QueryRequest
from src.query.planner import QueryPlanner
from src.query.service import QueryService


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
ETA_PROMPTS = [
    "Prepare a Sweden-bound shift handover for the next 12 hours: due soon, low-speed exceptions, ETA changes, and stale signals.",
    "Which AIS-visible vessels reporting Swedish destinations are due in the next 6 hours? Show vessel, destination, reported ETA, last position, speed, and observation time.",
    "Show the next AIS-visible vessel reporting an ETA for Stockholm, Gothenburg, Nynäshamn, Malmö, or Trelleborg.",
    "Which Sweden-bound vessels due in the next 6 hours are moving below 2 knots?",
    "Which Swedish destinations have the most AIS-reported inbound vessels in the next 24 hours?",
    "Which Sweden-bound vessels changed their reported ETA by more than 30 minutes in the last hour?",
    "Which Sweden-bound vessels have a stale position report or no valid reported ETA?",
    "Where is the next AIS-visible vessel reporting a Swedish destination, and what ETA is it transmitting?",
    "Build a 12-hour Baltic inbound watchlist for Tallinn, Riga, Klaipėda, Gdańsk, Helsinki, and Turku.",
    "Which Baltic-bound vessels are due in the next 6 hours, and where were they last observed?",
]
AD_HOC_NEXT_TEN_PROMPT = (
    "What are the next 10 vessel-reported ETAs for Stockholm, Gothenburg, "
    "Nynäshamn, Malmö, and Trelleborg?"
)


@dataclass
class _Result:
    status: str
    operation: str
    table: pd.DataFrame | None
    summary: dict[str, Any]
    sections: dict[str, pd.DataFrame]
    snapshot_at: datetime = NOW
    data_updated_at: datetime = NOW
    horizon_end: datetime | None = None
    answer: str = "Provider prose must not be published."
    coverage_notes: list[str] | None = None
    caveats: list[str] | None = None
    failure_reason: str | None = None
    source_kind: str = "aisstream_observation"

    def __post_init__(self) -> None:
        self.horizon_end = self.horizon_end or datetime(
            2026,
            7,
            28,
            12,
            tzinfo=timezone.utc,
        )
        self.coverage_notes = list(self.coverage_notes or [])
        self.caveats = list(self.caveats or [])

    @property
    def chart(self) -> pd.DataFrame | None:
        return self.table


def _vessel_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mmsi": "265000001",
                "vessel_name": "NORDIC STAR",
                "destination_locode": "SESTO",
                "destination_name": "Stockholm",
                "eta_utc": "2026-07-27T14:00:00Z",
                "eta_observed_at_utc": "2026-07-27T11:56:00Z",
                "latitude": 59.1,
                "longitude": 18.1,
                "sog_kn": 8.2,
                "position_observed_at_utc": "2026-07-27T11:58:00Z",
                "position_age_minutes": 2,
                "position_stale": False,
            },
            {
                "mmsi": "265000002",
                "vessel_name": "BALTIC ONE",
                "destination_locode": "SEGOT",
                "destination_name": "Gothenburg",
                "eta_utc": "2026-07-27T15:00:00Z",
                "eta_observed_at_utc": "2026-07-27T11:55:00Z",
                "latitude": 57.6,
                "longitude": 11.8,
                "sog_kn": 1.2,
                "position_observed_at_utc": "2026-07-27T11:57:00Z",
                "position_age_minutes": 3,
                "position_stale": False,
            },
        ]
    )


def _many_vessel_rows(
    *,
    valid_count: int,
    total_count: int = 12,
) -> pd.DataFrame:
    locodes = ("SESTO", "SEGOT", "SENYN", "SEMMA", "SETRG")
    names = ("Stockholm", "Gothenburg", "Nynäshamn", "Malmö", "Trelleborg")
    rows: list[dict[str, Any]] = []
    for index in range(total_count):
        eta = NOW + timedelta(hours=index + 1)
        rows.append(
            {
                "mmsi": f"26510{index:04d}",
                "vessel_name": f"VESSEL {index + 1:02d}",
                "destination_locode": locodes[index % len(locodes)],
                "destination_name": names[index % len(names)],
                "eta_utc": eta.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "eta_observed_at_utc": (
                    "2026-07-27T11:58:00Z"
                    if index < valid_count
                    else "2026-07-27T11:40:00Z"
                ),
            }
        )
    return pd.DataFrame(rows)


class _Provider:
    provider = "aisstream"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def source_health(self) -> SimpleNamespace:
        return SimpleNamespace(status="healthy")

    def capabilities(self) -> dict[str, Any]:
        return {"provider": self.provider, "available": True}

    def query(self, operation: str, **kwargs: Any) -> _Result:
        self.calls.append({"operation": operation, **kwargs})
        rows = _vessel_rows()
        if operation == "low_speed":
            rows = rows.iloc[[1]].reset_index(drop=True)
        elif operation == "destination_load":
            rows = pd.DataFrame(
                [
                    {
                        "destination_locode": "SESTO",
                        "destination_name": "Stockholm",
                        "vessel_count": 2,
                        "next_eta_utc": "2026-07-27T14:00:00Z",
                    },
                    {
                        "destination_locode": "SEGOT",
                        "destination_name": "Gothenburg",
                        "vessel_count": 1,
                        "next_eta_utc": "2026-07-27T15:00:00Z",
                    },
                ]
            )
            return _Result(
                "ok",
                operation,
                rows,
                {"matched_vessels": 3, "destination_count": 2},
                {},
            )
        elif operation == "eta_revisions":
            rows = pd.DataFrame(
                [
                    {
                        "mmsi": "265000001",
                        "vessel_name": "NORDIC STAR",
                        "destination_locode": "SESTO",
                        "destination_name": "Stockholm",
                        "previous_eta_utc": "2026-07-27T13:10:00Z",
                        "current_eta_utc": "2026-07-27T14:00:00Z",
                        "eta_revision_minutes": 50,
                        "revision_observed_at_utc": "2026-07-27T11:50:00Z",
                    }
                ]
            )
        elif operation == "stale_missing":
            rows = pd.DataFrame(
                [
                    {
                        "mmsi": "265000003",
                        "vessel_name": "WATCH REQUIRED",
                        "destination_locode": "SENYN",
                        "destination_name": "Nynäshamn",
                        "eta_utc": None,
                        "latitude": None,
                        "longitude": None,
                        "position_observed_at_utc": None,
                        "position_age_minutes": None,
                        "position_stale": True,
                        "validation_reasons": [
                            "missing_position",
                            "missing_eta",
                        ],
                    }
                ]
            )
        summary = {
            "matched_vessels": len(rows),
            "displayed_vessels": len(rows),
        }
        sections: dict[str, pd.DataFrame] = {}
        if operation == "shift_handover":
            revision = pd.DataFrame(
                [
                    {
                        "mmsi": "265000001",
                        "eta_revision_minutes": 50,
                        "revision_observed_at_utc": "2026-07-27T11:50:00Z",
                    }
                ]
            )
            sections = {
                "inbound_watchlist": rows,
                "low_speed": rows.iloc[[1]].reset_index(drop=True),
                "eta_revisions": revision,
                "stale_missing": pd.DataFrame(
                    [
                        {
                            "mmsi": "265000099",
                            "vessel_name": "STALE SIGNAL",
                            "destination_locode": "SESTO",
                            "destination_name": "Stockholm",
                            "eta_utc": "2026-06-20T08:00:00Z",
                            "eta_observed_at_utc": "2026-07-27T11:56:00Z",
                            "latitude": None,
                            "longitude": None,
                            "position_observed_at_utc": None,
                            "position_age_minutes": None,
                            "position_stale": True,
                        }
                    ]
                ),
            }
            summary.update(
                {
                    "inbound_vessels": 2,
                    "low_speed_vessels": 1,
                    "eta_revision_vessels": 1,
                    "stale_or_missing_vessels": 1,
                }
            )
        return _Result("ok", operation, rows, summary, sections)


class _ManyVesselProvider(_Provider):
    def __init__(
        self,
        *,
        valid_count: int,
        total_count: int = 12,
    ) -> None:
        super().__init__()
        self.valid_count = valid_count
        self.total_count = total_count

    def query(self, operation: str, **kwargs: Any) -> _Result:
        self.calls.append({"operation": operation, **kwargs})
        rows = _many_vessel_rows(
            valid_count=self.valid_count,
            total_count=self.total_count,
        )
        return _Result(
            "ok",
            operation,
            rows,
            {
                "matched_vessels": len(rows),
                "displayed_vessels": len(rows),
            },
            {},
        )


def _service(
    tmp_path: Path,
    provider: _Provider | None,
) -> QueryService:
    processed = Path("data/processed")
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / "eta-watch.sqlite3"),
        live_eta=provider,
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


@pytest.mark.parametrize(
    ("index", "intent", "horizon"),
    [
        (0, ETAWatchIntent.SHIFT_HANDOVER, 12),
        (1, ETAWatchIntent.INBOUND_WATCHLIST, 6),
        (2, ETAWatchIntent.INBOUND_WATCHLIST, 24),
        (3, ETAWatchIntent.LOW_SPEED_EXCEPTIONS, 6),
        (4, ETAWatchIntent.DESTINATION_LOAD, 24),
        (5, ETAWatchIntent.ETA_REVISIONS, 48),
        (6, ETAWatchIntent.SIGNAL_QUALITY, 48),
        (7, ETAWatchIntent.VESSEL_STATUS, 24),
        (8, ETAWatchIntent.INBOUND_WATCHLIST, 12),
        (9, ETAWatchIntent.INBOUND_WATCHLIST, 6),
    ],
)
def test_eta_watch_samples_have_distinct_typed_intents(
    index: int,
    intent: ETAWatchIntent,
    horizon: int,
) -> None:
    plan = QueryPlanner().plan(ETA_PROMPTS[index])

    assert plan.eta_watch_intent == intent
    assert plan.horizon_hours == horizon
    assert plan.source_scope == "aisstream"
    assert plan.date_scope.is_current is True
    if index == 2:
        assert plan.limit == 1


@pytest.mark.parametrize(
    "question",
    [
        "Show the official arrival schedule at Stockholm.",
        "How delayed is the next vessel at Helsinki?",
        "Which berth is assigned to the next vessel at Gothenburg?",
    ],
)
def test_official_delay_and_berth_questions_are_honestly_unsupported(
    tmp_path: Path,
    question: str,
) -> None:
    provider = _Provider()
    envelope = _service(tmp_path, provider).query(
        QueryRequest(question=question, top_k_evidence=0)
    )

    assert envelope.state == AnswerState.UNSUPPORTED
    assert provider.calls == []
    assert "AISStream does not provide" in envelope.answer
    assert [visual.kind for visual in envelope.visualizations] == ["omitted"]


def test_inbound_answer_and_facts_are_deterministic_and_field_complete(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    envelope = _service(tmp_path, provider).query(
        QueryRequest(question=ETA_PROMPTS[1], top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.operational_brief is not None
    assert envelope.operational_brief.intent == ETAWatchIntent.INBOUND_WATCHLIST
    assert envelope.operational_brief.matched_count == 2
    assert envelope.operational_brief.displayed_count == 2
    assert "NORDIC STAR" in envelope.answer
    assert "Stockholm (SESTO)" in envelope.answer
    assert "2026-07-27T14:00:00Z" in envelope.answer
    assert "59.1000, 18.1000" in envelope.answer
    assert "8.2 knots" in envelope.answer
    assert "2026-07-27T11:58:00Z" in envelope.answer
    assert "Provider prose" not in envelope.answer
    assert "not an official schedule" in envelope.answer
    assert {visual.kind for visual in envelope.visualizations} == {
        "map",
        "timeline",
    }
    assert next(
        fact.value for fact in envelope.facts if fact.name == "matched_count"
    ) == 2
    assert all(not fact.name.startswith("number_") for fact in envelope.facts)
    assert provider.calls[0]["operation"] == "inbound_watchlist"
    assert "SESTO" in provider.calls[0]["ports"]
    assert "SEGOT" in provider.calls[0]["ports"]


def test_next_ten_lists_ten_validated_vessels_in_answer_brief_and_facts(
    tmp_path: Path,
) -> None:
    provider = _ManyVesselProvider(valid_count=12)
    envelope = _service(tmp_path, provider).query(
        QueryRequest(question=AD_HOC_NEXT_TEN_PROMPT, top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.plan.limit == 10
    assert provider.calls[0]["limit"] == 500
    assert envelope.operational_brief is not None
    assert envelope.operational_brief.matched_count == 12
    assert envelope.operational_brief.displayed_count == 10
    assert len(envelope.operational_brief.prioritized_items) == 10
    dataset = next(item for item in envelope.datasets if item.id == "table")
    assert dataset.row_count == 10
    assert "Showing 10 of 12" in envelope.answer
    for index in range(1, 11):
        assert f"VESSEL {index:02d}" in envelope.answer
        assert any(
            fact.name == f"displayed_{index}_vessel"
            and fact.value == f"VESSEL {index:02d}"
            for fact in envelope.facts
        )
    assert "VESSEL 11" not in envelope.answer
    assert all(
        not fact.name.startswith("displayed_11_")
        for fact in envelope.facts
    )


def test_next_ten_does_not_inflate_stale_candidates_into_validated_matches(
    tmp_path: Path,
) -> None:
    provider = _ManyVesselProvider(valid_count=2)
    envelope = _service(tmp_path, provider).query(
        QueryRequest(question=AD_HOC_NEXT_TEN_PROMPT, top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.operational_brief is not None
    assert envelope.operational_brief.matched_count == 2
    assert envelope.operational_brief.displayed_count == 2
    assert len(envelope.operational_brief.prioritized_items) == 2
    dataset = next(item for item in envelope.datasets if item.id == "table")
    assert dataset.row_count == 2
    assert (
        "Only 2 source-validated AIS-visible vessel signals are currently "
        "available for the requested next 10"
    ) in envelope.answer
    assert (
        "10 other matching vessel signals are retained separately as awaiting "
        "a fresh ETA broadcast"
    ) in envelope.answer
    assert "Showing 2 of 2" in envelope.answer
    assert "VESSEL 01" in envelope.answer
    assert "VESSEL 02" in envelope.answer
    assert "VESSEL 03" not in envelope.answer
    assert next(
        fact.value for fact in envelope.facts if fact.name == "matched_count"
    ) == 2
    candidates = next(
        item
        for item in envelope.datasets
        if item.id == "eta_freshness_candidates"
    )
    assert candidates.row_count == 8
    assert all("reported_eta_utc" not in row for row in candidates.rows)
    assert all(
        row["validation_status"] == "awaiting_fresh_eta"
        for row in candidates.rows
    )
    assert all(
        "current publication requires 10 minutes or less"
        in row["validation_reason"]
        for row in candidates.rows
    )


def test_next_ten_reports_one_current_eight_stale_and_one_missing_slot(
    tmp_path: Path,
) -> None:
    provider = _ManyVesselProvider(valid_count=1, total_count=9)
    envelope = _service(tmp_path, provider).query(
        QueryRequest(question=AD_HOC_NEXT_TEN_PROMPT, top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.plan.horizon_hours == 48
    assert "Only 1 source-validated AIS-visible vessel signal is" in envelope.answer
    assert "signal are" not in envelope.answer
    assert (
        "8 other matching vessel signals are retained separately as awaiting "
        "a fresh ETA broadcast"
    ) in envelope.answer
    assert (
        "no additional matching vessel for 1 requested slot inside the "
        "48-hour window"
    ) in envelope.answer
    current = next(item for item in envelope.datasets if item.id == "table")
    candidates = next(
        item
        for item in envelope.datasets
        if item.id == "eta_freshness_candidates"
    )
    assert current.row_count == 1
    assert candidates.row_count == 8
    assert all(
        visual.dataset_id != "eta_freshness_candidates"
        for visual in envelope.visualizations
    )


def test_next_vessel_sample_returns_exactly_one_complete_vessel_record(
    tmp_path: Path,
) -> None:
    provider = _ManyVesselProvider(valid_count=12)
    envelope = _service(tmp_path, provider).query(
        QueryRequest(question=ETA_PROMPTS[2], top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    assert envelope.plan.limit == 1
    assert envelope.plan.horizon_hours == 24
    assert envelope.operational_brief is not None
    assert envelope.operational_brief.matched_count == 12
    assert envelope.operational_brief.displayed_count == 1
    assert len(envelope.operational_brief.prioritized_items) == 1
    dataset = next(item for item in envelope.datasets if item.id == "table")
    assert dataset.row_count == 1
    assert dataset.rows[0]["vessel_label"] == "VESSEL 01"
    timeline_dataset = next(
        item for item in envelope.datasets if item.id == "eta_timeline"
    )
    assert timeline_dataset.row_count == 1
    timeline = next(
        item for item in envelope.visualizations if item.kind == "timeline"
    )
    assert timeline.dataset_id == "eta_timeline"
    assert timeline.table_fallback_dataset_id == "table"
    assert timeline.lane_field is None
    assert timeline.title == "Next vessel-reported ETA"
    assert "the next vessel is listed by earliest reported ETA" in envelope.answer
    assert "VESSEL 01" in envelope.answer
    assert "VESSEL 02" not in envelope.answer
    assert "awaiting a fresh ETA broadcast" not in envelope.answer
    assert "requested slot" not in envelope.answer
    assert all(
        item.id != "eta_freshness_candidates"
        for item in envelope.datasets
    )


@pytest.mark.parametrize(
    ("prompt_index", "expected_visuals"),
    [
        (0, {"map", "timeline", "table"}),
        (3, {"map", "cartesian", "table"}),
        (4, {"cartesian"}),
        (5, {"cartesian"}),
        (6, {"table"}),
        (7, {"map", "timeline"}),
    ],
)
def test_eta_watch_intents_receive_semantic_visuals(
    tmp_path: Path,
    prompt_index: int,
    expected_visuals: set[str],
) -> None:
    envelope = _service(tmp_path, _Provider()).query(
        QueryRequest(question=ETA_PROMPTS[prompt_index], top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    assert {visual.kind for visual in envelope.visualizations} == expected_visuals
    assert all(visual.kind != "omitted" for visual in envelope.visualizations)


def test_shift_handover_never_publishes_a_stale_raw_eta_as_current(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path, _Provider()).query(
        QueryRequest(question=ETA_PROMPTS[0], top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    assert "including 2 due soon" in envelope.answer
    assert "2026-06-20T08:00:00Z" not in envelope.answer
    dataset = next(item for item in envelope.datasets if item.id == "table")
    stale = next(row for row in dataset.rows if row["mmsi"] == "265000099")
    assert stale["reported_eta_utc"] is None
    assert stale["source_reported_eta_utc"] == "2026-06-20T08:00:00Z"
    assert stale["is_missing_eta"] is True
    eta_timeline = next(
        item for item in envelope.datasets if item.id == "eta_timeline"
    )
    assert 0 < eta_timeline.row_count < dataset.row_count
    assert all(row["reported_eta_utc"] for row in eta_timeline.rows)
    assert [row["reported_eta_utc"] for row in eta_timeline.rows] == sorted(
        row["reported_eta_utc"] for row in eta_timeline.rows
    )
    assert all(row["mmsi"] != "265000099" for row in eta_timeline.rows)
    timeline = next(
        item for item in envelope.visualizations if item.kind == "timeline"
    )
    assert timeline.dataset_id == "eta_timeline"
    assert timeline.table_fallback_dataset_id == "table"
    assert timeline.lane_field == "vessel_label"
    assert timeline.title == "Due-soon vessel-reported ETAs"


def test_low_speed_query_uses_speed_ranking_instead_of_generic_eta_rail(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path, _Provider()).query(
        QueryRequest(question=ETA_PROMPTS[3], top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    assert all(visual.kind != "timeline" for visual in envelope.visualizations)
    speed_ranking = next(
        visual
        for visual in envelope.visualizations
        if visual.kind == "cartesian"
    )
    assert speed_ranking.id == "eta_low_speed_ranking"
    assert speed_ranking.chart_type == "bar"
    assert speed_ranking.orientation == "horizontal"
    assert speed_ranking.sort == "ascending"
    assert speed_ranking.y_fields == ["speed_kn"]
    assert speed_ranking.y_unit == "knots"
    assert len(speed_ranking.reference_lines) == 1
    assert speed_ranking.reference_lines[0].value == 2.0


def test_eta_revision_chart_preserves_absolute_magnitude_priority(
    tmp_path: Path,
) -> None:
    class _RevisionMagnitudeProvider(_Provider):
        def query(self, operation: str, **kwargs: Any) -> _Result:
            if operation != "eta_revisions":
                return super().query(operation, **kwargs)
            self.calls.append({"operation": operation, **kwargs})
            rows = pd.DataFrame(
                [
                    {
                        "mmsi": "265000001",
                        "vessel_name": "SMALL LATER",
                        "destination_locode": "SESTO",
                        "destination_name": "Stockholm",
                        "previous_eta_utc": "2026-07-27T13:10:00Z",
                        "current_eta_utc": "2026-07-27T14:00:00Z",
                        "eta_revision_minutes": 50,
                        "revision_observed_at_utc": "2026-07-27T11:50:00Z",
                    },
                    {
                        "mmsi": "265000002",
                        "vessel_name": "LARGE EARLIER",
                        "destination_locode": "SEGOT",
                        "destination_name": "Gothenburg",
                        "previous_eta_utc": "2026-07-27T17:00:00Z",
                        "current_eta_utc": "2026-07-27T15:00:00Z",
                        "eta_revision_minutes": -120,
                        "revision_observed_at_utc": "2026-07-27T11:52:00Z",
                    },
                ]
            )
            return _Result(
                "ok",
                operation,
                rows,
                {
                    "matched_vessels": len(rows),
                    "displayed_vessels": len(rows),
                },
                {},
            )

    envelope = _service(tmp_path, _RevisionMagnitudeProvider()).query(
        QueryRequest(question=ETA_PROMPTS[5], top_k_evidence=0)
    )

    assert envelope.state == AnswerState.COMPUTED
    dataset = next(item for item in envelope.datasets if item.id == "table")
    assert [row["vessel_label"] for row in dataset.rows] == [
        "LARGE EARLIER",
        "SMALL LATER",
    ]
    chart = next(
        visual
        for visual in envelope.visualizations
        if visual.kind == "cartesian"
    )
    assert chart.sort == "none"
    assert chart.reference_lines[0].value == 0.0
    assert chart.reference_lines[0].label == "No ETA change"


def test_source_not_configured_never_falls_back_to_historical_data(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path, None).query(
        QueryRequest(question=ETA_PROMPTS[0], top_k_evidence=0)
    )

    assert envelope.state == AnswerState.NO_CURRENT_DATA
    assert envelope.availability is not None
    assert envelope.availability.code == "source_unavailable"
    assert envelope.operational_brief is not None
    assert envelope.operational_brief.source_health == "unavailable"
    assert envelope.datasets == []
    assert "AISSTREAM_API_KEY" in envelope.answer
    assert "historical" not in envelope.answer.lower()
    assert "substitut" not in envelope.answer.lower()
