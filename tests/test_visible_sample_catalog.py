from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd

from src.app.streamlit_app import SAMPLE_QUERIES_BY_CATEGORY
from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.models import AnswerState, ETAWatchIntent, QueryOperation, QueryRequest
from src.query.service import QueryService


GRAPH_VISUAL_KINDS = {"cartesian", "forecast", "distribution", "heatmap", "map", "timeline"}
PROHIBITED_PUBLIC_METHOD_LABELS = re.compile(
    r"\b(?:assurance|confidence|caveat|proxy|heuristic|reconstruct(?:ed|ion)?|"
    r"estimated|fallback|publication[- ]gate|partial result)\b",
    re.IGNORECASE,
)
EXPECTED_EMPTY_RESULT_OMISSIONS = {
    "For MMSI 212575000, summarize suspicious AIS jumps on 2021-01-01.",
    "List any AIS jump anomalies for MMSI 266232000 between 2021-01-01 and 2021-01-03.",
    "Show movement anomalies for MMSI 246650000 in March 2022.",
    "How many anomaly events were detected for MMSI 255806245 in 2022-03?",
}
FIXED_LIVE_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
ETA_WATCH_PROMPTS = [
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


def _aisstream_rows() -> pd.DataFrame:
    """Fresh, source-shaped AIS observations for catalog tests—never web calls."""

    rows = [
        ("265000001", "NORDIC STAR", "SESTO", "Stockholm", 14, 0, 59.10, 18.10, 8.2),
        ("265000002", "BALTIC ONE", "SEGOT", "Gothenburg", 15, 0, 57.60, 11.80, 1.2),
        ("265000003", "NYNASHAMN LINK", "SENYN", "Nynäshamn", 16, 0, 58.80, 18.00, 6.4),
        ("265000004", "MALMO SOUND", "SEMMA", "Malmö", 17, 0, 55.60, 12.80, 7.1),
        ("265000005", "TRELLEBORG STAR", "SETRG", "Trelleborg", 18, 0, 55.30, 12.90, 9.0),
        ("230000006", "HELSINKI LINK", "FIHEL", "Helsinki", 13, 30, 59.90, 24.80, 8.0),
        ("276000007", "TALLINN LINK", "EETLL", "Tallinn", 14, 30, 59.50, 24.60, 8.4),
        ("275000008", "RIGA LINK", "LVRIX", "Riga", 15, 30, 57.00, 23.80, 7.8),
        ("277000009", "KLAIPEDA LINK", "LTKLJ", "Klaipėda", 16, 30, 56.00, 20.80, 8.1),
        ("261000010", "GDANSK LINK", "PLGDN", "Gdańsk", 17, 30, 55.20, 18.70, 9.2),
        ("230000011", "TURKU LINK", "FITKU", "Turku", 18, 30, 59.70, 21.20, 7.4),
    ]
    return pd.DataFrame(
        [
            {
                "mmsi": mmsi,
                "vessel_name": vessel_name,
                "destination_locode": locode,
                "destination_name": destination,
                "destination_raw": destination.upper(),
                "destination_verified": True,
                "eta_utc": datetime(
                    2026,
                    7,
                    27,
                    hour,
                    minute,
                    tzinfo=timezone.utc,
                ).isoformat(),
                "eta_valid": True,
                "static_observed_at_utc": "2026-07-27T11:57:00+00:00",
                "latitude": latitude,
                "longitude": longitude,
                "sog_kn": speed,
                "position_observed_at_utc": "2026-07-27T11:58:00+00:00",
                "position_age_minutes": 2.0,
                "position_stale": False,
            }
            for (
                mmsi,
                vessel_name,
                locode,
                destination,
                hour,
                minute,
                latitude,
                longitude,
                speed,
            ) in rows
        ]
    )


@dataclass
class _AISStreamFixtureResult:
    status: str
    operation: str
    table: pd.DataFrame | None
    summary: dict[str, Any]
    sections: dict[str, pd.DataFrame]
    horizon_end: datetime
    snapshot_at: datetime = FIXED_LIVE_NOW
    data_updated_at: datetime = datetime(
        2026,
        7,
        27,
        11,
        59,
        tzinfo=timezone.utc,
    )
    answer: str = "Provider prose must never replace deterministic ETA Watch prose."
    coverage_notes: list[str] | None = None
    caveats: list[str] | None = None
    failure_reason: str | None = None
    source_kind: str = "aisstream_observation"

    def __post_init__(self) -> None:
        self.coverage_notes = list(self.coverage_notes or [])
        self.caveats = list(self.caveats or [])

    @property
    def chart(self) -> pd.DataFrame | None:
        return self.table


class _AISStreamFixtureProvider:
    """Deterministic implementation of QueryService's LiveETAProvider protocol."""

    provider = "aisstream"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def source_health(self) -> SimpleNamespace:
        return SimpleNamespace(status="live")

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": True,
            "state": "live",
        }

    def query(self, operation: str, **kwargs: Any) -> _AISStreamFixtureResult:
        self.calls.append({"operation": operation, **kwargs})
        horizon_hours = int(kwargs.get("horizon_hours") or 24)
        horizon_end = FIXED_LIVE_NOW + timedelta(hours=horizon_hours)
        rows = _aisstream_rows()
        ports = {str(port) for port in kwargs.get("ports") or []}
        if ports:
            rows = rows[rows["destination_locode"].isin(ports)].reset_index(drop=True)

        sections: dict[str, pd.DataFrame] = {}
        if operation == "low_speed":
            rows = rows[rows["sog_kn"] < float(kwargs.get("speed_threshold_kn") or 2)]
        elif operation == "eta_revisions":
            rows = pd.DataFrame(
                [
                    {
                        "mmsi": "265000001",
                        "vessel_name": "NORDIC STAR",
                        "destination_locode": "SESTO",
                        "destination_name": "Stockholm",
                        "previous_eta_utc": "2026-07-27T13:10:00+00:00",
                        "current_eta_utc": "2026-07-27T14:00:00+00:00",
                        "eta_revision_minutes": 50.0,
                        "revision_observed_at_utc": "2026-07-27T11:50:00+00:00",
                    }
                ]
            )
        elif operation == "stale_missing":
            rows = pd.DataFrame(
                [
                    {
                        "mmsi": "265000012",
                        "vessel_name": "WATCH REQUIRED",
                        "destination_locode": "SENYN",
                        "destination_name": "Nynäshamn",
                        "eta_utc": None,
                        "eta_valid": False,
                        "latitude": None,
                        "longitude": None,
                        "sog_kn": None,
                        "position_observed_at_utc": None,
                        "position_age_minutes": None,
                        "position_stale": True,
                        "validation_reasons": ["missing_position", "missing_eta"],
                    }
                ]
            )
        elif operation == "shift_handover":
            inbound = rows[rows["destination_locode"].str.startswith("SE")].reset_index(
                drop=True
            )
            low_speed = inbound[inbound["sog_kn"] < 2].reset_index(drop=True)
            revisions = pd.DataFrame(
                [
                    {
                        "mmsi": "265000001",
                        "vessel_name": "NORDIC STAR",
                        "destination_locode": "SESTO",
                        "destination_name": "Stockholm",
                        "previous_eta_utc": "2026-07-27T13:10:00+00:00",
                        "current_eta_utc": "2026-07-27T14:00:00+00:00",
                        "eta_revision_minutes": 50.0,
                        "revision_observed_at_utc": "2026-07-27T11:50:00+00:00",
                    }
                ]
            )
            stale = pd.DataFrame(
                [
                    {
                        "mmsi": "265000012",
                        "vessel_name": "WATCH REQUIRED",
                        "destination_locode": "SENYN",
                        "destination_name": "Nynäshamn",
                        "eta_utc": None,
                        "eta_valid": False,
                        "position_stale": True,
                    }
                ]
            )
            rows = inbound
            sections = {
                "inbound_watchlist": inbound,
                "low_speed": low_speed,
                "eta_revisions": revisions,
                "stale_missing": stale,
            }

        rows = rows.reset_index(drop=True)
        matched = len(rows)
        summary = {
            "matched_count": matched,
            "matched_vessels": matched,
            "displayed_vessels": matched,
            "source_health": "live",
        }
        return _AISStreamFixtureResult(
            status="ok",
            operation=operation,
            table=rows,
            summary=summary,
            sections=sections,
            horizon_end=horizon_end,
        )


def _service(
    tmp_path: Path,
    *,
    live_eta: Optional[_AISStreamFixtureProvider] = None,
    database_name: str = "visible_samples.sqlite3",
) -> QueryService:
    processed = Path("data/processed")
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / database_name),
        events_path=processed / "events.parquet",
        live_eta=live_eta,
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


def test_every_advertised_non_refusal_sample_returns_a_valid_result(tmp_path: Path) -> None:
    """The UI must never advertise a sample that its canonical backend rejects."""

    historical_service = _service(tmp_path)
    live_service = _service(
        tmp_path,
        live_eta=_AISStreamFixtureProvider(),
        database_name="visible_live_samples.sqlite3",
    )
    failures: list[str] = []
    graph_failures: list[str] = []
    omitted_prompts: set[str] = set()
    published_historical = 0
    no_data_historical = 0
    for category, prompts in SAMPLE_QUERIES_BY_CATEGORY.items():
        if category == "Unsupported Scope":
            continue
        service = (
            live_service
            if category == "ETA & Delay"
            else historical_service
        )
        for index, prompt in enumerate(prompts):
            envelope = service.query(
                QueryRequest(
                    question=prompt,
                    conversation_id=f"visible-sample-{category}-{index}",
                    top_k_evidence=0,
                )
            )
            if envelope.state not in {
                AnswerState.COMPUTED,
                AnswerState.PARTIAL,
                AnswerState.NO_DATA,
            }:
                failures.append(
                    f"{category}: {prompt!r} -> {envelope.state.value}/"
                    f"{envelope.plan.operation.value}: {envelope.answer}"
                )
                continue
            if category != "ETA & Delay":
                if envelope.state in {AnswerState.COMPUTED, AnswerState.PARTIAL}:
                    published_historical += 1
                elif envelope.state == AnswerState.NO_DATA:
                    no_data_historical += 1
            if envelope.state in {AnswerState.COMPUTED, AnswerState.PARTIAL}:
                assert envelope.confidence == "high", prompt
                assert envelope.assurance is not None, prompt
                assert envelope.assurance.status == "verified", prompt
                assert envelope.assurance.level == "high", prompt
                assert envelope.availability is not None, prompt
                assert envelope.availability.code == "available", prompt
                assert envelope.datasets, prompt
                assert envelope.caveats == [], prompt
                public_parts = [envelope.answer]
                public_parts.extend(insight.statement for insight in envelope.chart_insights)
                for visual in envelope.visualizations:
                    public_parts.extend(
                        str(value)
                        for value in (
                            getattr(visual, "title", None),
                            getattr(visual, "accessible_summary", None),
                        )
                        if value
                    )
                for dataset in envelope.datasets:
                    for column in dataset.columns:
                        public_parts.extend((column.field, column.label))
                assert PROHIBITED_PUBLIC_METHOD_LABELS.search(
                    "\n".join(str(value) for value in public_parts if value)
                ) is None, prompt
            if not envelope.visualizations:
                failures.append(f"{category}: {prompt!r} returned no visualization contract")
                continue

            kinds = [visual.kind for visual in envelope.visualizations]
            has_data = any(dataset.row_count > 0 for dataset in envelope.datasets)
            has_graph = any(kind in GRAPH_VISUAL_KINDS for kind in kinds)
            if category == "ETA & Delay" and kinds == ["table"]:
                # Signal-quality triage is intentionally a status table rather
                # than a decorative graph.
                has_graph = True
            if has_data and not has_graph:
                graph_failures.append(
                    f"{category}: {prompt!r} -> {envelope.plan.operation.value}/{kinds}"
                )
            if "omitted" in kinds:
                omitted_prompts.add(prompt)

    assert not failures, "\n" + "\n".join(failures)
    assert not graph_failures, "\n" + "\n".join(graph_failures)
    assert EXPECTED_EMPTY_RESULT_OMISSIONS.issubset(omitted_prompts)
    assert published_historical == 34
    assert no_data_historical == 6


def test_unsupported_visible_samples_remain_graph_free(tmp_path: Path) -> None:
    service = _service(tmp_path)
    for index, prompt in enumerate(SAMPLE_QUERIES_BY_CATEGORY["Unsupported Scope"]):
        envelope = service.query(
            QueryRequest(
                question=prompt,
                conversation_id=f"visible-unsupported-{index}",
                top_k_evidence=0,
            )
        )
        assert envelope.state == AnswerState.UNSUPPORTED, prompt
        assert envelope.datasets == [], prompt
        assert [visual.kind for visual in envelope.visualizations] == ["omitted"], prompt


def test_eta_watch_replaces_exactly_ten_prompts_without_catalog_drift() -> None:
    assert SAMPLE_QUERIES_BY_CATEGORY["ETA & Delay"] == ETA_WATCH_PROMPTS
    assert sum(len(prompts) for prompts in SAMPLE_QUERIES_BY_CATEGORY.values()) == 57
    assert (
        sum(
            len(prompts)
            for category, prompts in SAMPLE_QUERIES_BY_CATEGORY.items()
            if category != "ETA & Delay"
        )
        == 47
    )


def test_eta_samples_are_aisstream_gated_and_graphable(tmp_path: Path) -> None:
    provider = _AISStreamFixtureProvider()
    service = _service(tmp_path, live_eta=provider)
    expected_intents = [
        ETAWatchIntent.SHIFT_HANDOVER,
        ETAWatchIntent.INBOUND_WATCHLIST,
        ETAWatchIntent.INBOUND_WATCHLIST,
        ETAWatchIntent.LOW_SPEED_EXCEPTIONS,
        ETAWatchIntent.DESTINATION_LOAD,
        ETAWatchIntent.ETA_REVISIONS,
        ETAWatchIntent.SIGNAL_QUALITY,
        ETAWatchIntent.VESSEL_STATUS,
        ETAWatchIntent.INBOUND_WATCHLIST,
        ETAWatchIntent.INBOUND_WATCHLIST,
    ]
    expected_visuals = [
        {"map", "timeline", "table"},
        {"map", "timeline"},
        {"timeline"},
        {"map", "cartesian", "table"},
        {"cartesian"},
        {"cartesian"},
        {"table"},
        {"map", "timeline"},
        {"timeline"},
        {"map", "timeline"},
    ]
    for index, prompt in enumerate(ETA_WATCH_PROMPTS):
        envelope = service.query(
            QueryRequest(
                question=prompt,
                conversation_id=f"visible-eta-{index}",
                top_k_evidence=0,
            )
        )
        assert envelope.state == AnswerState.COMPUTED, prompt
        assert envelope.plan.eta_watch_intent == expected_intents[index], prompt
        assert envelope.plan.source_scope == "aisstream", prompt
        assert envelope.plan.operation not in {
            QueryOperation.FORECAST_ARRIVALS,
            QueryOperation.FORECAST_CONGESTION,
            QueryOperation.FORECAST_COMPARISON,
        }, prompt
        assert envelope.freshness.historical is False, prompt
        assert envelope.freshness.as_of == "2026-07-27T11:59:00Z", prompt
        assert envelope.confidence == "high", prompt
        assert envelope.assurance is not None, prompt
        assert envelope.assurance.status == "verified", prompt
        assert envelope.assurance.level == "high", prompt
        assert envelope.assurance.basis == "direct_computation", prompt
        assert envelope.availability is not None, prompt
        assert envelope.availability.code == "available", prompt
        assert envelope.availability.provider == "aisstream", prompt
        assert envelope.operational_brief is not None, prompt
        assert envelope.operational_brief.intent == expected_intents[index], prompt
        assert envelope.operational_brief.source_health == "live", prompt
        assert envelope.datasets, prompt
        assert all(visual.kind != "omitted" for visual in envelope.visualizations), prompt
        assert set(visual.kind for visual in envelope.visualizations) == expected_visuals[
            index
        ], prompt
        assert "2022" not in envelope.answer, prompt
        assert "Provider prose" not in envelope.answer, prompt
        assert "official arrival" not in envelope.answer.lower(), prompt
        assert "confirmed delay" in envelope.answer.lower(), prompt
        assert all(
            item.url == "https://aisstream.io/documentation.html"
            for item in envelope.evidence
        ), prompt
    assert len(provider.calls) == 10


def test_eta_catalog_preserves_requested_fields_nulls_and_semantic_visuals(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, live_eta=_AISStreamFixtureProvider())

    inbound = service.query(QueryRequest(question=ETA_WATCH_PROMPTS[1], top_k_evidence=0))
    inbound_table = next(item for item in inbound.datasets if item.id == "table")
    first = inbound_table.rows[0]
    for field in (
        "vessel_name",
        "destination_name",
        "reported_eta_utc",
        "latitude",
        "longitude",
        "speed_kn",
        "observation_time_utc",
    ):
        assert first[field] is not None, field
    assert {visual.kind for visual in inbound.visualizations} == {"map", "timeline"}
    assert next(
        visual for visual in inbound.visualizations if visual.kind == "map"
    ).geometry_mode == "points"

    revisions = service.query(
        QueryRequest(question=ETA_WATCH_PROMPTS[5], top_k_evidence=0)
    )
    revisions_table = next(item for item in revisions.datasets if item.id == "table")
    assert revisions_table.rows[0]["eta_change_minutes"] == 50.0
    assert {visual.kind for visual in revisions.visualizations} == {"cartesian"}

    signal_quality = service.query(
        QueryRequest(question=ETA_WATCH_PROMPTS[6], top_k_evidence=0)
    )
    quality_table = next(
        item for item in signal_quality.datasets if item.id == "table"
    )
    assert quality_table.rows
    assert quality_table.rows[0]["reported_eta_utc"] is None
    assert quality_table.rows[0]["latitude"] is None
    assert quality_table.rows[0]["longitude"] is None
    assert {visual.kind for visual in signal_quality.visualizations} == {"table"}

    destination_load = service.query(
        QueryRequest(question=ETA_WATCH_PROMPTS[4], top_k_evidence=0)
    )
    load_table = next(
        item for item in destination_load.datasets if item.id == "table"
    )
    assert load_table.rows
    assert all(row["inbound_vessels"] > 0 for row in load_table.rows)
    assert {visual.kind for visual in destination_load.visualizations} == {
        "cartesian"
    }


def test_eta_watch_without_provider_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    envelope = _service(tmp_path).query(
        QueryRequest(
            question=SAMPLE_QUERIES_BY_CATEGORY["ETA & Delay"][0],
            top_k_evidence=0,
        )
    )
    assert envelope.state == AnswerState.NO_CURRENT_DATA
    assert envelope.confidence == "not_applicable"
    assert envelope.assurance is not None
    assert envelope.assurance.status == "unavailable"
    assert envelope.assurance.level == "not_applicable"
    assert envelope.availability is not None
    assert envelope.availability.code == "source_unavailable"
    assert envelope.datasets == []
    assert envelope.visualizations[0].kind == "omitted"
    assert envelope.visualizations[0].reason_code == "stale_data"
