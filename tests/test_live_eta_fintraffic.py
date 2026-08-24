"""Legacy-boundary coverage for the retained Fintraffic adapter.

Fintraffic is no longer a public ETA Watch source. These tests intentionally
cover only its read-only adapter behavior and the boundary that prevents
official-schedule or confirmed-delay requests from reaching any live provider.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.live_eta.fintraffic import (
    FINTRAFFIC_AIS_LOCATIONS_PATH,
    FINTRAFFIC_PORT_CALLS_PATH,
    FintrafficETAAdapter,
    normalize_baltic_port,
    normalize_finnish_port,
)
from src.query.context import ConversationStore
from src.query.models import AnswerState, QueryMode, QueryOperation, QueryRequest
from src.query.planner import QueryPlanner
from src.query.service import QueryService


FIXTURES = Path(__file__).parent / "fixtures" / "fintraffic"
FIXED_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class _Response:
    def __init__(
        self,
        status_code: int,
        payload: Any = None,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}

    def json(self) -> Any:
        return copy.deepcopy(self.payload)


class _FixtureClient:
    def __init__(
        self,
        *,
        portnet: Optional[Dict[str, Any]] = None,
        vessel_collection: Optional[List[Dict[str, Any]]] = None,
        location_collection: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.portnet = portnet if portnet is not None else _json("port_calls.json")
        self.vessel_collection = (
            vessel_collection
            if vessel_collection is not None
            else _json("ais_vessels_baltic.json")
        )
        self.location_collection = (
            location_collection
            if location_collection is not None
            else _json("ais_locations_baltic.json")
        )
        self.metadata = {
            "230123250": _json("ais_vessel_230123250.json"),
            **{
                str(item["mmsi"]): item
                for item in self.vessel_collection
            },
        }
        helsinki_location = _json("ais_locations_230123250.json")
        self.locations = {
            "230123250": helsinki_location,
            **{
                str(feature["mmsi"]): {
                    **self.location_collection,
                    "features": [feature],
                }
                for feature in self.location_collection["features"]
            },
        }
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        if url.endswith(FINTRAFFIC_PORT_CALLS_PATH):
            return _Response(200, self.portnet, headers={"ETag": '"portnet-v1"'})
        if url.endswith("/api/ais/v1/vessels"):
            return _Response(
                200,
                self.vessel_collection,
                headers={"ETag": '"ais-vessels-v1"'},
            )
        if "/api/ais/v1/vessels/" in url:
            mmsi = url.rsplit("/", 1)[-1]
            payload = self.metadata.get(mmsi)
            return _Response(200, payload) if payload is not None else _Response(404)
        if url.endswith(FINTRAFFIC_AIS_LOCATIONS_PATH):
            requested_mmsi = (kwargs.get("params") or {}).get("mmsi")
            if requested_mmsi is None:
                return _Response(
                    200,
                    self.location_collection,
                    headers={"ETag": '"ais-locations-v1"'},
                )
            payload = self.locations.get(str(requested_mmsi))
            return _Response(200, payload) if payload is not None else _Response(404)
        return _Response(404)


class _SequenceClient:
    def __init__(self, responses: List[_Response]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


class _NeverQueriedProvider:
    provider = "aisstream"

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def query(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise AssertionError("unsupported requests must not query a live provider")

    def capabilities(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "available": True,
            "status": "live",
            "maximum_horizon_hours": 48,
        }


def _adapter(client: Any, **kwargs: Any) -> FintrafficETAAdapter:
    return FintrafficETAAdapter(
        http_client=client,
        now_fn=lambda: FIXED_NOW,
        sleep_fn=lambda _: None,
        **kwargs,
    )


def _service(tmp_path: Path, provider: Any) -> QueryService:
    processed = "data/processed"
    return QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=ConversationStore(tmp_path / "legacy-boundary.sqlite3"),
        live_eta=provider,
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Helsinki", "FIHEL"),
        ("Turku", "FITKU"),
        ("Oulu", "FIOUL"),
        ("Pori", "FIPOR"),
        ("FIHEL", "FIHEL"),
    ],
)
def test_finnish_port_aliases_use_portnet_locodes(
    value: str,
    expected: str,
) -> None:
    assert normalize_finnish_port(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Stockholm", "SESTO"),
        ("Karlshamn", "SEKAN"),
        ("Klaipėda", "LTKLJ"),
        ("FIHEL", "FIHEL"),
        ("USNYC", None),
    ],
)
def test_baltic_destination_aliases_are_conservative(
    value: str,
    expected: Optional[str],
) -> None:
    assert normalize_baltic_port(value) == expected


def test_http_retry_headers_etag_and_304_cache_reuse() -> None:
    payload = {"dataUpdatedTime": "2026-07-27T11:58:00Z", "portCalls": []}
    retry_client = _SequenceClient(
        [
            _Response(503),
            _Response(200, payload, headers={"ETag": '"schedule-1"'}),
        ]
    )
    retry_adapter = _adapter(retry_client)

    assert retry_adapter._get_json(FINTRAFFIC_PORT_CALLS_PATH) == payload
    assert len(retry_client.calls) == 2
    assert retry_client.calls[0]["timeout"] == 5.0
    assert retry_client.calls[0]["headers"]["Accept-Encoding"] == "gzip"
    assert retry_client.calls[0]["headers"]["Digitraffic-User"] == "EagleEye/2.0"

    etag_client = _SequenceClient(
        [
            _Response(200, payload, headers={"ETag": '"schedule-2"'}),
            _Response(304),
        ]
    )
    etag_adapter = _adapter(etag_client, cache_ttl_seconds=0)

    assert etag_adapter._get_json(FINTRAFFIC_PORT_CALLS_PATH) == payload
    assert etag_adapter._get_json(FINTRAFFIC_PORT_CALLS_PATH) == payload
    assert etag_client.calls[1]["headers"]["If-None-Match"] == '"schedule-2"'


@pytest.mark.parametrize(
    "identity",
    [
        {"mmsi": "230123250"},
        {"imo": "9543756"},
        {"vessel_name": "ARUNA CIHAN"},
    ],
)
def test_direct_adapter_identity_filters_preserve_validated_overlay(
    identity: Dict[str, str],
) -> None:
    result = _adapter(_FixtureClient()).query(
        operation="vessel_eta",
        ports=["Helsinki"],
        limit=10,
        **identity,
    )

    assert result.status == "ok"
    assert result.source_kind == "portnet_with_ais"
    assert result.table is not None
    assert len(result.table) == 1
    row = result.table.iloc[0]
    assert row["port_locode"] == "FIHEL"
    assert row["vessel_name"] == "ARUNA CIHAN"
    assert row["mmsi"] == "230123250"
    assert row["imo"] == "9543756"
    assert row["announced_delay_minutes"] == 90
    assert row["variance_status"] == "validated"


def test_direct_adapter_accepts_only_exact_regional_destination_tokens() -> None:
    accepted = _adapter(_FixtureClient()).query(
        operation="vessel_eta",
        ports=["Stockholm"],
        limit=10,
    )

    assert accepted.status == "ok"
    assert accepted.source_kind == "ais_destination_only"
    assert accepted.table is not None
    assert len(accepted.table) == 1
    row = accepted.table.iloc[0]
    assert row["port_locode"] == "SESTO"
    assert row["ais_destination"] == "SE STO"
    assert row["ais_destination_match"] == "exact_unlocode_token"
    assert row["official_eta_utc"] is None
    assert row["announced_delay_minutes"] is None

    route_like = _json("ais_vessels_baltic.json")
    stockholm = next(
        item for item in route_like if item["destination"] == "SE STO"
    )
    stockholm["destination"] = "SESTO>EEMUG"
    rejected = _adapter(
        _FixtureClient(vessel_collection=route_like)
    ).query(
        operation="vessel_eta",
        ports=["Stockholm"],
        limit=10,
    )

    assert rejected.status == "no_data"
    assert rejected.table is None


@pytest.mark.parametrize(
    "question",
    [
        "Show the next 10 official arrivals at Helsinki.",
        "Show the official arrival schedule at Stockholm.",
        "What is the confirmed delay for MMSI 230123250 at Helsinki?",
    ],
)
def test_public_official_schedule_and_confirmed_delay_requests_are_unsupported(
    tmp_path: Path,
    question: str,
) -> None:
    provider = _NeverQueriedProvider()
    plan = QueryPlanner().plan(question)

    assert plan.mode == QueryMode.UNSUPPORTED
    assert plan.operation == QueryOperation.UNSUPPORTED
    assert plan.source_scope is None
    assert plan.eta_watch_intent is None

    envelope = _service(tmp_path, provider).query(
        QueryRequest(question=question, top_k_evidence=0)
    )

    assert envelope.state == AnswerState.UNSUPPORTED
    assert provider.calls == []
    assert envelope.datasets == []
    assert [visual.kind for visual in envelope.visualizations] == ["omitted"]
    assert "AISStream does not provide official schedules, confirmed delays" in envelope.answer
    assert "vessel-reported ETA changes" in envelope.answer
    assert "AIS-visible inbound watchlist" in envelope.answer
    assert "last observed vessel position" in envelope.answer
