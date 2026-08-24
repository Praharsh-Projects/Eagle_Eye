from __future__ import annotations

import json
import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.server import app
from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.query import KPIQueryEngine
from src.query.context import ConversationStore
from src.query.service import QueryService


def _client(tmp_path: Path) -> TestClient:
    processed = "data/processed"
    store = ConversationStore(tmp_path / "api.sqlite3")
    service = QueryService(
        kpi=KPIQueryEngine(processed),
        forecaster=ForecastEngine(processed),
        carbon=CarbonQueryEngine(processed, auto_build=False),
        conversation_store=store,
        processed_dir=processed,
        export_dir=tmp_path / "exports",
    )
    app.state.runtime = {
        "query_service": service,
        "conversation_store": store,
    }
    return TestClient(app)


def test_v2_query_stream_feedback_export_and_compatibility(tmp_path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/api/v2/query",
        json={"question": "How many arrivals at Gothenburg in March 2022?", "filters": {}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_version"] == "2.0"
    assert payload["visualization_contract_version"] == "2.1"
    assert payload["state"] == "COMPUTED"
    assert payload["plan"]["operation"] == "arrivals"
    assert payload["visualizations"][0]["kind"] == "kpi"
    assert payload["confidence"] == "high"
    assert payload["assurance"]["status"] == "verified"
    assert payload["assurance"]["level"] == "high"
    assert payload["assurance"]["basis"] == "direct_computation"
    assert payload["availability"]["code"] == "available"
    assert payload["availability"]["provider"] == "structured_datasets"

    stream = client.post(
        "/api/v2/query/stream",
        json={"question": "Plot daily arrivals at Gothenburg for March 2022.", "filters": {}},
    )
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in stream.text
    assert "event: text" in stream.text
    assert "event: final" in stream.text
    stream_events = {
        block.splitlines()[0].removeprefix("event: "): json.loads(
            next(line for line in block.splitlines() if line.startswith("data: ")).removeprefix("data: ")
        )
        for block in stream.text.strip().split("\n\n")
        if block.startswith("event:")
    }
    stream_final = stream_events["final"]
    assert stream_events["text"]["delta"] == stream_final["answer"]
    canonical_stream_prompt = client.post(
        "/api/v2/query",
        json={"question": "Plot daily arrivals at Gothenburg for March 2022.", "filters": {}},
    ).json()
    assert stream_final["trace"]["result_hash"] == canonical_stream_prompt["trace"]["result_hash"]

    feedback = client.post(
        "/api/v2/feedback",
        json={
            "prompt": payload["question"],
            "trace_id": payload["trace"]["trace_id"],
            "note": "Result needs review",
        },
    )
    assert feedback.status_code == 202
    assert feedback.json()["status"] == "accepted"

    exported = client.post(
        "/api/v2/exports",
        json={
            "conversation_id": payload["conversation_id"],
            "turn_id": payload["turn_id"],
            "dataset_id": "table",
            "format": "json",
        },
    )
    assert exported.status_code == 200
    export_path = Path(exported.json()["path"])
    assert export_path.is_file()
    exported_payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported_payload["row_count"] == 31

    legacy = client.post(
        "/ask",
        json={"question": "How many arrivals at Gothenburg in March 2022?", "filters": {}},
    )
    assert legacy.status_code == 200
    assert legacy.json()["result"]["status"] == "ok"
    assert "488 vessel arrivals" in legacy.json()["result"]["answer"]

    chat = client.post(
        "/api/v1/chat",
        json={"message": "How many arrivals at Gothenburg in March 2022?", "filters": {}},
    )
    assert chat.status_code == 200

    def normalized_hash(envelope: dict) -> str:
        stable = {
            key: envelope[key]
            for key in (
                "api_version",
                "visualization_contract_version",
                "mode",
                "state",
                "answer",
                "plan",
                "facts",
                "applied_scope",
                "datasets",
                "visualizations",
                "chart_insights",
                "evidence",
                "freshness",
                "confidence",
                "assurance",
                "availability",
                "caveats",
            )
        }
        return hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    assert normalized_hash(payload) == normalized_hash(legacy.json()["result"]["canonical"])
    assert normalized_hash(payload) == normalized_hash(chat.json()["deterministic_result"]["canonical"])
    assert payload["trace"]["result_hash"] == legacy.json()["result"]["canonical"]["trace"]["result_hash"]
    assert payload["trace"]["result_hash"] == chat.json()["deterministic_result"]["canonical"]["trace"]["result_hash"]

    # SPA fallback must never turn an unknown API route into index.html.
    missing_api = client.get("/api/v2/does-not-exist")
    assert missing_api.status_code == 404
    assert "text/html" not in missing_api.headers.get("content-type", "")


def test_feedback_rejects_blank_or_oversized_public_fields(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.post("/api/v2/feedback", json={"prompt": "   ", "trace_id": "trace_1"}).status_code == 422
    assert client.post("/api/v2/feedback", json={"prompt": "wrong", "trace_id": "   "}).status_code == 422


def test_capabilities_expose_the_live_shaped_manifest_inventory(
    tmp_path: Path,
) -> None:
    payload = _client(tmp_path).get("/api/v2/capabilities").json()
    tables = payload["data_manifest"]["tables"]

    assert len(tables) == 9
    assert sum(int(summary["rows"]) for summary in tables.values()) == 2_879_206
    assert all(summary["readable"] is True for summary in tables.values())
    assert "operations" in payload
    live_eta = payload["live_eta"]
    assert live_eta["provider"] == "aisstream"
    assert live_eta["available"] is False
    assert live_eta["maximum_horizon_hours"] == 48
    assert live_eta["official_schedule_country_scope"] == []
    assert live_eta["official_eta_authority"] is None
    assert live_eta["prediction"] is False
    assert "vessel-reported" in live_eta["regional_ais_scope"]
    assert "not a complete or official arrival board" in live_eta["regional_ais_scope"]


def test_zero_evidence_top_k_is_accepted_by_v2_and_compatibility_routes(tmp_path) -> None:
    client = _client(tmp_path)
    payload = {
        "question": "How many arrivals at Gothenburg in March 2022?",
        "top_k_evidence": 0,
        "filters": {},
    }

    assert client.post("/api/v2/query", json=payload).status_code == 200
    assert client.post("/ask", json=payload).status_code == 200
    assert client.post(
        "/api/v1/chat",
        json={"message": payload["question"], "top_k_evidence": 0, "filters": {}},
    ).status_code == 200
