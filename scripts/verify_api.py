#!/usr/bin/env python3
"""Deterministic API acceptance checks without retrieval or model network calls."""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)
from fastapi.testclient import TestClient

from src.api.server import _build_state, app


def _assert_status(response, expected: int, label: str) -> None:
    if response.status_code != expected:
        raise AssertionError(
            f"{label}: expected HTTP {expected}, got {response.status_code}: {response.text[:500]}"
        )


def main() -> int:
    saved_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        app.state.runtime = _build_state()
    finally:
        if saved_key is not None:
            os.environ["OPENAI_API_KEY"] = saved_key

    client = TestClient(app)

    health = client.get("/health")
    _assert_status(health, 200, "health")
    assert health.json()["status"] == "ok"

    paper = client.post(
        "/ask",
        json={
            "question": "How many vessel arrivals were recorded at Gothenburg in March 2022?",
            "top_k_evidence": 1,
            "filters": {},
        },
    )
    _assert_status(paper, 200, "paper arrivals")
    assert "488 vessel arrivals" in paper.json()["result"]["answer"]

    unknown = client.post(
        "/ask",
        json={
            "question": "Show WTW CO2e emissions at UNKNOWNPORT between 2022-02-01 and 2022-02-28.",
            "top_k_evidence": 1,
            "filters": {},
        },
    )
    _assert_status(unknown, 200, "unknown port")
    unknown_result = unknown.json()["result"]
    assert unknown_result["status"] == "no_data"
    assert "canonical port catalog" in " ".join(unknown_result["caveats"])

    carbon = client.get(
        "/api/v1/carbon/ports/SEGOT/emissions",
        params={
            "from": "2022-03-01",
            "to": "2022-03-31",
            "boundary": "TTW",
            "pollutants": "CO2e",
        },
    )
    _assert_status(carbon, 200, "core carbon")
    assert carbon.json()["result"]["carbon"]["result_state"] in {"COMPUTED", "COMPUTED_ZERO"}

    retired = [
        ("POST", "/api/v1/voyages/resolve"),
        ("GET", "/api/v1/voyages/example"),
        ("GET", "/api/v1/voyages/example/segments"),
        ("GET", "/api/v1/emissions/voyages/example"),
        ("GET", "/api/v1/audit/runs/example"),
        ("GET", "/api/v1/regulatory/zones/by-point"),
    ]
    for method, path in retired:
        response = client.request(method, path)
        _assert_status(response, 404, f"retired route {path}")

    print("API checks passed: health, paper result, unknown-port guard, core carbon, retired routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
