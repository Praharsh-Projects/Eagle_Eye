import json
from pathlib import Path

from src.review.ui_audit import _build_summary_markdown, _write_artifacts


def test_review_artifacts_contract(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "run_id": "20260403T120000Z",
        "timestamp_utc": "2026-04-03T12:00:00Z",
        "git_sha": "abc123",
        "base_url": "http://127.0.0.1:8501",
        "api_base_url": "http://127.0.0.1:8000",
        "overall_status": "pass",
        "totals": {
            "scenarios": 2,
            "passed": 2,
            "failed": 0,
            "api_passed": 2,
            "api_failed": 0,
        },
        "scenarios": [
            {
                "id": "traffic_descriptive",
                "category": "traffic_descriptive",
                "status": "pass",
                "error_code": "",
                "extracted": {"carbon_result_state": ""},
                "screenshots": ["screenshots/traffic_descriptive.png"],
            }
        ],
        "api_checks": [
            {"name": "health", "status": "pass", "http_code": 200, "latency_ms": 10.5, "message": "ok"}
        ],
    }

    _write_artifacts(tmp_path, payload)
    assert (tmp_path / "review_index.json").exists()
    assert (tmp_path / "review_summary.md").exists()

    parsed = json.loads((tmp_path / "review_index.json").read_text(encoding="utf-8"))
    assert parsed["run_id"] == "20260403T120000Z"
    assert parsed["overall_status"] == "pass"


def test_review_summary_contains_key_sections() -> None:
    payload = {
        "run_id": "20260403T120000Z",
        "timestamp_utc": "2026-04-03T12:00:00Z",
        "git_sha": "abc123",
        "base_url": "http://127.0.0.1:8501",
        "api_base_url": "http://127.0.0.1:8000",
        "overall_status": "fail",
        "totals": {"scenarios": 1, "passed": 0, "failed": 1, "api_passed": 0, "api_failed": 1},
        "scenarios": [
            {
                "id": "unsupported_scope",
                "category": "unsupported",
                "status": "fail",
                "error_code": "validation_failed",
                "extracted": {"carbon_result_state": ""},
                "screenshots": ["screenshots/unsupported_scope.png"],
            }
        ],
        "api_checks": [
            {"name": "health", "status": "fail", "http_code": 500, "latency_ms": 120.0, "message": "boom"}
        ],
    }
    md = _build_summary_markdown(payload)
    assert "## Scenario Results" in md
    assert "## API Sanity" in md
    assert "unsupported_scope" in md
    assert "validation_failed" in md

