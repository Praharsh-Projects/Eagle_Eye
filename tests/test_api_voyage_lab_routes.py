from __future__ import annotations

from src.api import server


class _StubVoyageLab:
    def resolve_voyage(self, payload):
        return {"status": "ok", "resolved_voyage_id": "voy_123", "request": payload}

    def get_voyage(self, voyage_id: str):
        return {"status": "ok", "voyage": {"voyage_id": voyage_id}}

    def get_segments(self, voyage_id: str):
        return {"status": "ok", "voyage_id": voyage_id, "segment_count": 2}

    def get_emissions(self, voyage_id: str, boundary: str = "TTW", pollutants=None):
        return {
            "status": "ok",
            "voyage_id": voyage_id,
            "boundary": boundary,
            "pollutants": pollutants or [],
            "result_state": "COMPUTED",
        }

    def get_audit_run(self, run_id: str):
        return {"status": "ok", "run_id": run_id, "manifest": {"run_id": run_id}}

    def lookup_regulatory_zones(self, *, latitude: float, longitude: float, max_results: int = 25):
        return {
            "status": "ok",
            "latitude": float(latitude),
            "longitude": float(longitude),
            "match_count": 1,
            "zones": [{"zone_id": "z1", "zone_name": "Test Zone"}][:max_results],
        }


def test_voyage_resolve_route(monkeypatch) -> None:
    monkeypatch.setattr(server, "_require_voyage_lab", lambda: _StubVoyageLab())
    payload = server.VoyageResolveRequestPayload(mmsi="123456789")
    out = server.voyage_resolve(payload)
    assert out["status"] == "ok"
    assert out["resolved_voyage_id"] == "voy_123"


def test_voyage_emissions_route(monkeypatch) -> None:
    monkeypatch.setattr(server, "_require_voyage_lab", lambda: _StubVoyageLab())
    out = server.voyage_emissions(voyage_id="voy_123", boundary="WTW", pollutants="CO2e,NOx")
    assert out["status"] == "ok"
    assert out["boundary"] == "WTW"
    assert out["pollutants"] == ["CO2e", "NOx"]


def test_audit_run_route(monkeypatch) -> None:
    monkeypatch.setattr(server, "_require_voyage_lab", lambda: _StubVoyageLab())
    out = server.audit_run("run_1")
    assert out["status"] == "ok"
    assert out["run_id"] == "run_1"


def test_regulatory_by_point_route(monkeypatch) -> None:
    monkeypatch.setattr(server, "_require_voyage_lab", lambda: _StubVoyageLab())
    out = server.regulatory_by_point(lat=56.16, lon=15.58, max_results=5)
    assert out["status"] == "ok"
    assert out["match_count"] == 1
    assert out["zones"][0]["zone_id"] == "z1"
