from __future__ import annotations

from src.api.server import app


def test_voyage_lab_routes_are_not_registered() -> None:
    registered = {route.path for route in app.routes}
    retired = {
        "/api/v1/voyages/resolve",
        "/api/v1/voyages/{voyage_id}",
        "/api/v1/voyages/{voyage_id}/segments",
        "/api/v1/emissions/voyages/{voyage_id}",
        "/api/v1/audit/runs/{run_id}",
        "/api/v1/regulatory/zones/by-point",
    }
    assert registered.isdisjoint(retired)


def test_core_carbon_routes_remain_registered() -> None:
    registered = {route.path for route in app.routes}
    assert "/api/v1/carbon/ports/{port_id}/emissions" in registered
    assert "/api/v1/carbon/vessels/{mmsi}/calls/{call_id}" in registered
