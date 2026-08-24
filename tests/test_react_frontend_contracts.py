from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


STREAMLIT_SOURCE = Path("src/app/streamlit_app.py")
FRONTEND_CATALOG = Path("web/src/data/queryCatalog.json")
FRONTEND_GUIDANCE = Path("web/src/data/queryGuidance.json")
FRONTEND_SAMPLE_MODULE = Path("web/src/data/sampleQueries.ts")
FRONTEND_PAGE_MODULE = Path("web/src/data/analysisPages.ts")
FRONTEND_APP = Path("web/src/App.tsx")
FRONTEND_OVERVIEW = Path("web/src/components/OperationalOverview.tsx")
FRONTEND_OVERVIEW_STYLES = Path(
    "web/src/components/operational-overview.css"
)
FRONTEND_STYLES = Path("web/src/styles.css")
FRONTEND_API = Path("web/src/api.ts")
FRONTEND_TYPES = Path("web/src/types.ts")

EXPECTED_CATALOG_SHA256 = (
    "1df9188bbdfc1a6d411d8a9b56426ea7cafaa6838e9424046c727e7b72282a47"
)
EXPECTED_CATEGORIES = [
    "Traffic Monitoring",
    "Vessel Investigation",
    "ETA & Delay",
    "Port Pressure",
    "Carbon Emissions",
    "Unsupported Scope",
]
EXPECTED_COUNTS = [14, 8, 10, 8, 10, 7]
EXPECTED_ROUTES = [
    "/overview",
    "/analysis",
    "/traffic-monitoring",
    "/vessel-investigation",
    "/eta-delay",
    "/port-pressure",
    "/carbon-emissions",
]


def _literal_assignment(name: str) -> Any:
    module = ast.parse(STREAMLIT_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find literal assignment for {name}")


def _compact_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_react_catalog_is_byte_for_byte_equal_to_the_frozen_python_catalog() -> None:
    python_catalog = _literal_assignment("SAMPLE_QUERIES_BY_CATEGORY")
    react_catalog = json.loads(FRONTEND_CATALOG.read_text(encoding="utf-8"))

    assert react_catalog == python_catalog
    assert list(react_catalog) == EXPECTED_CATEGORIES
    assert [len(react_catalog[category]) for category in react_catalog] == EXPECTED_COUNTS
    assert sum(map(len, react_catalog.values())) == 57
    assert _compact_hash(react_catalog) == EXPECTED_CATALOG_SHA256


def test_react_page_guidance_is_equal_to_the_existing_streamlit_guidance() -> None:
    python_guidance = _literal_assignment("QUERY_CATEGORY_HELP")
    react_guidance = json.loads(FRONTEND_GUIDANCE.read_text(encoding="utf-8"))

    assert react_guidance == python_guidance
    assert list(react_guidance) == EXPECTED_CATEGORIES


def test_sample_module_exposes_the_stable_catalog_interface() -> None:
    source = FRONTEND_SAMPLE_MODULE.read_text(encoding="utf-8")
    for export_name in (
        "SAMPLE_QUERIES_BY_CATEGORY",
        "QUERY_CATEGORY_HELP",
        "QUERY_CATEGORY_PAGE_ORDER",
        "ETA_WATCH_SAMPLE_GROUPS",
        "UNSUPPORTED_SCOPE_PROMPTS",
        "ALL_SAMPLE_PROMPTS",
        "QUERY_CATALOG_SHA256",
    ):
        assert re.search(rf"\bexport const {export_name}\b", source)
    assert EXPECTED_CATALOG_SHA256 in source


def test_frontend_page_registry_preserves_routes_order_and_public_labels() -> None:
    source = FRONTEND_PAGE_MODULE.read_text(encoding="utf-8")
    route_positions = [source.index(f'route: "{route}"') for route in EXPECTED_ROUTES]

    assert route_positions == sorted(route_positions)
    assert 'displayLabel: "Analysis Desk"' in source
    assert 'internalLabel: "Chat Assistant"' in source
    assert 'displayLabel: "ETA Watch"' in source
    assert 'internalLabel: "ETA & Delay"' in source
    assert 'displayLabel: "Carbon Emissions"' in source
    assert 'DEFAULT_WORKSPACE_ROUTE: WorkspaceRoute = "/overview"' in source


def test_baltic_situation_sheet_exposes_historical_atlas_and_accessibility_contracts() -> None:
    app_source = FRONTEND_APP.read_text(encoding="utf-8")
    overview_source = FRONTEND_OVERVIEW.read_text(encoding="utf-8")
    overview_styles = FRONTEND_OVERVIEW_STYLES.read_text(encoding="utf-8")

    for test_id in (
        "operational-overview",
        "overview-enter-analysis",
        "overview-provenance-button",
        "overview-historical-atlas",
        "overview-coverage-disclosure",
    ):
        assert f'data-testid="{test_id}"' in overview_source

    for visible_contract in (
        "Eagle Eye / Baltic situation sheet",
        "Baltic archive footprint",
        "Historical coverage—not traffic volume or current vessel activity",
        "View coverage data",
        "recent work",
        "Start analysis",
        "Data provenance",
        "Row total unavailable",
    ):
        assert visible_contract in overview_source

    assert app_source.count('Navigate to="/overview"') >= 2
    assert 'navigate("/analysis")' in app_source
    assert "overviewRailForcedCompact || settings.railCollapsed" in app_source
    assert "allowCollapse={!overviewRailForcedCompact}" in app_source
    assert "recentRecords={history.slice(0, 6)}" in app_source
    assert "hasCompleteRowCounts ? totalRows : null" in app_source
    assert ': "Not reported"' in app_source
    assert "<Navigation" in app_source
    assert "<UtilityBar" in app_source
    assert 'data-testid="nav-rail"' in app_source
    assert 'data-testid="nav-toggle"' in app_source
    assert 'aria-haspopup="dialog"' in overview_source
    assert 'aria-busy={capabilityState === "loading"}' in overview_source
    assert "deriveHistoricalCountryCoverage" in overview_source
    assert "coverageTier" in overview_source
    for country_code in ("DE", "DK", "EE", "FI", "LT", "LV", "PL", "SE"):
        assert f'"{country_code}"' in overview_source
    assert "<svg" in overview_source
    assert "<title" in overview_source
    assert "<desc" in overview_source
    assert 'className="situation-readiness-ribbon"' in overview_source
    assert 'className="situation-activity-tape"' in overview_source
    assert 'className="situation-atlas-fallback"' in overview_source
    assert "@media (prefers-reduced-motion: reduce)" in overview_styles
    assert "@media (max-width: 699px)" in overview_styles

    combined_overview_source = app_source + overview_source + overview_styles
    for rejected_overview_contract in (
        "<canvas",
        "WebGL",
        "ThreeTheatre",
        "ImmersiveOverview",
        "overview-stage-",
        "data-chapter",
        "overview-theatre",
        "overview-webgl",
        "operational-status-ledger",
        "operational-country-ledger",
        "operational-source-register",
        'data-testid="overview-map"',
        'data-testid="overview-source-spine"',
        'data-testid="overview-command-log"',
        'data-testid="overview-focus-button"',
    ):
        assert rejected_overview_contract not in combined_overview_source


def test_frontend_is_permanently_dark_without_a_public_theme_control() -> None:
    app_source = FRONTEND_APP.read_text(encoding="utf-8")
    styles_source = FRONTEND_STYLES.read_text(encoding="utf-8")

    assert 'document.documentElement.dataset.theme = "dark"' in app_source
    assert 'document.documentElement.style.colorScheme = "dark"' in app_source
    assert 'data-testid="theme-select"' not in app_source
    assert "prefers-color-scheme" not in app_source
    assert "selectedTheme" not in app_source
    assert ':root[data-theme="light"]' not in styles_source


def test_frontend_types_mirror_the_canonical_query_envelope_and_trace() -> None:
    source = FRONTEND_TYPES.read_text(encoding="utf-8")
    required_interfaces = (
        "QueryRequestPayload",
        "QueryPlan",
        "AnswerEnvelope",
        "TraceInfo",
        "CapabilityResponse",
        "ExportRequest",
        "ExportResponse",
    )
    for interface_name in required_interfaces:
        assert f"interface {interface_name}" in source

    assert "plan: QueryPlan;" in source
    assert "operational_brief?: OperationalBrief | null;" in source
    assert "bounding_boxes?:" in source
    for field in (
        "retrieval_mode",
        "retrieval_backend",
        "retrieval_status",
        "retrieval_top_k",
    ):
        assert f"{field}:" in source or f"{field}?:" in source


def test_frontend_api_uses_existing_v2_query_export_feedback_and_capability_routes() -> None:
    source = FRONTEND_API.read_text(encoding="utf-8")

    assert "submitQuery(query: QueryRequestPayload)" in source
    assert '("/query",' in source
    assert '("/capabilities")' in source
    assert '("/exports",' in source
    assert '("/feedback",' in source
    assert "body: JSON.stringify(payload)" in source
