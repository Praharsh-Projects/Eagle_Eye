from __future__ import annotations

import re
from pathlib import Path


APP = Path("web/src/App.tsx")
STYLES = Path("web/src/styles.css")
OVERVIEW = Path("web/src/components/OperationalOverview.tsx")
OVERVIEW_STYLES = Path("web/src/components/operational-overview.css")
INDEX = Path("web/index.html")
DEFAULT_LAUNCHER = Path("run_eagle_eye.sh")
QA_LAUNCHER = Path("run_streamlit.sh")


def test_frontend_rejects_the_frozen_ai_dashboard_patterns() -> None:
    source = "\n".join(
        (
            APP.read_text(encoding="utf-8"),
            STYLES.read_text(encoding="utf-8"),
            OVERVIEW.read_text(encoding="utf-8"),
            OVERVIEW_STYLES.read_text(encoding="utf-8"),
            INDEX.read_text(encoding="utf-8"),
        )
    )
    forbidden = {
        "Inter font": r"font-family\s*:\s*[\"']?Inter\b",
        "decorative linear gradient": r"linear-gradient\s*\(",
        "decorative radial gradient": r"radial-gradient\s*\(",
        "decorative conic gradient": r"conic-gradient\s*\(",
        "glass blur": r"backdrop-filter\s*:",
        "pill geometry": r"border-radius\s*:\s*(?:999(?:px)?|100%)",
        "decorative animation": r"@keyframes\b",
        "radar decoration": r"\bradar\b",
        "generic assistant prompt": r"How can I help",
        "rejected marketing claim": r"Turn maritime data into a decision",
        "starter-card shell": r"\b(?:starter-grid|welcome-panel)\b",
    }

    for label, pattern in forbidden.items():
        assert re.search(pattern, source, flags=re.IGNORECASE) is None, label


def test_public_shell_uses_exact_routes_labels_and_accessibility_hooks() -> None:
    source = APP.read_text(encoding="utf-8")
    for route in (
        "/overview",
        "/analysis",
        "/traffic-monitoring",
        "/vessel-investigation",
        "/eta-delay",
        "/port-pressure",
        "/carbon-emissions",
    ):
        assert f'path: "{route}"' in source

    assert 'label: "Analysis Desk"' in source
    assert 'internalId: "Chat Assistant"' in source
    assert source.count("Chat Assistant") == 1
    assert 'label: "Carbon Emissions"' in source
    assert 'aria-live="polite"' in source
    assert 'id="main-content"' in source
    assert 'href="#main-content"' in INDEX.read_text(encoding="utf-8")


def test_dark_only_theme_and_mobile_content_order_are_contractual() -> None:
    source = STYLES.read_text(encoding="utf-8")
    for token in (
        "--canvas: #071014",
        "--surface: #101c21",
        "--raised: #142228",
        "--border: #26363d",
        "--text: #e7ece9",
        "--action: #c9a85a",
        "--data-accent: #4bae9e",
    ):
        assert token in source

    assert ':root[data-theme="light"]' not in source
    assert "color-scheme: light" not in source
    assert "@media (max-width: 699px)" in source
    assert ".answer-record" in source and "order: 1" in source
    assert ".visualization-stack" in source and "order: 2" in source
    assert ".result-metadata" in source and "order: 3" in source
    assert ".evidence-inspector" in source and "order: 4" in source
    assert "@media (prefers-reduced-motion: reduce)" in source
    assert "min-height: 44px" in source


def test_overview_uses_flat_two_dimensional_baltic_situation_sheet() -> None:
    source = OVERVIEW.read_text(encoding="utf-8")
    styles = OVERVIEW_STYLES.read_text(encoding="utf-8")
    combined = source + styles

    assert 'data-testid="operational-overview"' in source
    assert 'data-testid="overview-historical-atlas"' in source
    assert 'className="situation-readiness-ribbon"' in source
    assert 'className="situation-activity-tape"' in source
    assert "Eagle Eye / Baltic situation sheet" in source
    assert "Baltic archive footprint" in source
    assert "Historical coverage—not traffic volume or current vessel activity" in source
    assert "View coverage data" in source
    assert "recent work" in source
    assert "@media (max-width: 699px)" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles

    for rejected_pattern in (
        r"<canvas",
        r"\bWebGL\b",
        r"\bThree\b",
        r"\bchapter\b",
        r"\baperture\b",
        r"\bcard\b",
        r"operational-status-ledger",
        r"operational-country-ledger",
        r"operational-source-register",
        r"@keyframes\b",
        r"box-shadow\s*:",
        r"(?:linear|radial|conic)-gradient\s*\(",
    ):
        assert re.search(rejected_pattern, combined, re.IGNORECASE) is None


def test_default_and_qa_launchers_remain_separate() -> None:
    default = DEFAULT_LAUNCHER.read_text(encoding="utf-8")
    qa = QA_LAUNCHER.read_text(encoding="utf-8")

    assert 'exec "$EAGLE_EYE_ROOT/run_fastapi.sh"' in default
    assert '"$VENV_DIR/bin/streamlit" run app/streamlit_app.py' in qa
    assert 'PORT="${PORT:-8501}"' in qa
