from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


STREAMLIT_SOURCE_PATH = Path("src/app/streamlit_app.py")

EXPECTED_INTERNAL_PAGES = [
    "Overview",
    "Chat Assistant",
    "Traffic Monitoring",
    "Vessel Investigation",
    "ETA & Delay",
    "Port Pressure",
    "Carbon Emissions",
]

EXPECTED_SAMPLE_CATEGORIES = [
    "Traffic Monitoring",
    "Vessel Investigation",
    "ETA & Delay",
    "Port Pressure",
    "Carbon Emissions",
    "Unsupported Scope",
]

EXPECTED_SAMPLE_COUNTS = {
    "Traffic Monitoring": 14,
    "Vessel Investigation": 8,
    "ETA & Delay": 10,
    "Port Pressure": 8,
    "Carbon Emissions": 10,
    "Unsupported Scope": 7,
}

# Canonical JSON uses insertion order, so this freezes every prompt plus its
# category and position without maintaining a second, easy-to-drift copy.
EXPECTED_SAMPLE_CATALOG_SHA256 = (
    "1df9188bbdfc1a6d411d8a9b56426ea7cafaa6838e9424046c727e7b72282a47"
)

EXPECTED_GRAPH_KINDS = {
    "cartesian",
    "forecast",
    "distribution",
    "heatmap",
    "map",
    "timeline",
}


def _source_and_module() -> tuple[str, ast.Module]:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _function_source(source: str, module: ast.Module, name: str) -> str:
    rendered = ast.get_source_segment(source, _function(module, name))
    assert rendered is not None
    return rendered


def _literal_assignment(module: ast.Module, name: str) -> Any:
    for node in ast.walk(module):
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            value = node.value
        if value is not None:
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "frozenset"
                and len(value.args) == 1
            ):
                return frozenset(ast.literal_eval(value.args[0]))
            return ast.literal_eval(value)
    raise AssertionError(f"Could not find literal assignment for {name}")


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _call_name(node) == name
    ]


def test_internal_routes_keep_the_previous_identifiers_and_use_one_display_alias() -> None:
    source, module = _source_and_module()

    assert _literal_assignment(module, "analyst_pages") == EXPECTED_INTERNAL_PAGES
    assert "Advanced" not in EXPECTED_INTERNAL_PAGES
    assert "Voyage-grade Emissions Workspace" not in EXPECTED_INTERNAL_PAGES

    display_label = _function_source(source, module, "_display_page_label")
    assert '"Chat Assistant"' in display_label
    assert '"Analysis Desk"' in display_label
    assert '"Carbon Emissions"' not in display_label

    main = _function(module, "main")
    page_radio = next(
        call
        for call in _calls(main, "radio")
        if any(
            keyword.arg == "options"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "page_options"
            for keyword in call.keywords
        )
    )
    radio_keywords = {keyword.arg: keyword.value for keyword in page_radio.keywords}
    assert isinstance(radio_keywords.get("format_func"), ast.Name)
    assert radio_keywords["format_func"].id == "_display_page_label"

    main_source = _function_source(source, module, "main")
    assert 'selected_page == "Chat Assistant"' in main_source
    assert '"Carbon Emissions"' in main_source


def test_visible_sample_catalog_is_byte_for_byte_equivalent_in_value_and_order() -> None:
    _, module = _source_and_module()
    catalog = _literal_assignment(module, "SAMPLE_QUERIES_BY_CATEGORY")

    assert list(catalog) == EXPECTED_SAMPLE_CATEGORIES
    assert {category: len(prompts) for category, prompts in catalog.items()} == (
        EXPECTED_SAMPLE_COUNTS
    )
    assert sum(EXPECTED_SAMPLE_COUNTS.values()) == 57

    canonical_json = json.dumps(
        catalog,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical_json).hexdigest() == EXPECTED_SAMPLE_CATALOG_SHA256


def test_chat_and_category_session_and_widget_keys_remain_stable() -> None:
    source, module = _source_and_module()
    chat_source = _function_source(source, module, "_render_page_chat")
    category_source = _function_source(source, module, "_render_query_category_page")
    main_source = _function_source(source, module, "main")

    chat_fragments = {
        '"chat_history"',
        '"chat_conversation_id"',
        '"chat_message_page"',
        'key="chat_message_page"',
        'key="chat_send_page"',
        'key="chat_new_page"',
        '"chat_latest_result_bundle"',
    }
    category_fragments = {
        'f"ask_question_{key}"',
        'f"sample_select_{key}"',
        'f"ask_result_bundle_{key}"',
        'f"use_date_range_{key}"',
        'f"ui_port_{key}"',
        'f"ui_vessel_type_{key}"',
        'f"ui_anomaly_{key}"',
        'f"ui_from_{key}"',
        'f"ui_to_{key}"',
        'f"load_sample_{key}"',
        'f"ask_btn_{key}"',
        'f"canonical_conversation_{key}"',
    }
    main_fragments = {
        'st.session_state["chat_history"]',
        'st.session_state["chat_conversation_id"]',
        'st.session_state["chat_message"]',
    }

    assert all(fragment in chat_source for fragment in chat_fragments)
    assert all(fragment in category_source for fragment in category_fragments)
    assert all(fragment in main_source for fragment in main_fragments)


def test_workspace_settings_are_collapsed_without_changing_control_defaults() -> None:
    _, module = _source_and_module()
    main = _function(module, "main")
    settings = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and _call_name(item.context_expr) == "expander"
            and item.context_expr.args
            and isinstance(item.context_expr.args[0], ast.Constant)
            and item.context_expr.args[0].value == "Workspace settings"
            for item in node.items
        )
    )
    expander = next(
        item.context_expr
        for item in settings.items
        if isinstance(item.context_expr, ast.Call)
        and _call_name(item.context_expr) == "expander"
    )
    expander_options = {keyword.arg: keyword.value for keyword in expander.keywords}
    assert isinstance(expander_options.get("expanded"), ast.Constant)
    assert expander_options["expanded"].value is False

    slider = next(call for call in _calls(settings, "slider"))
    slider_options = {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in slider.keywords
    }
    assert slider_options["min_value"] == 0
    assert slider_options["max_value"] == 8
    assert slider_options["value"] == 5

    toggle = next(call for call in _calls(settings, "toggle"))
    toggle_options = {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in toggle.keywords
    }
    assert toggle_options["value"] is False


def test_previous_generic_ai_marketing_shell_is_removed() -> None:
    source, module = _source_and_module()
    rendered_markup = (
        _function_source(source, module, "_render_app_header")
        + _function_source(source, module, "_render_query_category_intro")
    ).lower()
    forbidden_markup = {
        "ee-app-hero",
        "ee-page-hero",
        "ee-sidecar",
        "ee-radar",
        "ee-chip",
    }
    forbidden_copy = {
        "auditable maritime intelligence",
        "harbor signal deck",
        "mission-ready marine workspace",
        "live maritime control room",
        "marine control-deck interface",
        "deterministic truth first",
        "evidence-backed explanations",
        "operationally scoped outputs",
        "generic dashboard shell",
    }

    assert "ee-masthead" in rendered_markup
    assert "ee-query-context" in rendered_markup
    assert sorted(token for token in forbidden_markup if token in rendered_markup) == []
    lowered_source = source.lower()
    assert sorted(token for token in forbidden_copy if token in lowered_source) == []


def test_flat_operations_styles_include_palette_focus_motion_and_breakpoints() -> None:
    source, module = _source_and_module()
    styles = _function_source(source, module, "_apply_global_app_styles").lower()

    for color in ("#071014", "#0d171d", "#27363e", "#e8eff1", "#9fb0b8", "#28b7a5"):
        assert color in styles
    assert "avenir next" in styles
    assert "sf mono" in styles or "sfmono" in styles

    assert ":focus-visible" in styles
    assert "outline:" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    reduced_motion = styles.split("@media (prefers-reduced-motion: reduce)", maxsplit=1)[1]
    assert re.search(
        r"animation(?:-duration)?:\s*(?:none|0(?:\.0+)?(?:ms|s)?|0\.01ms)",
        reduced_motion,
    )
    assert re.search(
        r"transition(?:-duration)?:\s*(?:none|0(?:\.0+)?(?:ms|s)?|0\.01ms)",
        reduced_motion,
    )

    breakpoints = {
        int(value)
        for value in re.findall(r"@media\s*\(max-width:\s*(\d+)px\)", styles)
    }
    assert len(breakpoints) >= 2
    assert any(value >= 900 for value in breakpoints)
    assert any(value <= 720 for value in breakpoints)


def test_query_submissions_still_use_the_canonical_bridge_with_the_same_arguments() -> None:
    _, module = _source_and_module()

    for function_name in ("_render_page_chat", "_render_query_category_page"):
        function = _function(module, function_name)
        calls = _calls(function, "run_canonical_query")
        assert len(calls) == 1
        call = calls[0]
        assert call.args and isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "query_service"
        assert {keyword.arg for keyword in call.keywords} == {
            "question",
            "conversation_id",
            "top_k_evidence",
            "user_filters",
        }
        assert not _calls(function, "_handle_ask_question")
        assert not _calls(function, "classify_question")
        assert not _calls(function, "run_turn")


def test_validated_chart_kinds_and_accessible_fallback_contract_are_preserved() -> None:
    source, module = _source_and_module()
    renderer = _function(module, "_render_canonical_visualizations")
    renderer_source = _function_source(
        source,
        module,
        "_render_canonical_visualizations",
    )

    handled_kinds = {
        comparator.value
        for node in ast.walk(renderer)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "kind"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)
    }
    assert handled_kinds == EXPECTED_GRAPH_KINDS | {"kpi", "omitted", "table"}
    assert "summary = visual.accessible_summary" in renderer_source
    assert "_safe_dataframe(" in renderer_source
    assert "st.pyplot(" in renderer_source
    assert "st.caption(summary)" in renderer_source

    assert _literal_assignment(module, "_GRAPH_VISUALIZATION_KINDS") == (
        EXPECTED_GRAPH_KINDS
    )

    chat_function = _function(module, "_render_page_chat")
    chat_source = _function_source(source, module, "_render_page_chat")
    category_source = _function_source(source, module, "_render_query_category_page")
    chat_compact_calls = _calls(chat_function, "_render_compact_result")
    assert len(chat_compact_calls) == 1
    chat_compact_options = {
        keyword.arg: keyword.value for keyword in chat_compact_calls[0].keywords
    }
    assert isinstance(
        chat_compact_options.get("surface_all_canonical_visuals"),
        ast.Constant,
    )
    assert chat_compact_options["surface_all_canonical_visuals"].value is True
    assert isinstance(chat_compact_options.get("canonical_envelope"), ast.Name)
    assert chat_compact_options["canonical_envelope"].id == "envelope"
    assert "_render_primary_canonical_visualization(" not in chat_source
    assert "surface_all_canonical_visuals=True" in category_source
    assert 'canonical_envelope=bundle.get("canonical_envelope")' in category_source
