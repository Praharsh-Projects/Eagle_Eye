from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import src.app.streamlit_app as streamlit_app


STREAMLIT_SOURCE_PATH = Path("src/app/streamlit_app.py")


class _EnvelopeStub:
    def __init__(self, visualizations):
        self.visualizations = list(visualizations)

    def model_copy(self, *, update):
        return _EnvelopeStub(update["visualizations"])


def test_primary_graph_renderer_selects_one_graph_and_skips_scalar_presentations(monkeypatch) -> None:
    kpi = SimpleNamespace(kind="kpi", id="scalar")
    table = SimpleNamespace(kind="table", id="fallback")
    line = SimpleNamespace(kind="cartesian", id="daily-arrivals")
    map_visual = SimpleNamespace(kind="map", id="ais-map")
    envelope = _EnvelopeStub([kpi, table, line, map_visual])
    rendered = []

    monkeypatch.setattr(
        streamlit_app,
        "_render_canonical_visualizations",
        lambda selected, **options: rendered.append((selected, options)),
    )

    assert streamlit_app._primary_graph_visualization(envelope) is line
    assert streamlit_app._render_primary_canonical_visualization(
        envelope,
        compact=True,
        show_title=False,
    )
    assert [visual.id for visual in rendered[0][0].visualizations] == ["daily-arrivals"]
    assert rendered[0][1] == {"compact": True, "show_title": False}


def test_primary_graph_renderer_does_not_turn_kpi_or_omission_into_a_graph(monkeypatch) -> None:
    rendered = []
    monkeypatch.setattr(
        streamlit_app,
        "_render_canonical_visualizations",
        lambda *args, **kwargs: rendered.append((args, kwargs)),
    )
    envelope = _EnvelopeStub(
        [
            SimpleNamespace(kind="kpi", id="scalar"),
            SimpleNamespace(kind="omitted", id="honest-omission"),
        ]
    )

    assert streamlit_app._primary_graph_visualization(envelope) is None
    assert streamlit_app._render_primary_canonical_visualization(envelope) is False
    assert rendered == []


def test_chat_persists_and_surfaces_full_result_without_legacy_detail_expander() -> None:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    chat_function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_render_page_chat"
    )
    chat_source = ast.get_source_segment(source, chat_function)
    assert chat_source is not None

    assert 'stored_bundle = st.session_state.get("chat_latest_result_bundle")' in chat_source
    assert 'st.session_state["chat_latest_result_bundle"] = latest_bundle' in chat_source
    assert 'st.session_state.pop("chat_latest_result_bundle", None)' in chat_source
    assert "_render_chat_result(stored_bundle)" in chat_source

    render_chat_result = next(
        node
        for node in ast.walk(chat_function)
        if isinstance(node, ast.FunctionDef) and node.name == "_render_chat_result"
    )
    render_source = ast.get_source_segment(source, render_chat_result)
    assert render_source is not None
    assert "_render_compact_result(" in render_source
    assert "surface_all_canonical_visuals=True" in render_source
    assert "_render_primary_canonical_visualization" not in render_source
    assert 'st.expander("Deterministic result used by chat", expanded=False)' not in chat_source


def test_category_snapshot_surfaces_every_canonical_visualization_by_default() -> None:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    category_function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_render_query_category_page"
    )
    category_source = ast.get_source_segment(source, category_function)
    assert category_source is not None

    assert "surface_all_canonical_visuals=True" in category_source
    assert "canonical_envelope=bundle.get(\"canonical_envelope\")" in category_source


def test_streamlit_pages_reuse_the_canonical_preloaded_engines() -> None:
    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    main_function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main_function)
    assert main_source is not None

    assert "kpi_engine = query_service.kpi" in main_source
    assert "forecast_engine = query_service.forecaster" in main_source
    assert "carbon_engine = query_service.carbon" in main_source
    assert "_init_kpi_engine(" not in main_source


def test_percentile_visual_rows_use_labels_and_semantic_order() -> None:
    frame = pd.DataFrame(
        {
            "percentile": ["P90", "P25", "P99", "P50"],
            "median_duration_h": [9.0, 2.5, 12.0, 5.0],
        }
    )

    ordered = streamlit_app._ordered_percentile_frame(
        frame,
        value_field="median_duration_h",
        category_field="percentile",
    )

    assert ordered["category"].tolist() == ["P25", "P50", "P90", "P99"]
    assert ordered["value"].tolist() == [2.5, 5.0, 9.0, 12.0]


def test_computed_source_metadata_discloses_rag_and_keeps_numeric_authority() -> None:
    envelope = SimpleNamespace(
        state=streamlit_app.AnswerState.COMPUTED,
        evidence=[SimpleNamespace(source_type="computed")],
    )

    label, detail = streamlit_app._canonical_source_metadata(
        envelope,
        source_label="Computed analytics",
        source_detail="Deterministic values from validated structured datasets.",
    )

    assert label == "Structured data + supporting evidence"
    assert "structured datasets remain the numeric authority" in detail


def test_public_evidence_table_contains_retrieved_ais_and_document_fields() -> None:
    visible = streamlit_app._evidence_rows_for_display(
        [
            {
                "evidence_id": "ais-1",
                "source_type": "computed",
                "title": "Supporting AIS event",
                "excerpt": "Vessel observed near the selected port.",
                "timestamp_full": "2022-03-01T09:00:00Z",
                "mmsi": "123456789",
            },
            {
                "evidence_id": "doc-1",
                "source_type": "local_document",
                "title": "SOLAS chapter",
                "excerpt": "Local document excerpt.",
                "url": "https://example.test/solas",
            },
        ]
    )

    assert {"Evidence ID", "Source type", "Source / event", "Evidence excerpt"}.issubset(
        visible.columns
    )
    assert visible["Source type"].tolist() == ["computed", "local_document"]
    assert "MMSI" in visible.columns
