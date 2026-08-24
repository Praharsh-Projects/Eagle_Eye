from __future__ import annotations

import ast
from pathlib import Path


STREAMLIT_SOURCE_PATH = Path("src/app/streamlit_app.py")


def test_result_orders_answer_visual_metadata_and_detail_tabs_for_mobile_stacking() -> None:
    """A narrow viewport must reach the answer and graph before secondary detail."""

    source = STREAMLIT_SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    compact_result = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_render_compact_result"
    )
    canonical_branch = next(
        node
        for node in ast.walk(compact_result)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "canonical_envelope"
        and any(
            isinstance(comparator, ast.Constant) and comparator.value is None
            for comparator in node.test.comparators
        )
    )

    answer_lines = [
        node.lineno
        for node in ast.walk(canonical_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "markdown"
        and "ee-answer-label" in (ast.get_source_segment(source, node) or "")
    ]
    chart_lines = [
        node.lineno
        for node in ast.walk(canonical_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_render_chart"
    ]
    source_meta_lines = [
        node.lineno
        for node in ast.walk(canonical_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_meta_card"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Source"
    ]
    detail_tabs = [
        node
        for node in ast.walk(canonical_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tabs"
        and node.args
    ]

    assert len(answer_lines) == 1
    assert len(chart_lines) == 1
    assert len(source_meta_lines) == 2
    canonical_tabs = [
        node
        for node in detail_tabs
        if ast.literal_eval(node.args[0]) == ["Evidence", "Data & scope"]
    ]
    assert len(canonical_tabs) == 1
    assert (
        answer_lines[0]
        < chart_lines[0]
        < min(source_meta_lines)
        <= max(source_meta_lines)
        < canonical_tabs[0].lineno
    )
