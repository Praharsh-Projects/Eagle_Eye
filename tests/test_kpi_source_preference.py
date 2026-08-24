from __future__ import annotations

import pandas as pd

from src.kpi.query import AnalyticsResult, KPIQueryEngine
from src.utils.confidence import extract_confidence_label


def _build_engine() -> KPIQueryEngine:
    engine = KPIQueryEngine(processed_dir="data/processed")
    engine._port_catalog = pd.DataFrame(
        [
            {
                "port_key": "SEGOT",
                "locode_norm": "SEGOT",
                "port_label": "Göteborg (SEGOT)",
                "port_name_norm": "göteborg",
                "source_kind": "port_call",
                "arrivals_total": 1000,
            },
            {
                "port_key": "SEGOT",
                "locode_norm": "SEGOT",
                "port_label": "SEGOT",
                "port_name_norm": "segot",
                "source_kind": "ais_destination_proxy",
                "arrivals_total": 100,
            },
        ]
    )
    return engine


def test_filter_port_does_not_match_embedded_destination_tokens() -> None:
    engine = _build_engine()
    df = pd.DataFrame(
        [
            {"port_key": "SEGOT", "locode_norm": "SEGOT", "port_label": "Göteborg (SEGOT)", "port_name_norm": "göteborg"},
            {"port_key": "LVVNT SEGOT", "locode_norm": "", "port_label": "LVVNT SEGOT", "port_name_norm": "lvvnt segot"},
            {"port_key": "SEGOT PLSZZ", "locode_norm": "", "port_label": "SEGOT PLSZZ", "port_name_norm": "segot plszz"},
        ]
    )

    filtered = engine._filter_port(df, "SEGOT")

    assert filtered["port_key"].tolist() == ["SEGOT"]


def test_prefer_arrival_source_uses_port_call_for_same_port_and_day() -> None:
    engine = _build_engine()
    df = pd.DataFrame(
        [
            {
                "source_kind": "ais_destination_proxy",
                "port_key": "SEGOT",
                "port_label": "SEGOT",
                "locode_norm": "SEGOT",
                "port_name_norm": "segot",
                "date": pd.Timestamp("2022-03-01", tz="UTC"),
                "arrivals_vessels": 2,
                "arrivals_events": 2,
            },
            {
                "source_kind": "port_call",
                "port_key": "SEGOT",
                "port_label": "Göteborg (SEGOT)",
                "locode_norm": "SEGOT",
                "port_name_norm": "göteborg",
                "date": pd.Timestamp("2022-03-01", tz="UTC"),
                "arrivals_vessels": 10,
                "arrivals_events": 10,
            },
            {
                "source_kind": "ais_destination_proxy",
                "port_key": "SEGOT",
                "port_label": "SEGOT",
                "locode_norm": "SEGOT",
                "port_name_norm": "segot",
                "date": pd.Timestamp("2022-03-02", tz="UTC"),
                "arrivals_vessels": 3,
                "arrivals_events": 3,
            },
        ]
    )

    preferred = (
        engine._prefer_arrival_source(df, "date", allow_day_gap_fallback=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    assert preferred["source_kind"].tolist() == ["port_call", "ais_destination_proxy"]
    assert preferred["arrivals_vessels"].tolist() == [10, 3]
    assert preferred.loc[0, "port_label"] == "Göteborg (SEGOT)"


def test_category_filtered_source_selection_does_not_gap_fill_from_ais_proxy() -> None:
    engine = _build_engine()
    df = pd.DataFrame(
        [
            {
                "source_kind": "port_call",
                "port_key": "LVVNT",
                "date": pd.Timestamp("2022-03-01", tz="UTC"),
                "vessel_type_norm": "tanker",
            },
            {
                "source_kind": "ais_destination_proxy",
                "port_key": "LVVNT",
                "date": pd.Timestamp("2022-03-02", tz="UTC"),
                "vessel_type_norm": "tanker",
            },
        ]
    )

    preferred = engine._prefer_arrival_source(df, "date", allow_day_gap_fallback=False)

    assert preferred["source_kind"].tolist() == ["port_call"]


def test_confidence_is_high_for_direct_port_call_result() -> None:
    result = AnalyticsResult(
        status="ok",
        answer="Matched 473 vessel arrivals across 31 day buckets for SEGOT.",
        table=pd.DataFrame([{"date": "2022-03-01", "arrivals_vessels": 18}]),
        chart=None,
        coverage_notes=[
            "Coverage window: 2022-03-01 to 2022-03-31",
            "Data sources used: port_call",
            "Rows used: 31",
        ],
        caveats=["Arrivals are computed from structured port-call rows for the matched scope."],
    )

    assert extract_confidence_label(result).startswith("high")
