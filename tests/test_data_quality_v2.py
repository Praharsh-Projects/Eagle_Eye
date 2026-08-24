from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.carbon.query import CarbonQueryEngine
from src.forecast.forecast import ForecastEngine
from src.kpi.build_kpis import _build_congestion_daily
from src.kpi.data_manifest import build_data_manifest
from src.kpi.query import KPIQueryEngine
from src.kpi.reconstruct_voyages import reconstruct_voyages
from src.qa.intent import classify_question


def _call(
    mmsi: str,
    port: str,
    arrival: str,
    departure: str,
    *,
    vessel_type: str = "cargo ship",
) -> dict[str, object]:
    return {
        "source_kind": "port_call",
        "source_file": "calls.csv",
        "mmsi": mmsi,
        "port_key": port,
        "port_label": port,
        "locode_norm": port,
        "arrival_time": pd.Timestamp(arrival),
        "departure_time": pd.Timestamp(departure),
        "vessel_type_norm": vessel_type,
    }


def test_voyage_reconstruction_pairs_departure_to_next_valid_arrival() -> None:
    calls = pd.DataFrame(
        [
            _call("111111111", "SEGOT", "2022-03-01T08:00:00Z", "2022-03-01T10:00:00Z"),
            _call("111111111", "SEKAN", "2022-03-01T15:00:00Z", "2022-03-01T18:00:00Z"),
            _call("111111111", "PLGDN", "2022-03-02T06:00:00Z", "2022-03-02T09:00:00Z"),
            _call("222222222", "SEGOT", "2022-01-01T08:00:00Z", "2022-01-01T10:00:00Z"),
            _call("222222222", "SEKAN", "2022-02-15T08:00:00Z", "2022-02-15T10:00:00Z"),
        ]
    )
    voyages = reconstruct_voyages(calls, max_gap_days=30)

    assert len(voyages) == 2
    first = voyages.iloc[0]
    assert first["origin_port_key"] == "SEGOT"
    assert first["destination_port_key"] == "SEKAN"
    assert first["duration_h"] == 5.0
    assert first["origin_call_id"]
    assert first["destination_call_id"]
    assert first["voyage_id"].startswith("voyage_")
    assert set(voyages["source_kind"]) == {"reconstructed_port_calls"}


def test_pressure_v2_uses_weighted_clipped_ratios_and_marks_partial_rows() -> None:
    dates = pd.to_datetime(
        ["2022-03-07", "2022-03-14", "2022-03-21", "2022-03-28"], utc=True
    )
    arrivals = pd.DataFrame(
        {
            "source_kind": "port_call",
            "port_key": "SEGOT",
            "port_label": "Gothenburg",
            "locode_norm": "SEGOT",
            "port_name_norm": "gothenburg",
            "date": dates,
            "arrivals_vessels": [10, 10, 10, 200],
            "arrivals_events": [10, 10, 10, 200],
        }
    )
    dwell = pd.DataFrame(
        {
            "port_key": ["SEGOT", "SEGOT", "SEGOT"],
            "arrival_date": dates[:3],
            "dwell_minutes": [100.0, 200.0, 100.0],
        }
    )
    pressure = _build_congestion_daily(arrivals, dwell)

    assert set(pressure["pressure_version"]) == {"pressure_v2"}
    assert pressure["arrivals_ratio"].between(0, 5).all()
    assert pressure["congestion_index"].between(0, 5).all()
    assert set(pressure["pressure_kind"]) == {"full", "partial_arrivals_proxy"}
    baseline = pressure.iloc[0]
    assert baseline["congestion_index"] == 1.0
    partial = pressure.iloc[-1]
    assert partial["congestion_index"] == 5.0
    assert not bool(partial["is_cross_scope_comparable"])


def test_intent_parser_handles_regression_phrasings_without_arrivals_default() -> None:
    month = classify_question("How many ship calls did Gothenburg receive during March of 2022?")
    assert month.entities["date_from"] == "2022-03-01"
    assert month.entities["date_to"] == "2022-03-31"
    assert month.entities["metric"] == "arrival_count"

    weekdays = classify_question("Is Monday busier than Friday at Gothenburg in March 2022?")
    assert weekdays.intent == "B"
    assert weekdays.entities["dow"] == "Monday"
    assert weekdays.entities["dow_compare"] == "Friday"

    composition = classify_question("Show the share of arrivals by vessel type at Gothenburg in March 2022.")
    assert composition.entities["aggregation"] == "vessel_type_composition"
    distribution = classify_question("Show the distribution of vessel dwell times at Gothenburg.")
    assert distribution.entities["aggregation"] == "dwell_distribution"

    assert classify_question("Hello, what can you do?").intent == "G"
    assert classify_question("What does SOLAS Chapter V require?").intent == "G"
    current = classify_question("What is the weather at Gothenburg today?")
    assert current.intent == "G"
    assert current.entities["requires_current_data"] is True
    assert current.entities["temporal_reference"] == "current"


def test_composition_distribution_and_historical_freshness(tmp_path: Path) -> None:
    calls = pd.DataFrame(
        [
            _call("111111111", "SEGOT", "2022-03-01T08:00:00Z", "2022-03-01T10:00:00Z"),
            _call("222222222", "SEGOT", "2022-03-02T08:00:00Z", "2022-03-02T12:00:00Z", vessel_type="tanker"),
        ]
    )
    calls["dwell_minutes"] = (
        calls["departure_time"] - calls["arrival_time"]
    ).dt.total_seconds() / 60.0
    calls["arrival_date"] = calls["arrival_time"].dt.floor("D")
    calls.to_parquet(tmp_path / "dwell_time.parquet", index=False)
    arrivals = pd.DataFrame(
        {
            "source_kind": ["port_call", "port_call"],
            "port_key": ["SEGOT", "SEGOT"],
            "port_label": ["Gothenburg", "Gothenburg"],
            "locode_norm": ["SEGOT", "SEGOT"],
            "port_name_norm": ["gothenburg", "gothenburg"],
            "date": pd.to_datetime(["2022-03-01", "2022-03-02"], utc=True),
            "vessel_type_norm": ["cargo ship", "tanker"],
            "arrivals_vessels": [3, 1],
            "arrivals_events": [3, 1],
        }
    )
    arrivals.to_parquet(tmp_path / "arrivals_daily.parquet", index=False)
    pd.DataFrame(
        [{"port_key": "SEGOT", "locode_norm": "SEGOT", "port_label": "Gothenburg", "port_name_norm": "gothenburg", "source_kind": "port_call", "arrivals_total": 4}]
    ).to_parquet(tmp_path / "port_catalog.parquet", index=False)
    (tmp_path / "kpi_capabilities.json").write_text(
        json.dumps({"date_min": "2022-03-01", "date_max": "2022-03-02"}), encoding="utf-8"
    )
    engine = KPIQueryEngine(tmp_path)

    composition = engine.get_arrival_composition("SEGOT", "2022-03-01", "2022-03-31")
    assert composition.status == "ok"
    assert composition.table is not None
    assert composition.table.iloc[0]["share_percent"] == 75.0
    dwell = engine.get_dwell_distribution("SEGOT", "2022-03-01", "2022-03-31")
    assert dwell.status == "ok"
    assert dwell.chart is not None
    assert int(dwell.table["calls"].sum()) == 2
    assert engine.has_current_data("2026-07-22") is False
    assert "2022-03-02" in engine.no_current_data("today").answer


def test_date_only_upper_bound_includes_the_full_final_day() -> None:
    engine = KPIQueryEngine("data/processed")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2022-03-31T00:00:00Z", "2022-03-31T23:59:59Z", "2022-04-01T00:00:00Z"]
            )
        }
    )
    selected = engine._filter_dates(frame, "timestamp", "2022-03-01", "2022-03-31")
    assert len(selected) == 2


def test_forecast_keeps_horizon_gate_while_quality_metrics_are_diagnostic(tmp_path: Path) -> None:
    dates = pd.date_range("2022-01-01", periods=90, freq="D", tz="UTC")
    arrivals = pd.DataFrame(
        {
            "source_kind": "port_call",
            "port_key": "SEGOT",
            "port_label": "Gothenburg",
            "locode_norm": "SEGOT",
            "port_name_norm": "gothenburg",
            "date": dates,
            "vessel_type_norm": "cargo ship",
            "arrivals_vessels": [10 + (i % 7) for i in range(90)],
            "arrivals_events": [10 + (i % 7) for i in range(90)],
        }
    )
    arrivals.to_parquet(tmp_path / "arrivals_daily.parquet", index=False)
    pd.DataFrame(
        [{"port_key": "SEGOT", "locode_norm": "SEGOT", "port_label": "Gothenburg", "port_name_norm": "gothenburg", "source_kind": "port_call", "arrivals_total": 1000}]
    ).to_parquet(tmp_path / "port_catalog.parquet", index=False)
    (tmp_path / "kpi_capabilities.json").write_text(
        json.dumps({"date_min": "2022-01-01", "date_max": "2022-03-31"}), encoding="utf-8"
    )
    (tmp_path / "forecast_backtest.json").write_text(
        json.dumps(
            {
                "arrivals": {
                    "per_port": [
                        {"port_key": "SEGOT", "mase": 0.8, "interval_80_coverage": 0.8, "gate_passed": True}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    engine = ForecastEngine(tmp_path)
    assert engine.forecast_arrivals("SEGOT", horizon_weeks=4).status == "ok"
    assert engine.forecast_arrivals("SEGOT", horizon_weeks=9).status == "no_data"

    payload = json.loads((tmp_path / "forecast_backtest.json").read_text(encoding="utf-8"))
    payload["arrivals"]["per_port"][0]["mase"] = 1.1
    (tmp_path / "forecast_backtest.json").write_text(json.dumps(payload), encoding="utf-8")
    finite_result = ForecastEngine(tmp_path).forecast_arrivals("SEGOT", horizon_weeks=4)
    assert finite_result.status == "ok"
    assert finite_result.forecast is not None
    assert not finite_result.forecast.empty
    assert finite_result.forecast["predicted"].notna().all()


def test_manifest_and_carbon_queries_are_read_only_until_explicit_export(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    arrivals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-03-01"], utc=True),
            "port_key": ["SEGOT"],
            "source_kind": ["port_call"],
            "arrivals_vessels": [1],
        }
    )
    arrivals.to_parquet(processed / "arrivals_daily.parquet", index=False)
    pd.DataFrame(
        [{"port_key": "SEGOT", "source_kind": "port_call"}]
    ).to_parquet(processed / "port_catalog.parquet", index=False)
    (processed / "kpi_capabilities.json").write_text(
        json.dumps({"date_min": "2022-03-01", "date_max": "2022-03-01"}), encoding="utf-8"
    )
    manifest = build_data_manifest(processed, models_dir=tmp_path / "models")
    assert manifest["schema_version"] == "1.0"
    assert manifest["data_contract"]["historical_only"] is True
    assert "SEGOT" in manifest["available_ports"]

    carbon_root = tmp_path / "carbon"
    carbon_root.mkdir()
    engine = CarbonQueryEngine(
        processed_dir=carbon_root,
        factor_registry_path=Path(__file__).resolve().parents[1] / "config/carbon_factors.v1.json",
        auto_build=False,
    )
    before = set(carbon_root.rglob("*"))
    result = engine.estimate_with_assumptions(
        {"vessel_type": "cargo ship", "mode": "transit", "duration_h": 1, "speed_kn": 10}
    )
    assert result.status == "ok"
    assert set(carbon_root.rglob("*")) == before
    assert result.export_csv_path is None
    assert result.export_json_path is None

    csv_path, json_path = engine.export_result(result, prefix="carbon_estimate")
    assert csv_path and Path(csv_path).exists()
    assert json_path and Path(json_path).exists()
