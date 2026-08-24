"""Build the versioned runtime data manifest used by API capabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from src.kpi.reconstruct_voyages import RECONSTRUCTION_VERSION
from src.utils.parquet_io import read_parquet_safely


MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FILE = "data_manifest.json"

TABLE_DATE_COLUMNS: Dict[str, tuple[str, ...]] = {
    "arrivals_daily.parquet": ("date",),
    "arrivals_hourly.parquet": ("datetime_hour",),
    "dwell_time.parquet": ("arrival_time", "departure_time"),
    "occupancy_hourly.parquet": ("datetime_hour",),
    "congestion_daily.parquet": ("date",),
    "voyages.parquet": ("departure_time", "arrival_time"),
    "carbon_emissions_daily_port.parquet": ("date",),
    "carbon_emissions_call.parquet": ("timestamp_start", "timestamp_end"),
    "events.parquet": ("timestamp_full",),
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_date(value: Any) -> Optional[str]:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _table_summary(path: Path, date_columns: Iterable[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    try:
        data = read_parquet_safely(path)
    except Exception as exc:
        summary.update({"readable": False, "error": type(exc).__name__})
        return summary
    summary.update({"readable": True, "rows": int(len(data)), "columns": list(data.columns)})
    coverage: Dict[str, Dict[str, Optional[str]]] = {}
    for column in date_columns:
        if column not in data.columns:
            continue
        values = pd.to_datetime(data[column], errors="coerce", utc=True).dropna()
        if values.empty:
            continue
        coverage[column] = {"min": _iso_date(values.min()), "max": _iso_date(values.max())}
    if coverage:
        summary["coverage"] = coverage
    return summary


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _forecast_validation(processed: Path) -> Dict[str, Any]:
    payload = _load_json(processed / "forecast_backtest.json")
    if not payload:
        return {"available": False, "reason": "forecast_backtest.json is missing or invalid"}
    output: Dict[str, Any] = {"available": True, "settings": payload.get("settings", {})}
    for label in ("arrivals", "congestion"):
        metric = payload.get(label, {})
        per_port = metric.get("per_port", []) if isinstance(metric, dict) else []
        passing = [
            str(row.get("port_key"))
            for row in per_port
            if isinstance(row, dict)
            and row.get("gate_passed") is True
        ]
        output[label] = {
            "metric": metric.get("metric") if isinstance(metric, dict) else None,
            "ports_evaluated": int(metric.get("ports_evaluated", 0)) if isinstance(metric, dict) else 0,
            "passing_ports": sorted(passing),
            "mase_mean": metric.get("mase_mean") if isinstance(metric, dict) else None,
            "interval_80_coverage_mean": metric.get("interval_80_coverage_mean") if isinstance(metric, dict) else None,
        }
    return output


def build_data_manifest(
    processed_dir: str | Path = "data/processed",
    *,
    models_dir: str | Path = "models",
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    processed = Path(processed_dir)
    models = Path(models_dir)
    table_summaries: Dict[str, Any] = {}
    for filename, date_columns in TABLE_DATE_COLUMNS.items():
        path = processed / filename
        if path.exists():
            table_summaries[filename] = _table_summary(path, date_columns)

    catalog_path = processed / "port_catalog.parquet"
    ports: list[str] = []
    if catalog_path.exists():
        catalog = read_parquet_safely(catalog_path)
        if not catalog.empty and "port_key" in catalog.columns:
            if "source_kind" in catalog.columns and (catalog["source_kind"] == "port_call").any():
                catalog = catalog[catalog["source_kind"] == "port_call"]
            ports = sorted(catalog["port_key"].dropna().astype(str).unique().tolist())

    capabilities = _load_json(processed / "kpi_capabilities.json")
    forecast_validation = _forecast_validation(processed)
    voyages = table_summaries.get("voyages.parquet", {})
    enabled_operations = [
        "arrivals",
        "arrival_composition",
        "temporal_patterns",
        "dwell_distribution",
        "pressure_v2",
        "anomaly_detection",
        "carbon_emissions",
    ]
    if int(voyages.get("rows", 0)) > 0:
        enabled_operations.extend(["first_route_vessel", "route_travel_time_summary"])
    if forecast_validation.get("arrivals", {}).get("passing_ports"):
        enabled_operations.append("arrival_forecast")
    if forecast_validation.get("congestion", {}).get("passing_ports"):
        enabled_operations.append("pressure_forecast")

    model_validation: Dict[str, Any] = {"forecast": forecast_validation}
    for filename in ("eta_metrics.json", "destination_metrics.json", "anomaly_metrics.json"):
        payload = _load_json(models / filename)
        if payload:
            model_validation[filename.removesuffix("_metrics.json")] = payload

    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_contract": {
            "historical_only": True,
            "pressure_version": "pressure_v2",
            "voyage_reconstruction_version": RECONSTRUCTION_VERSION,
            "forecast_gate": {
                "mase_lt": 1.0,
                "interval_80_coverage_min": 0.70,
                "interval_80_coverage_max": 0.90,
                "maximum_horizon_days": 56,
            },
        },
        "tables": table_summaries,
        "available_ports": ports,
        "enabled_operations": sorted(set(enabled_operations)),
        "capabilities": capabilities,
        "model_validation": model_validation,
    }
    destination = Path(out_path) if out_path is not None else processed / MANIFEST_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Eagle Eye's versioned data manifest.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    payload = build_data_manifest(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        out_path=args.out,
    )
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "tables": len(payload["tables"]),
                "ports": len(payload["available_ports"]),
                "enabled_operations": payload["enabled_operations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
