"""Reproducible thesis data pipeline for PRJ912 + PRJ896.

Outputs cleaned parquet tables + engineered features used by chunking/evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.thesis.common import (
    haversine_km,
    load_csv_with_schema,
    normalize_destination,
    normalize_identifier,
    normalize_locode,
    normalize_vessel_type,
    to_numeric,
)

AIS_REQUIRED = {
    "MMSI",
    "TimePosition",
    "Latitude",
    "Longitude",
    "Speed",
    "Course",
    "Heading",
    "NavStatus",
    "IMO",
    "Name",
    "Callsign",
    "Flag",
    "VesselType",
    "Destination",
    "TimeETA",
    "Draught",
}

PORT_REQUIRED = {
    "portID",
    "portName",
    "portLocode",
    "portArrival",
    "portDeparture",
    "vesselMMSI",
    "vesselIMO",
    "vesselName",
    "vesselDestinationArrival",
    "vesselDestinationDeparture",
    "vesselType",
}


def _is_ais_schema(columns: Sequence[str]) -> bool:
    return len(AIS_REQUIRED.intersection(set(columns))) >= 8


def _is_port_schema(columns: Sequence[str]) -> bool:
    return len(PORT_REQUIRED.intersection(set(columns))) >= 6


def _safe_mode(series: pd.Series, fallback: str = "UNKNOWN") -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return fallback
    mode = values.mode()
    if mode.empty:
        return fallback
    out = str(mode.iloc[0]).strip()
    return out if out else fallback


def _prepare_port_calls(df_raw: pd.DataFrame, source_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df_raw.copy()
    for col in PORT_REQUIRED:
        if col not in df.columns:
            df[col] = pd.NA

    df["mmsi"] = df["vesselMMSI"].map(normalize_identifier)
    df["imo"] = df["vesselIMO"].map(normalize_identifier)
    df["arrival_time"] = pd.to_datetime(df["portArrival"], errors="coerce", utc=True)
    df["departure_time"] = pd.to_datetime(df["portDeparture"], errors="coerce", utc=True)
    df["port_name"] = df["portName"].fillna("unknown").astype("string").str.strip()
    df["port_name_norm"] = df["port_name"].str.lower()
    df["locode_norm"] = df["portLocode"].map(normalize_locode)
    df["port_key"] = np.where(df["locode_norm"] != "", df["locode_norm"], df["port_name_norm"].str.upper().str.replace(" ", "_", regex=False))

    df["vessel_name"] = df["vesselName"].fillna("unknown").astype("string").str.strip()
    df["vessel_type_norm"] = df["vesselType"].map(normalize_vessel_type)
    df["destination_arrival_norm"] = df["vesselDestinationArrival"].map(normalize_destination)
    df["destination_departure_norm"] = df["vesselDestinationDeparture"].map(normalize_destination)

    df = df.dropna(subset=["arrival_time"])
    df = df[(df["mmsi"] != "") & (df["port_key"] != "")]

    df["date"] = df["arrival_time"].dt.floor("D")
    df["hour"] = df["arrival_time"].dt.floor("h")
    df["day_of_week"] = df["arrival_time"].dt.day_name()

    df["dwell_minutes"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60.0
    df["dwell_hours"] = df["dwell_minutes"] / 60.0
    df.loc[(df["dwell_minutes"] <= 0) | (df["dwell_minutes"] > 60 * 24 * 45), ["dwell_minutes", "dwell_hours"]] = np.nan

    # Port-aware dwell anomaly thresholds.
    q10 = df.groupby("port_key", dropna=False)["dwell_minutes"].transform(lambda x: x.quantile(0.10))
    q90 = df.groupby("port_key", dropna=False)["dwell_minutes"].transform(lambda x: x.quantile(0.90))
    df["anomaly_short_dwell"] = (df["dwell_minutes"].notna()) & (df["dwell_minutes"] < q10)
    df["anomaly_long_dwell"] = (df["dwell_minutes"].notna()) & (df["dwell_minutes"] > q90)
    df["anomaly_flag"] = df["anomaly_short_dwell"] | df["anomaly_long_dwell"]

    arrival_ts = df["arrival_time"].dt.strftime("%Y-%m-%dT%H-%M-%S")
    df["event_id"] = (
        df["mmsi"].astype(str)
        + "_"
        + arrival_ts
        + "_"
        + df["port_key"].astype(str)
        + "_portcall"
    )
    df["source_file"] = source_file

    clean_cols = [
        "event_id",
        "source_file",
        "mmsi",
        "imo",
        "vessel_name",
        "vessel_type_norm",
        "port_key",
        "port_name",
        "port_name_norm",
        "locode_norm",
        "arrival_time",
        "departure_time",
        "date",
        "hour",
        "day_of_week",
        "destination_arrival_norm",
        "destination_departure_norm",
        "dwell_minutes",
        "dwell_hours",
        "anomaly_short_dwell",
        "anomaly_long_dwell",
        "anomaly_flag",
    ]
    clean = df[clean_cols].copy()

    arrivals_hour = (
        clean.groupby(["port_key", "port_name", "locode_norm", "hour", "vessel_type_norm"], dropna=False)
        .agg(arrivals_events=("event_id", "size"), arrivals_vessels=("mmsi", "nunique"))
        .reset_index()
        .sort_values(["port_key", "hour", "vessel_type_norm"])
    )

    port_day = (
        clean.groupby(["port_key", "port_name", "locode_norm", "date"], dropna=False)
        .agg(
            arrivals_events=("event_id", "size"),
            arrivals_vessels=("mmsi", "nunique"),
            avg_dwell_hours=("dwell_hours", "mean"),
            median_dwell_hours=("dwell_hours", "median"),
            short_dwell_count=("anomaly_short_dwell", "sum"),
            long_dwell_count=("anomaly_long_dwell", "sum"),
            anomaly_count=("anomaly_flag", "sum"),
        )
        .reset_index()
    )

    mode_origin = (
        clean.groupby(["port_key", "date"], dropna=False)["destination_arrival_norm"]
        .apply(_safe_mode)
        .rename("most_frequent_origin")
        .reset_index()
    )
    mode_depart = (
        clean.groupby(["port_key", "date"], dropna=False)["destination_departure_norm"]
        .apply(_safe_mode)
        .rename("most_frequent_next_destination")
        .reset_index()
    )

    port_day = port_day.merge(mode_origin, on=["port_key", "date"], how="left")
    port_day = port_day.merge(mode_depart, on=["port_key", "date"], how="left")
    port_day["source"] = "PRJ896"

    route_freq = (
        clean.assign(origin_port=clean["port_key"], destination=clean["destination_departure_norm"])
        .query("destination.notnull()")
        .groupby(["origin_port", "destination"], dropna=False)
        .agg(route_count=("event_id", "size"), vessels=("mmsi", "nunique"))
        .reset_index()
        .sort_values("route_count", ascending=False)
    )

    return clean, arrivals_hour, port_day, route_freq


def _prepare_ais(df_raw: pd.DataFrame, source_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_raw.copy()
    for col in AIS_REQUIRED:
        if col not in df.columns:
            df[col] = pd.NA

    df["mmsi"] = df["MMSI"].map(normalize_identifier)
    df["imo"] = df["IMO"].map(normalize_identifier)
    df["timestamp"] = pd.to_datetime(df["TimePosition"], errors="coerce", utc=True)
    df["eta_time"] = pd.to_datetime(df["TimeETA"], errors="coerce", utc=True)

    df["latitude"] = to_numeric(df["Latitude"])
    df["longitude"] = to_numeric(df["Longitude"])
    df["speed_kn"] = to_numeric(df["Speed"])
    df["course_deg"] = to_numeric(df["Course"])
    df["heading_deg"] = to_numeric(df["Heading"])
    df["draught_m"] = to_numeric(df["Draught"])

    df["vessel_name"] = df["Name"].fillna("unknown").astype("string").str.strip()
    df["callsign"] = df["Callsign"].fillna("unknown").astype("string").str.strip()
    df["flag"] = df["Flag"].fillna("unknown").astype("string").str.strip().str.upper()
    df["vessel_type_norm"] = df["VesselType"].map(normalize_vessel_type)
    df["nav_status"] = df["NavStatus"].fillna("unknown").astype("string").str.strip().str.lower()
    df["destination_norm"] = df["Destination"].map(normalize_destination)

    df = df.dropna(subset=["timestamp", "latitude", "longitude"])
    df = df[df["mmsi"] != ""]
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]

    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    g = df.groupby("mmsi", sort=False)
    prev_ts = g["timestamp"].shift(1)
    prev_lat = g["latitude"].shift(1)
    prev_lon = g["longitude"].shift(1)

    df["delta_minutes"] = (df["timestamp"] - prev_ts).dt.total_seconds() / 60.0
    df["delta_km"] = haversine_km(prev_lat.fillna(df["latitude"]), prev_lon.fillna(df["longitude"]), df["latitude"], df["longitude"])
    df.loc[df["delta_minutes"] <= 0, ["delta_minutes", "delta_km"]] = np.nan

    hours = df["delta_minutes"] / 60.0
    df["speed_est_kn"] = (df["delta_km"] / hours) / 1.852
    df["anomaly_jump_30m"] = (
        df["delta_minutes"].notna()
        & (df["delta_minutes"] <= 30)
        & (df["delta_km"] >= 80)
    )

    ts_fmt = df["timestamp"].dt.strftime("%Y-%m-%dT%H-%M-%S")
    df["event_id"] = (
        df["mmsi"].astype(str)
        + "_"
        + ts_fmt
        + "_"
        + df["latitude"].round(5).astype(str)
        + "_"
        + df["longitude"].round(5).astype(str)
        + "_ais"
    )
    df["date"] = df["timestamp"].dt.floor("D")
    df["hour"] = df["timestamp"].dt.floor("h")
    df["source_file"] = source_file

    clean_cols = [
        "event_id",
        "source_file",
        "mmsi",
        "imo",
        "timestamp",
        "date",
        "hour",
        "latitude",
        "longitude",
        "speed_kn",
        "course_deg",
        "heading_deg",
        "draught_m",
        "speed_est_kn",
        "delta_minutes",
        "delta_km",
        "anomaly_jump_30m",
        "vessel_name",
        "callsign",
        "flag",
        "vessel_type_norm",
        "nav_status",
        "destination_norm",
        "eta_time",
    ]
    clean = df[clean_cols].copy()

    daily_proxy = (
        clean[clean["destination_norm"] != "UNKNOWN"]
        .groupby(["destination_norm", "date", "vessel_type_norm"], dropna=False)
        .agg(arrivals_events=("event_id", "size"), arrivals_vessels=("mmsi", "nunique"))
        .reset_index()
        .rename(columns={"destination_norm": "port_key"})
    )
    daily_proxy["source"] = "PRJ912_destination_proxy"

    return clean, daily_proxy


def _estimate_memory_gb() -> Optional[float]:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return round(total / (1024**3), 2)
    except Exception:
        return None
    return None


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() or torch.backends.mps.is_available())
    except Exception:
        return False


def _samples(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    sample = df.head(n).copy()
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].astype("string")
    return sample.to_dict(orient="records")


def _output_structure(root: Path) -> Dict[str, str]:
    files = [
        "port_calls_clean.parquet",
        "ais_clean.parquet",
        "arrivals_per_hour.parquet",
        "port_day_metrics.parquet",
        "route_frequency.parquet",
        "ais_destination_daily.parquet",
        "dataset_profile.json",
        "thesis_context.json",
    ]
    return {name: str(root / name) for name in files}


def build_dataset(
    prj912_path: Path,
    prj896_path: Path,
    out_dir: Path,
    limit_rows: Optional[int] = None,
) -> Dict[str, Any]:
    raw_912 = load_csv_with_schema(prj912_path, limit_rows=limit_rows)
    raw_896 = load_csv_with_schema(prj896_path, limit_rows=limit_rows)

    if not _is_ais_schema(list(raw_912.columns)):
        raise RuntimeError(f"{prj912_path} does not look like PRJ912 AIS schema")
    if not _is_port_schema(list(raw_896.columns)):
        raise RuntimeError(f"{prj896_path} does not look like PRJ896 port-call schema")

    port_clean, arrivals_hour, port_day, route_freq = _prepare_port_calls(raw_896, source_file=prj896_path.name)
    ais_clean, ais_daily = _prepare_ais(raw_912, source_file=prj912_path.name)

    # Attach AIS destination proxy volume to port-day table when destination key matches.
    ais_daily_tot = (
        ais_daily.groupby(["port_key", "date"], dropna=False)
        .agg(ais_proxy_arrivals_events=("arrivals_events", "sum"), ais_proxy_arrivals_vessels=("arrivals_vessels", "sum"))
        .reset_index()
    )
    port_day = port_day.merge(ais_daily_tot, on=["port_key", "date"], how="left")

    out_dir.mkdir(parents=True, exist_ok=True)

    port_clean.to_parquet(out_dir / "port_calls_clean.parquet", index=False)
    ais_clean.to_parquet(out_dir / "ais_clean.parquet", index=False)
    arrivals_hour.to_parquet(out_dir / "arrivals_per_hour.parquet", index=False)
    port_day.to_parquet(out_dir / "port_day_metrics.parquet", index=False)
    route_freq.to_parquet(out_dir / "route_frequency.parquet", index=False)
    ais_daily.to_parquet(out_dir / "ais_destination_daily.parquet", index=False)

    dataset_profile = {
        "prj912_columns": list(raw_912.columns),
        "prj896_columns": list(raw_896.columns),
        "prj912_sample_rows": _samples(raw_912, n=5),
        "prj896_sample_rows": _samples(raw_896, n=5),
        "date_formats": {
            "prj912_timeposition": "ISO-8601 UTC (e.g., 2021-01-01T00:04:03.000Z)",
            "prj896_portarrival": "ISO-8601 UTC (e.g., 2021-01-01T00:38:44.000Z)",
            "prj896_portdeparture": "ISO-8601 UTC",
        },
        "coverage": {
            "port_calls_min_date": str(port_clean["date"].min()) if not port_clean.empty else None,
            "port_calls_max_date": str(port_clean["date"].max()) if not port_clean.empty else None,
            "ais_min_date": str(ais_clean["date"].min()) if not ais_clean.empty else None,
            "ais_max_date": str(ais_clean["date"].max()) if not ais_clean.empty else None,
        },
        "row_counts": {
            "port_calls_clean": int(len(port_clean)),
            "ais_clean": int(len(ais_clean)),
            "arrivals_per_hour": int(len(arrivals_hour)),
            "port_day_metrics": int(len(port_day)),
            "route_frequency": int(len(route_freq)),
            "ais_destination_daily": int(len(ais_daily)),
        },
    }

    thesis_context = {
        "dataset_columns": {
            "PRJ912": list(raw_912.columns),
            "PRJ896": list(raw_896.columns),
        },
        "sample_rows": {
            "PRJ912": dataset_profile["prj912_sample_rows"],
            "PRJ896": dataset_profile["prj896_sample_rows"],
        },
        "date_format": dataset_profile["date_formats"],
        "target_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_db": "Chroma (local persistent)",
        "output_folder_structure": _output_structure(out_dir),
        "operating_system": platform.platform(),
        "gpu_available": _gpu_available(),
        "memory_gb": _estimate_memory_gb(),
        "internet_allowed": True,
        "python": sys.version,
    }

    with (out_dir / "dataset_profile.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_profile, f, indent=2)
    with (out_dir / "thesis_context.json").open("w", encoding="utf-8") as f:
        json.dump(thesis_context, f, indent=2)

    return {
        "out_dir": str(out_dir),
        "row_counts": dataset_profile["row_counts"],
        "coverage": dataset_profile["coverage"],
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build thesis processed datasets from PRJ912 + PRJ896")
    parser.add_argument("--prj912", default="data/PRJ912.csv", help="Path to PRJ912 AIS CSV")
    parser.add_argument("--prj896", default="data/PRJ896.csv", help="Path to PRJ896 port-call CSV")
    parser.add_argument("--out_dir", default="data/thesis_processed", help="Output directory")
    parser.add_argument("--limit_rows", type=int, default=None, help="Optional row cap per CSV for fast dev")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    summary = build_dataset(
        prj912_path=Path(args.prj912),
        prj896_path=Path(args.prj896),
        out_dir=Path(args.out_dir),
        limit_rows=args.limit_rows,
    )
    print("Thesis data pipeline completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
