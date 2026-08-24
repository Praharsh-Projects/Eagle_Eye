"""Build KPI tables for congestion analytics and forecasting from PRJ912/PRJ896 CSVs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.kpi.reconstruct_voyages import RECONSTRUCTION_VERSION, reconstruct_voyages
from src.predict.data_prep import DEFAULT_DEST_ALIASES, normalize_destination

AIS_COLUMNS = {
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

PORT_CALL_COLUMNS = {
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


def _normalize_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if re.match(r"^\d+\.0+$", text):
        return text.split(".")[0]
    return text


def _normalize_locode(value: object) -> str:
    if value is None:
        return ""
    text = str(value).upper().strip().replace(" ", "")
    if text in {"", "NAN", "NONE"}:
        return ""
    return text


def _normalize_text(value: object, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return fallback
    return text


def _normalize_vessel_type(value: object) -> str:
    text = _normalize_text(value, fallback="unknown")
    return text.lower()


def _call_id(mmsi: object, arrival_time: object, port_key: object) -> str:
    ts = pd.to_datetime(arrival_time, errors="coerce", utc=True)
    stamp = pd.Timestamp(ts).strftime("%Y-%m-%dT%H-%M-%S") if pd.notna(ts) else "unknown"
    return f"{_normalize_id(mmsi) or 'unknown'}_{stamp}_{_normalize_text(port_key)}"


def _load_csv(csv_path: Path, limit_rows: Optional[int]) -> pd.DataFrame:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().replace('"', "")
    cols = [c.strip() for c in header.split(",")]
    dtypes = {c: "string" for c in cols}
    return pd.read_csv(
        csv_path,
        dtype=dtypes,
        low_memory=False,
        nrows=limit_rows if limit_rows and limit_rows > 0 else None,
    )


def _is_ais_schema(columns: Sequence[str]) -> bool:
    return len(AIS_COLUMNS.intersection(set(columns))) >= 8


def _is_port_schema(columns: Sequence[str]) -> bool:
    return len(PORT_CALL_COLUMNS.intersection(set(columns))) >= 6


def _prepare_ais(
    df_raw: pd.DataFrame,
    source_file: str,
    aliases: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_raw.copy()
    for col in AIS_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["mmsi"] = df["MMSI"].map(_normalize_id)
    df["timestamp"] = pd.to_datetime(df["TimePosition"], errors="coerce", utc=True)
    df["destination_norm"] = df["Destination"].map(lambda x: normalize_destination(x, aliases))
    df["vessel_type_norm"] = df["VesselType"].map(_normalize_vessel_type)

    df = df.dropna(subset=["timestamp"])
    df = df[(df["mmsi"] != "") & (df["destination_norm"] != "UNKNOWN")]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["date"] = df["timestamp"].dt.floor("D")
    df["datetime_hour"] = df["timestamp"].dt.floor("h")
    df["port_key"] = df["destination_norm"].astype(str)
    df["port_label"] = df["destination_norm"].astype(str)
    df["locode_norm"] = df["destination_norm"].str.extract(r"^([A-Z]{5})$", expand=False).fillna("")
    df["port_name_norm"] = df["destination_norm"].str.lower()
    df["source_kind"] = "ais_destination_proxy"

    day_unique = df.drop_duplicates(subset=["mmsi", "port_key", "date", "vessel_type_norm"])
    arrivals_daily = (
        day_unique.groupby(
            [
                "source_kind",
                "port_key",
                "port_label",
                "locode_norm",
                "port_name_norm",
                "date",
                "vessel_type_norm",
            ],
            dropna=False,
        )
        .agg(arrivals_vessels=("mmsi", "nunique"), arrivals_events=("mmsi", "size"))
        .reset_index()
    )
    arrivals_daily["source_file"] = source_file
    arrivals_daily["daily_distinct_vessels"] = arrivals_daily["arrivals_vessels"]
    arrivals_daily["arrival_count"] = arrivals_daily["arrivals_events"]

    hour_unique = df.drop_duplicates(subset=["mmsi", "port_key", "datetime_hour", "vessel_type_norm"])
    arrivals_hourly = (
        hour_unique.groupby(
            [
                "source_kind",
                "port_key",
                "port_label",
                "locode_norm",
                "port_name_norm",
                "datetime_hour",
                "vessel_type_norm",
            ],
            dropna=False,
        )
        .agg(arrivals_vessels=("mmsi", "nunique"), arrivals_events=("mmsi", "size"))
        .reset_index()
    )
    arrivals_hourly["source_file"] = source_file
    arrivals_hourly["daily_distinct_vessels"] = arrivals_hourly["arrivals_vessels"]
    arrivals_hourly["arrival_count"] = arrivals_hourly["arrivals_events"]

    return arrivals_daily, arrivals_hourly


def _prepare_port_calls(
    df_raw: pd.DataFrame,
    source_file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df_raw.copy()
    for col in PORT_CALL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["mmsi"] = df["vesselMMSI"].map(_normalize_id)
    df["arrival_time"] = pd.to_datetime(df["portArrival"], errors="coerce", utc=True)
    df["departure_time"] = pd.to_datetime(df["portDeparture"], errors="coerce", utc=True)
    df["port_name"] = df["portName"].map(_normalize_text)
    df["port_name_norm"] = df["port_name"].str.lower()
    df["locode_norm"] = df["portLocode"].map(_normalize_locode)
    df["vessel_type_norm"] = df["vesselType"].map(_normalize_vessel_type)
    df["imo"] = df["vesselIMO"].map(_normalize_id)
    df["vessel_name"] = df["vesselName"].map(_normalize_text)
    df["destination_arrival"] = df["vesselDestinationArrival"].map(_normalize_text)
    df["destination_departure"] = df["vesselDestinationDeparture"].map(_normalize_text)

    df = df.dropna(subset=["arrival_time"])
    df = df[df["mmsi"] != ""]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["port_key"] = np.where(
        df["locode_norm"] != "",
        df["locode_norm"],
        df["port_name_norm"].str.upper().str.replace(" ", "_", regex=False),
    )
    df["port_label"] = np.where(
        df["locode_norm"] != "",
        df["port_name"].astype(str) + " (" + df["locode_norm"].astype(str) + ")",
        df["port_name"].astype(str),
    )
    df["source_kind"] = "port_call"
    df["date"] = df["arrival_time"].dt.floor("D")
    df["datetime_hour"] = df["arrival_time"].dt.floor("h")
    df["call_id"] = [
        _call_id(mmsi, arrival_time, port_key)
        for mmsi, arrival_time, port_key in zip(df["mmsi"], df["arrival_time"], df["port_key"])
    ]

    arrivals_daily = (
        df.groupby(
            [
                "source_kind",
                "port_key",
                "port_label",
                "locode_norm",
                "port_name_norm",
                "date",
                "vessel_type_norm",
            ],
            dropna=False,
        )
        .agg(arrivals_vessels=("mmsi", "nunique"), arrivals_events=("mmsi", "size"))
        .reset_index()
    )
    arrivals_daily["source_file"] = source_file
    arrivals_daily["daily_distinct_vessels"] = arrivals_daily["arrivals_vessels"]
    arrivals_daily["arrival_count"] = arrivals_daily["arrivals_events"]

    arrivals_hourly = (
        df.groupby(
            [
                "source_kind",
                "port_key",
                "port_label",
                "locode_norm",
                "port_name_norm",
                "datetime_hour",
                "vessel_type_norm",
            ],
            dropna=False,
        )
        .agg(arrivals_vessels=("mmsi", "nunique"), arrivals_events=("mmsi", "size"))
        .reset_index()
    )
    arrivals_hourly["source_file"] = source_file
    arrivals_hourly["daily_distinct_vessels"] = arrivals_hourly["arrivals_vessels"]
    arrivals_hourly["arrival_count"] = arrivals_hourly["arrivals_events"]

    dwell = df[
        [
            "source_kind",
            "port_key",
            "port_label",
            "locode_norm",
            "port_name_norm",
            "mmsi",
            "imo",
            "vessel_name",
            "vessel_type_norm",
            "arrival_time",
            "departure_time",
            "call_id",
            "destination_arrival",
            "destination_departure",
        ]
    ].copy()
    dwell = dwell.dropna(subset=["arrival_time", "departure_time"])
    dwell["dwell_minutes"] = (dwell["departure_time"] - dwell["arrival_time"]).dt.total_seconds() / 60.0
    dwell = dwell[(dwell["dwell_minutes"] > 0) & (dwell["dwell_minutes"] <= 60 * 24 * 45)]
    dwell["arrival_date"] = dwell["arrival_time"].dt.floor("D")
    dwell["source_file"] = source_file

    return arrivals_daily, arrivals_hourly, dwell


def _build_occupancy_hourly(dwell: pd.DataFrame, max_expanded_rows: int = 3_500_000) -> pd.DataFrame:
    if dwell.empty:
        return pd.DataFrame()

    minutes = (dwell["departure_time"] - dwell["arrival_time"]).dt.total_seconds() / 60.0
    estimated_rows = int(np.ceil((minutes / 60.0).clip(lower=0, upper=24 * 30)).sum()) + len(dwell)
    if estimated_rows > max_expanded_rows:
        return pd.DataFrame()

    expanded_rows: List[Dict[str, object]] = []
    for row in dwell.itertuples(index=False):
        start = pd.Timestamp(row.arrival_time).floor("h")
        end = pd.Timestamp(row.departure_time).floor("h")
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        hours = pd.date_range(start, end, freq="h")
        for dt_hour in hours:
            expanded_rows.append(
                {
                    "source_kind": "port_call",
                    "port_key": row.port_key,
                    "port_label": row.port_label,
                    "locode_norm": row.locode_norm,
                    "port_name_norm": row.port_name_norm,
                    "datetime_hour": dt_hour,
                    "mmsi": row.mmsi,
                }
            )

    if not expanded_rows:
        return pd.DataFrame()

    expanded = pd.DataFrame(expanded_rows)
    occupancy = (
        expanded.groupby(
            ["source_kind", "port_key", "port_label", "locode_norm", "port_name_norm", "datetime_hour"],
            dropna=False,
        )
        .agg(occupancy_vessels=("mmsi", "nunique"), occupancy_events=("mmsi", "size"))
        .reset_index()
    )
    return occupancy


def _build_congestion_daily(arrivals_daily: pd.DataFrame, dwell: pd.DataFrame) -> pd.DataFrame:
    if arrivals_daily.empty:
        return pd.DataFrame()

    arrivals_total = (
        arrivals_daily.groupby(
            ["source_kind", "port_key", "port_label", "locode_norm", "port_name_norm", "date"],
            dropna=False,
        )
        .agg(arrivals_vessels=("arrivals_vessels", "sum"), arrivals_events=("arrivals_events", "sum"))
        .reset_index()
    )

    pc = arrivals_total[arrivals_total["source_kind"] == "port_call"].drop(columns=["source_kind"])
    ais = arrivals_total[arrivals_total["source_kind"] != "port_call"].drop(columns=["source_kind"])

    merged = ais.merge(pc, on=["port_key", "date"], how="outer", suffixes=("_ais", "_pc"))
    merged["source_kind"] = np.where(merged["arrivals_vessels_pc"].notna(), "port_call", "ais_destination_proxy")

    for col in ["port_label", "locode_norm", "port_name_norm", "arrivals_vessels", "arrivals_events"]:
        merged[col] = merged.get(f"{col}_pc").combine_first(merged.get(f"{col}_ais"))

    keep_cols = [
        "source_kind",
        "port_key",
        "port_label",
        "locode_norm",
        "port_name_norm",
        "date",
        "arrivals_vessels",
        "arrivals_events",
    ]
    out = merged[keep_cols].copy()
    out["daily_distinct_vessels"] = out["arrivals_vessels"]
    out["arrival_count"] = out["arrivals_events"]

    if dwell.empty:
        out["median_dwell_minutes"] = np.nan
        out["has_dwell"] = False
    else:
        dwell_daily = (
            dwell.groupby(["port_key", "arrival_date"], dropna=False)
            .agg(median_dwell_minutes=("dwell_minutes", "median"))
            .reset_index()
            .rename(columns={"arrival_date": "date"})
        )
        out = out.merge(dwell_daily, on=["port_key", "date"], how="left")
        out["has_dwell"] = out["median_dwell_minutes"].notna()

    out["day_of_week"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.day_name()

    def _robust_baseline(value_col: str) -> pd.Series:
        values = pd.to_numeric(out[value_col], errors="coerce")
        by_port_weekday = values.groupby([out["port_key"], out["day_of_week"]], dropna=False).transform("median")
        by_port = values.groupby(out["port_key"], dropna=False).transform("median")
        global_median = values[values > 0].median()
        if pd.isna(global_median) or float(global_median) <= 0:
            global_median = 1.0
        return by_port_weekday.where(by_port_weekday > 0).fillna(by_port.where(by_port > 0)).fillna(global_median)

    arrivals_baseline = _robust_baseline("arrivals_vessels")
    dwell_baseline = _robust_baseline("median_dwell_minutes")
    out["arrivals_baseline_median"] = arrivals_baseline
    out["dwell_baseline_median"] = dwell_baseline.where(out["has_dwell"])
    out["arrivals_ratio"] = (
        pd.to_numeric(out["arrivals_vessels"], errors="coerce") / arrivals_baseline.replace(0, np.nan)
    ).clip(lower=0.0, upper=5.0)
    out["dwell_ratio"] = (
        pd.to_numeric(out["median_dwell_minutes"], errors="coerce") / dwell_baseline.replace(0, np.nan)
    ).clip(lower=0.0, upper=5.0)

    full_mask = out["has_dwell"].fillna(False) & out["dwell_ratio"].notna()
    out["congestion_index"] = out["arrivals_ratio"].fillna(0.0)
    out.loc[full_mask, "congestion_index"] = (
        0.60 * out.loc[full_mask, "arrivals_ratio"].fillna(0.0)
        + 0.40 * out.loc[full_mask, "dwell_ratio"].fillna(0.0)
    )
    out["congestion_index"] = out["congestion_index"].clip(lower=0.0, upper=5.0)
    out["pressure_kind"] = np.where(full_mask, "full", "partial_arrivals_proxy")
    out["pressure_version"] = "pressure_v2"
    out["is_cross_scope_comparable"] = full_mask

    out = out.sort_values(["port_key", "date"]).reset_index(drop=True)
    return out


def _port_catalog(arrivals_daily: pd.DataFrame) -> pd.DataFrame:
    if arrivals_daily.empty:
        return pd.DataFrame()
    cat = (
        arrivals_daily.groupby(["port_key", "port_label", "locode_norm", "port_name_norm", "source_kind"], dropna=False)
        .agg(
            first_seen=("date", "min"),
            last_seen=("date", "max"),
            arrivals_total=("arrivals_vessels", "sum"),
        )
        .reset_index()
        .sort_values("arrivals_total", ascending=False)
    )
    return cat


def build_kpis(
    csv_paths: Sequence[Path],
    out_dir: Path,
    limit_rows: Optional[int] = None,
) -> Dict[str, object]:
    aliases = dict(DEFAULT_DEST_ALIASES)
    arrivals_daily_parts: List[pd.DataFrame] = []
    arrivals_hourly_parts: List[pd.DataFrame] = []
    dwell_parts: List[pd.DataFrame] = []

    stats: Dict[str, int] = {
        "input_files": len(csv_paths),
        "ais_rows": 0,
        "port_call_rows": 0,
    }

    for csv_path in csv_paths:
        raw = _load_csv(csv_path, limit_rows=limit_rows)
        cols = list(raw.columns)
        if _is_ais_schema(cols):
            daily, hourly = _prepare_ais(raw, source_file=csv_path.name, aliases=aliases)
            if not daily.empty:
                arrivals_daily_parts.append(daily)
                arrivals_hourly_parts.append(hourly)
                stats["ais_rows"] += int(daily["arrivals_events"].sum())
            continue

        if _is_port_schema(cols):
            daily, hourly, dwell = _prepare_port_calls(raw, source_file=csv_path.name)
            if not daily.empty:
                arrivals_daily_parts.append(daily)
                arrivals_hourly_parts.append(hourly)
                dwell_parts.append(dwell)
                stats["port_call_rows"] += int(daily["arrivals_events"].sum())
            continue

        print(f"Skipping unsupported schema: {csv_path}")

    if not arrivals_daily_parts:
        raise RuntimeError("No supported AIS or port-call rows found for KPI build.")

    out_dir.mkdir(parents=True, exist_ok=True)

    arrivals_daily = pd.concat(arrivals_daily_parts, ignore_index=True, sort=False)
    arrivals_hourly = pd.concat(arrivals_hourly_parts, ignore_index=True, sort=False)
    dwell = pd.concat(dwell_parts, ignore_index=True, sort=False) if dwell_parts else pd.DataFrame()
    occupancy = _build_occupancy_hourly(dwell)
    congestion = _build_congestion_daily(arrivals_daily, dwell)
    ports = _port_catalog(arrivals_daily)
    voyages = reconstruct_voyages(dwell, max_gap_days=30) if not dwell.empty else pd.DataFrame()

    arrivals_daily_path = out_dir / "arrivals_daily.parquet"
    arrivals_hourly_path = out_dir / "arrivals_hourly.parquet"
    dwell_path = out_dir / "dwell_time.parquet"
    occupancy_path = out_dir / "occupancy_hourly.parquet"
    congestion_path = out_dir / "congestion_daily.parquet"
    ports_path = out_dir / "port_catalog.parquet"
    voyages_path = out_dir / "voyages.parquet"
    capabilities_path = out_dir / "kpi_capabilities.json"

    arrivals_daily.to_parquet(arrivals_daily_path, index=False)
    arrivals_hourly.to_parquet(arrivals_hourly_path, index=False)
    dwell.to_parquet(dwell_path, index=False)
    occupancy.to_parquet(occupancy_path, index=False)
    congestion.to_parquet(congestion_path, index=False)
    ports.to_parquet(ports_path, index=False)
    voyages.to_parquet(voyages_path, index=False)

    date_min = arrivals_daily["date"].min() if not arrivals_daily.empty else pd.NaT
    date_max = arrivals_daily["date"].max() if not arrivals_daily.empty else pd.NaT
    capabilities = {
        "has_port_calls": bool((arrivals_daily["source_kind"] == "port_call").any()),
        "has_ais_destination_proxy": bool((arrivals_daily["source_kind"] == "ais_destination_proxy").any()),
        "has_dwell_time": not dwell.empty,
        "has_occupancy_hourly": not occupancy.empty,
        "has_reconstructed_voyages": not voyages.empty,
        "pressure_version": "pressure_v2",
        "voyage_reconstruction_version": RECONSTRUCTION_VERSION if not voyages.empty else None,
        "date_min": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else None,
        "date_max": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else None,
        "ports": int(ports["port_key"].nunique()) if not ports.empty else 0,
    }
    with capabilities_path.open("w", encoding="utf-8") as f:
        json.dump(capabilities, f, indent=2)

    summary = {
        "paths": {
            "arrivals_daily": str(arrivals_daily_path),
            "arrivals_hourly": str(arrivals_hourly_path),
            "dwell_time": str(dwell_path),
            "occupancy_hourly": str(occupancy_path),
            "congestion_daily": str(congestion_path),
            "port_catalog": str(ports_path),
            "voyages": str(voyages_path),
            "kpi_capabilities": str(capabilities_path),
        },
        "stats": {
            **stats,
            "arrivals_daily_rows": int(len(arrivals_daily)),
            "arrivals_hourly_rows": int(len(arrivals_hourly)),
            "dwell_rows": int(len(dwell)),
            "occupancy_rows": int(len(occupancy)),
            "congestion_rows": int(len(congestion)),
            "ports": int(ports["port_key"].nunique()) if not ports.empty else 0,
            "voyages": int(len(voyages)),
        },
        "capabilities": capabilities,
    }
    return summary


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build KPI tables for analytics + forecasting.")
    parser.add_argument("--traffic_csv", default=None, help="Primary CSV path (PRJ912/PRJ896)")
    parser.add_argument("--traffic_csvs", nargs="*", default=None, help="Additional CSV paths")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory for KPI tables")
    parser.add_argument("--limit_rows", type=int, default=None, help="Optional row limit per input CSV")
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    csv_paths: List[Path] = []
    if args.traffic_csv:
        csv_paths.append(Path(args.traffic_csv))
    if args.traffic_csvs:
        csv_paths.extend(Path(p) for p in args.traffic_csvs)

    if not csv_paths:
        default_csv = Path("data/PRJ912.csv")
        if default_csv.exists():
            csv_paths = [default_csv]
        else:
            raise RuntimeError("Provide --traffic_csv and/or --traffic_csvs")

    summary = build_kpis(csv_paths=csv_paths, out_dir=Path(args.out_dir), limit_rows=args.limit_rows)
    print("KPI build completed")
    print(json.dumps(summary["stats"], indent=2))
    print(json.dumps(summary["capabilities"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
