"""Reconstruct port-to-port voyages from consecutive vessel calls.

The materialized table is deliberately derived only from structured port-call
rows.  A voyage starts at a call's departure timestamp and ends at the first
subsequent arrival for the same MMSI.  Pairs with non-positive travel time or a
gap longer than the configured limit are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.parquet_io import read_parquet_safely


RECONSTRUCTION_VERSION = "voyage_reconstruction_v1"
DEFAULT_MAX_GAP_DAYS = 30


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _timestamp_token(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return "unknown"
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H-%M-%S")


def _call_id(row: pd.Series) -> str:
    existing = _clean_text(row.get("call_id"))
    if existing:
        return existing
    return "_".join(
        (
            _clean_text(row.get("mmsi")) or "unknown",
            _timestamp_token(row.get("arrival_time")),
            _clean_text(row.get("port_key")) or "unknown",
        )
    )


def _voyage_id(origin_call_id: str, destination_call_id: str) -> str:
    digest = hashlib.sha256(f"{origin_call_id}|{destination_call_id}".encode("utf-8")).hexdigest()[:20]
    return f"voyage_{digest}"


def _event_metadata(events_path: Optional[Path]) -> pd.DataFrame:
    """Return port-call identifiers and optional vessel metadata."""
    if events_path is None or not events_path.exists():
        return pd.DataFrame()
    wanted = [
        "event_kind",
        "stable_id",
        "mmsi",
        "imo",
        "name",
        "vessel_type_norm",
        "timestamp_full",
        "locode_norm",
        "port_name",
        "source_file",
    ]
    events = read_parquet_safely(events_path, columns=wanted)
    if events.empty or "event_kind" not in events.columns:
        return pd.DataFrame()
    events = events[events["event_kind"].fillna("").astype(str) == "port_call"].copy()
    if events.empty:
        return pd.DataFrame()
    events["mmsi"] = events["mmsi"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    events["arrival_time"] = pd.to_datetime(events["timestamp_full"], errors="coerce", utc=True)
    events["port_key"] = events["locode_norm"].fillna("").astype(str).str.strip().str.upper()
    events = events.dropna(subset=["arrival_time"])
    events = events[events["mmsi"].ne("") & events["port_key"].ne("")]
    return events.drop_duplicates(["mmsi", "arrival_time", "port_key"], keep="first")


def reconstruct_voyages(
    calls: pd.DataFrame,
    *,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
    event_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Pair each departure with the next valid arrival for the same MMSI."""
    required = {"mmsi", "port_key", "arrival_time", "departure_time"}
    missing = required.difference(calls.columns)
    if missing:
        raise ValueError(f"Cannot reconstruct voyages; missing columns: {', '.join(sorted(missing))}")
    if max_gap_days < 1:
        raise ValueError("max_gap_days must be at least 1")

    work = calls.copy()
    work["mmsi"] = work["mmsi"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    work["port_key"] = work["port_key"].fillna("").astype(str).str.strip().str.upper()
    work["arrival_time"] = pd.to_datetime(work["arrival_time"], errors="coerce", utc=True)
    work["departure_time"] = pd.to_datetime(work["departure_time"], errors="coerce", utc=True)
    work = work.dropna(subset=["arrival_time", "departure_time"])
    work = work[work["mmsi"].ne("") & work["port_key"].ne("")]
    work = work[work["departure_time"] >= work["arrival_time"]]
    work = work.drop_duplicates(["mmsi", "port_key", "arrival_time", "departure_time"], keep="first")
    if work.empty:
        return pd.DataFrame()

    # Enrich old dwell artifacts with stable event IDs and vessel metadata when
    # the canonical event store is available.
    meta = event_metadata if event_metadata is not None else pd.DataFrame()
    if not meta.empty:
        meta_cols = [
            c
            for c in [
                "mmsi",
                "arrival_time",
                "port_key",
                "stable_id",
                "imo",
                "name",
                "source_file",
                "vessel_type_norm",
            ]
            if c in meta.columns
        ]
        meta = meta[meta_cols].copy()
        rename = {
            "stable_id": "event_stable_id",
            "imo": "event_imo",
            "name": "event_vessel_name",
            "source_file": "event_source_file",
            "vessel_type_norm": "event_vessel_type_norm",
        }
        meta = meta.rename(columns=rename)
        work = work.merge(meta, on=["mmsi", "arrival_time", "port_key"], how="left")

    work["call_id"] = work.apply(_call_id, axis=1)
    if "event_stable_id" in work.columns:
        work["call_id"] = work["event_stable_id"].fillna(work["call_id"]).astype(str)
    work = work.sort_values(["mmsi", "arrival_time", "departure_time", "call_id"]).reset_index(drop=True)

    max_gap_hours = float(max_gap_days * 24)
    records: list[Dict[str, Any]] = []
    for _, vessel_calls in work.groupby("mmsi", sort=False):
        vessel_calls = vessel_calls.sort_values(["arrival_time", "departure_time", "call_id"]).reset_index(drop=True)
        arrivals_ns = vessel_calls["arrival_time"].astype("int64").to_numpy()
        departures_ns = vessel_calls["departure_time"].astype("int64").to_numpy()
        next_indexes = np.searchsorted(arrivals_ns, departures_ns, side="right")

        for origin_index, destination_index in enumerate(next_indexes):
            if destination_index >= len(vessel_calls):
                continue
            origin = vessel_calls.iloc[origin_index]
            destination = vessel_calls.iloc[int(destination_index)]
            duration_h = float(
                (pd.Timestamp(destination["arrival_time"]) - pd.Timestamp(origin["departure_time"])).total_seconds()
                / 3600.0
            )
            if not (0.0 < duration_h <= max_gap_hours):
                continue

            origin_call_id = _clean_text(origin.get("call_id"))
            destination_call_id = _clean_text(destination.get("call_id"))
            record: Dict[str, Any] = {
                "voyage_id": _voyage_id(origin_call_id, destination_call_id),
                "mmsi": _clean_text(origin.get("mmsi")),
                "imo": _clean_text(origin.get("imo")) or _clean_text(origin.get("event_imo")),
                "vessel_name": _clean_text(origin.get("vessel_name")) or _clean_text(origin.get("event_vessel_name")),
                "vessel_type_norm": _clean_text(origin.get("vessel_type_norm"))
                or _clean_text(origin.get("event_vessel_type_norm")),
                "origin_call_id": origin_call_id,
                "destination_call_id": destination_call_id,
                "origin_port_key": _clean_text(origin.get("port_key")),
                "origin_port_label": _clean_text(origin.get("port_label")) or _clean_text(origin.get("port_key")),
                "origin_locode_norm": _clean_text(origin.get("locode_norm")),
                "destination_port_key": _clean_text(destination.get("port_key")),
                "destination_port_label": _clean_text(destination.get("port_label"))
                or _clean_text(destination.get("port_key")),
                "destination_locode_norm": _clean_text(destination.get("locode_norm")),
                "departure_time": pd.Timestamp(origin["departure_time"]),
                "arrival_time": pd.Timestamp(destination["arrival_time"]),
                "duration_h": duration_h,
                "origin_source_file": _clean_text(origin.get("source_file"))
                or _clean_text(origin.get("event_source_file")),
                "destination_source_file": _clean_text(destination.get("source_file"))
                or _clean_text(destination.get("event_source_file")),
                "source_kind": "reconstructed_port_calls",
                "reconstruction_version": RECONSTRUCTION_VERSION,
                "max_gap_days": int(max_gap_days),
            }
            records.append(record)

    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates("voyage_id", keep="first")
        .sort_values(["departure_time", "mmsi", "voyage_id"])
        .reset_index(drop=True)
    )


def materialize_voyages(
    processed_dir: str | Path = "data/processed",
    *,
    out_path: str | Path | None = None,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> Dict[str, Any]:
    processed = Path(processed_dir)
    dwell_path = processed / "dwell_time.parquet"
    if not dwell_path.exists():
        raise FileNotFoundError(f"Missing structured port-call table: {dwell_path}")
    calls = read_parquet_safely(dwell_path)
    metadata = _event_metadata(processed / "events.parquet")
    voyages = reconstruct_voyages(calls, max_gap_days=max_gap_days, event_metadata=metadata)
    destination = Path(out_path) if out_path is not None else processed / "voyages.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    voyages.to_parquet(destination, index=False)

    route_count = (
        int(voyages[["origin_port_key", "destination_port_key"]].drop_duplicates().shape[0])
        if not voyages.empty
        else 0
    )
    return {
        "path": str(destination),
        "rows": int(len(voyages)),
        "vessels": int(voyages["mmsi"].nunique()) if not voyages.empty else 0,
        "routes": route_count,
        "date_min": voyages["departure_time"].min().strftime("%Y-%m-%d") if not voyages.empty else None,
        "date_max": voyages["arrival_time"].max().strftime("%Y-%m-%d") if not voyages.empty else None,
        "max_gap_days": int(max_gap_days),
        "reconstruction_version": RECONSTRUCTION_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct port-to-port voyages from structured calls.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-gap-days", type=int, default=DEFAULT_MAX_GAP_DAYS)
    args = parser.parse_args()
    summary = materialize_voyages(
        processed_dir=args.processed_dir,
        out_path=args.out,
        max_gap_days=args.max_gap_days,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
