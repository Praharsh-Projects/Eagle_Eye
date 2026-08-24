"""AIS anomaly helpers for runtime jump detection from event rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.parquet_io import read_parquet_safely

from src.utils.serialization import normalize_identifier


_MISSING_PORT_TOKENS = {"", "NA", "NAN", "NONE", "NULL", "UNK", "UNKNOWN"}


def _normalize_locode(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    token = "".join(ch for ch in str(value).strip().upper() if ch.isalnum())
    return "" if token in _MISSING_PORT_TOKENS else token


def _apply_observed_port_scope(
    frame: pd.DataFrame,
    locode: Optional[str],
) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """Apply only an observed row-level port code, never a destination proxy.

    Returns ``(filtered, field_used, error)``. A requested scope without a
    populated observed port/LOCODE column is rejected explicitly.
    """

    requested = _normalize_locode(locode)
    if not requested:
        return frame, None, None

    for field in ("locode_norm", "locode", "port_key"):
        if field not in frame.columns:
            continue
        normalized = frame[field].map(_normalize_locode)
        if not normalized.ne("").any():
            continue
        return frame[normalized == requested].copy(), field, None

    return (
        frame.iloc[0:0].copy(),
        None,
        (
            f"AIS position rows do not contain a populated observed port/LOCODE field for {requested}. "
            "Destination text cannot establish an observed port location."
        ),
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def _first_present(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "nat"}:
            return text
    return None


def detect_sudden_jump_events_from_parquet(
    events_path: str | Path,
    mmsi: Optional[str] = None,
    locode: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_minutes: int = 30,
    km_threshold: float = 20.0,
    speed_kn_threshold: float = 40.0,
    min_distance_km_for_speed_rule: float = 5.0,
    limit: int = 200,
) -> Dict[str, Any]:
    path = Path(events_path)
    if not path.exists():
        return {"count": 0, "events": [], "reason": f"Events file missing: {path}"}

    df = read_parquet_safely(path)
    if df.empty:
        return {"count": 0, "events": [], "reason": "Events parquet is empty."}

    work = df.copy()
    if "event_kind" in work.columns:
        work = work[work["event_kind"].astype(str) == "ais_position"]
    work, port_scope_field, port_scope_error = _apply_observed_port_scope(work, locode)
    if port_scope_error:
        return {
            "count": 0,
            "events": [],
            "reason": port_scope_error,
            "scope_status": "unsupported",
            "scope_applied": False,
            "requested_locode": _normalize_locode(locode),
        }
    scope_metadata = {
        "scope_status": "applied" if port_scope_field else "not_requested",
        "scope_applied": bool(port_scope_field),
        "scope_field": port_scope_field,
        "requested_locode": _normalize_locode(locode) or None,
    }
    if "mmsi" in work.columns:
        work["mmsi"] = work["mmsi"].astype(str).map(normalize_identifier)
    if mmsi and "mmsi" in work.columns:
        work = work[work["mmsi"] == normalize_identifier(str(mmsi).strip())]

    timestamp_source = "timestamp_full" if "timestamp_full" in work.columns else "timestamp"
    if timestamp_source not in work.columns:
        return {
            "count": 0,
            "events": [],
            "reason": "No timestamp column available for jump detection.",
            **scope_metadata,
        }

    work["timestamp_dt"] = pd.to_datetime(work[timestamp_source], errors="coerce", utc=True)
    if date_from:
        work = work[work["timestamp_dt"] >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        work = work[work["timestamp_dt"] <= pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if "latitude" not in work.columns or "longitude" not in work.columns:
        return {
            "count": 0,
            "events": [],
            "reason": "Latitude/longitude columns missing from events parquet.",
            **scope_metadata,
        }

    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work = work.dropna(subset=["timestamp_dt", "latitude", "longitude"])
    if "mmsi" not in work.columns:
        work["mmsi"] = None
    work = work.dropna(subset=["mmsi"])
    if work.empty:
        return {
            "count": 0,
            "events": [],
            "reason": "No AIS points available after filtering.",
            **scope_metadata,
        }

    work = work.sort_values(["mmsi", "timestamp_dt"])
    events: List[Dict[str, Any]] = []
    for _, group in work.groupby("mmsi"):
        g = group.copy()
        g["prev_timestamp_dt"] = g["timestamp_dt"].shift(1)
        g["prev_latitude"] = g["latitude"].shift(1)
        g["prev_longitude"] = g["longitude"].shift(1)
        dt_minutes = (g["timestamp_dt"] - g["prev_timestamp_dt"]).dt.total_seconds() / 60.0
        dlat = g["latitude"] - g["prev_latitude"]
        dlon = g["longitude"] - g["prev_longitude"]
        dist_km = ((dlat * 111.0) ** 2 + (dlon * 111.0) ** 2) ** 0.5
        implied_speed_kn = ((dist_km / (dt_minutes / 60.0)) / 1.852).replace(
            [float("inf"), float("-inf")], pd.NA
        )
        jump_mask = (
            (dt_minutes > 0)
            & (dt_minutes <= max_minutes)
            & (
                (dist_km >= km_threshold)
                | (
                    (dist_km >= min_distance_km_for_speed_rule)
                    & (implied_speed_kn >= speed_kn_threshold)
                )
            )
        )
        flagged = g[jump_mask].copy()
        if flagged.empty:
            continue
        flagged["dt_minutes"] = dt_minutes.loc[flagged.index].astype(float)
        flagged["distance_km"] = dist_km.loc[flagged.index].astype(float)
        flagged["implied_speed_kn"] = implied_speed_kn.loc[flagged.index].astype(float)
        flagged["trigger_rule"] = flagged.apply(
            lambda row: (
                "distance_threshold"
                if float(row.get("distance_km", 0.0)) >= km_threshold
                else "speed_threshold"
            ),
            axis=1,
        )
        for _, row in flagged.iterrows():
            events.append(
                {
                    "stable_id": str(row.get("stable_id", "")),
                    "mmsi": str(row.get("mmsi", "")),
                    "timestamp_full": row["timestamp_dt"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "latitude": _safe_float(row.get("latitude")),
                    "longitude": _safe_float(row.get("longitude")),
                    "prev_latitude": _safe_float(row.get("prev_latitude")),
                    "prev_longitude": _safe_float(row.get("prev_longitude")),
                    "dt_minutes": float(row.get("dt_minutes", 0.0)),
                    "distance_km": float(row.get("distance_km", 0.0)),
                    "implied_speed_kn": float(row.get("implied_speed_kn", 0.0)),
                    "trigger_rule": str(row.get("trigger_rule", "")),
                    "port": _first_present(
                        row.get("locode_norm"),
                        row.get("locode"),
                        row.get("destination_norm"),
                        row.get("port_name_norm"),
                    ),
                }
            )
            if len(events) >= limit:
                break
        if len(events) >= limit:
            break

    return {
        "count": len(events),
        "events": events,
        "reason": "Computed from row-level AIS events parquet.",
        **scope_metadata,
    }
