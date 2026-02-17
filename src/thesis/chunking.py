"""Chunking strategies for structured maritime CSV data.

A: Event row chunks
B: Aggregated port-day chunks
C: Hybrid temporal window chunks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    text = str(v).strip().lower()
    return text in {"1", "true", "yes"}


def _safe_num(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in meta.items():
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[key] = val
        elif isinstance(val, pd.Timestamp):
            out[key] = val.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            out[key] = str(val)
    return out


def _event_chunk_from_port_call(row: pd.Series) -> Tuple[str, Dict[str, Any], str]:
    arrival = pd.to_datetime(row.get("arrival_time"), errors="coerce", utc=True)
    departure = pd.to_datetime(row.get("departure_time"), errors="coerce", utc=True)
    arrival_text = arrival.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(arrival) else "unknown time"
    departure_text = departure.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(departure) else "unknown"

    dwell_h = _safe_num(row.get("dwell_hours"))
    dwell_text = f"{dwell_h:.2f} hours" if pd.notna(dwell_h) else "unknown"

    anomaly_short = _to_bool(row.get("anomaly_short_dwell"))
    anomaly_long = _to_bool(row.get("anomaly_long_dwell"))
    anomaly_desc = "none"
    if anomaly_short:
        anomaly_desc = "short dwell"
    if anomaly_long:
        anomaly_desc = "long dwell"

    text = (
        f"Vessel {row.get('vessel_name', 'unknown')} (MMSI {row.get('mmsi', 'unknown')}, IMO {row.get('imo', 'unknown')}) "
        f"arrived at {row.get('port_key', 'unknown')} on {arrival_text} from {row.get('destination_arrival_norm', 'UNKNOWN')}. "
        f"Departure {departure_text} to {row.get('destination_departure_norm', 'UNKNOWN')}. "
        f"Dwell time {dwell_text}. Vessel type: {row.get('vessel_type_norm', 'unknown')}. "
        f"Anomaly flag: {anomaly_desc}."
    )

    date = arrival.strftime("%Y-%m-%d") if pd.notna(arrival) else "unknown"
    metadata = {
        "strategy": "A",
        "chunk_type": "event",
        "event_type": "port_call",
        "source": "PRJ896",
        "port": str(row.get("port_key", "")),
        "locode": str(row.get("locode_norm", "")),
        "date": date,
        "vessel_type": str(row.get("vessel_type_norm", "unknown")),
        "anomaly": bool(_to_bool(row.get("anomaly_flag"))),
        "anomaly_kind": anomaly_desc,
        "mmsi": str(row.get("mmsi", "")),
        "event_id": str(row.get("event_id", "")),
    }
    chunk_id = f"A_PC_{row.get('event_id', '')}"
    return text, _sanitize_metadata(metadata), chunk_id


def _event_chunk_from_ais(row: pd.Series) -> Tuple[str, Dict[str, Any], str]:
    ts = pd.to_datetime(row.get("timestamp"), errors="coerce", utc=True)
    ts_text = ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(ts) else "unknown time"
    speed = _safe_num(row.get("speed_kn"))
    course = _safe_num(row.get("course_deg"))
    heading = _safe_num(row.get("heading_deg"))
    lat = _safe_num(row.get("latitude"))
    lon = _safe_num(row.get("longitude"))

    text = (
        f"At {ts_text}, vessel {row.get('vessel_name', 'unknown')} (MMSI {row.get('mmsi', 'unknown')}, IMO {row.get('imo', 'unknown')}, "
        f"flag {row.get('flag', 'unknown')}, type {row.get('vessel_type_norm', 'unknown')}) was at ({lat:.5f}, {lon:.5f}) "
        f"moving at {speed:.1f} kn, course {course:.1f} deg, heading {heading:.1f} deg. "
        f"Nav status: {row.get('nav_status', 'unknown')}. Destination: {row.get('destination_norm', 'UNKNOWN')}. "
        f"Anomaly jump flag: {'true' if _to_bool(row.get('anomaly_jump_30m')) else 'false'}."
    )

    date = ts.strftime("%Y-%m-%d") if pd.notna(ts) else "unknown"
    metadata = {
        "strategy": "A",
        "chunk_type": "event",
        "event_type": "ais_position",
        "source": "PRJ912",
        "port": str(row.get("destination_norm", "UNKNOWN")),
        "locode": "",
        "date": date,
        "vessel_type": str(row.get("vessel_type_norm", "unknown")),
        "anomaly": bool(_to_bool(row.get("anomaly_jump_30m"))),
        "anomaly_kind": "position_jump" if _to_bool(row.get("anomaly_jump_30m")) else "none",
        "mmsi": str(row.get("mmsi", "")),
        "event_id": str(row.get("event_id", "")),
    }
    chunk_id = f"A_AIS_{row.get('event_id', '')}"
    return text, _sanitize_metadata(metadata), chunk_id


def _strategy_a_chunks(port_calls: pd.DataFrame, ais: pd.DataFrame) -> Iterator[Dict[str, Any]]:
    for row in port_calls.itertuples(index=False):
        s = pd.Series(row._asdict())
        text, metadata, chunk_id = _event_chunk_from_port_call(s)
        yield {"id": chunk_id, "text": text, "metadata": metadata}

    for row in ais.itertuples(index=False):
        s = pd.Series(row._asdict())
        text, metadata, chunk_id = _event_chunk_from_ais(s)
        yield {"id": chunk_id, "text": text, "metadata": metadata}


def _strategy_b_chunks(port_day: pd.DataFrame, ais_daily: pd.DataFrame) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    for row in port_day.itertuples(index=False):
        s = pd.Series(row._asdict())
        date = pd.to_datetime(s.get("date"), errors="coerce", utc=True)
        date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else "unknown"

        avg_dwell = _safe_num(s.get("avg_dwell_hours"))
        med_dwell = _safe_num(s.get("median_dwell_hours"))
        anomaly_count = int(_safe_num(s.get("anomaly_count"), default=0))

        text = (
            f"On {date_str}, port {s.get('port_key', 'unknown')} recorded {int(s.get('arrivals_vessels', 0))} vessel arrivals "
            f"({int(s.get('arrivals_events', 0))} events). Average dwell time {avg_dwell:.2f} hours; "
            f"median dwell {med_dwell:.2f} hours. {anomaly_count} vessels flagged for unusual dwell. "
            f"Most frequent origin: {s.get('most_frequent_origin', 'UNKNOWN')}. "
            f"Most frequent next destination: {s.get('most_frequent_next_destination', 'UNKNOWN')}."
        )

        metadata = {
            "strategy": "B",
            "chunk_type": "port_day_summary",
            "event_type": "aggregated",
            "source": str(s.get("source", "PRJ896")),
            "port": str(s.get("port_key", "")),
            "locode": str(s.get("locode_norm", "")),
            "date": date_str,
            "vessel_type": "all",
            "anomaly": anomaly_count > 0,
            "anomaly_kind": "dwell_outlier" if anomaly_count > 0 else "none",
            "summary": True,
            "event_id": f"{s.get('port_key', '')}_{date_str}",
        }
        chunk_id = f"B_SUM_{s.get('port_key', '')}_{date_str}"
        chunks.append({"id": chunk_id, "text": text, "metadata": _sanitize_metadata(metadata)})

    # Add AIS-only aggregated day chunks for destinations that may not appear in port_calls.
    if not ais_daily.empty:
        ais_day = (
            ais_daily.groupby(["port_key", "date"], dropna=False)
            .agg(arrivals_vessels=("arrivals_vessels", "sum"), arrivals_events=("arrivals_events", "sum"))
            .reset_index()
        )
        existing_keys = {(c["metadata"]["port"], c["metadata"]["date"]) for c in chunks}

        for row in ais_day.itertuples(index=False):
            s = pd.Series(row._asdict())
            date = pd.to_datetime(s.get("date"), errors="coerce", utc=True)
            date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else "unknown"
            key = (str(s.get("port_key", "")), date_str)
            if key in existing_keys:
                continue
            text = (
                f"On {date_str}, AIS destination proxy for {s.get('port_key', 'unknown')} recorded "
                f"{int(s.get('arrivals_vessels', 0))} distinct vessels and {int(s.get('arrivals_events', 0))} telemetry events. "
                "Dwell-time fields are unavailable in this proxy summary."
            )
            metadata = {
                "strategy": "B",
                "chunk_type": "port_day_summary",
                "event_type": "aggregated",
                "source": "PRJ912",
                "port": str(s.get("port_key", "")),
                "locode": "",
                "date": date_str,
                "vessel_type": "all",
                "anomaly": False,
                "anomaly_kind": "none",
                "summary": True,
                "event_id": f"{s.get('port_key', '')}_{date_str}_ais",
            }
            chunk_id = f"B_AIS_{s.get('port_key', '')}_{date_str}"
            chunks.append({"id": chunk_id, "text": text, "metadata": _sanitize_metadata(metadata)})

    return chunks


def _strategy_c_chunks(port_calls: pd.DataFrame, window_size: int = 5) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    if port_calls.empty:
        return chunks

    work = port_calls.sort_values(["port_key", "arrival_time"]).reset_index(drop=True)
    for port_key, grp in work.groupby("port_key", dropna=False):
        grp = grp.reset_index(drop=True)
        for start_idx in range(0, len(grp), window_size):
            window = grp.iloc[start_idx : start_idx + window_size]
            if window.empty:
                continue

            start_ts = pd.to_datetime(window["arrival_time"].min(), errors="coerce", utc=True)
            end_ts = pd.to_datetime(window["arrival_time"].max(), errors="coerce", utc=True)
            start_str = start_ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(start_ts) else "unknown"
            end_str = end_ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(end_ts) else "unknown"
            date_str = start_ts.strftime("%Y-%m-%d") if pd.notna(start_ts) else "unknown"

            arrivals = int(window["event_id"].size)
            unique_vessels = int(window["mmsi"].nunique())
            mean_dwell = _safe_num(window["dwell_hours"].mean())
            anomaly_count = int(window["anomaly_flag"].sum())

            top_origins = (
                window["destination_arrival_norm"].fillna("UNKNOWN").astype(str).value_counts().head(2).index.tolist()
            )
            origin_text = ", ".join(top_origins) if top_origins else "UNKNOWN"

            text = (
                f"Between {start_str} and {end_str}, port {port_key} observed {arrivals} port-call events involving "
                f"{unique_vessels} vessels. Mean dwell time was {mean_dwell:.2f} hours. "
                f"Anomalies flagged: {anomaly_count}. Most common origin hints: {origin_text}."
            )

            metadata = {
                "strategy": "C",
                "chunk_type": "hybrid_window",
                "event_type": "window",
                "source": "PRJ896",
                "port": str(port_key),
                "locode": str(window["locode_norm"].dropna().astype(str).head(1).iloc[0] if not window["locode_norm"].dropna().empty else ""),
                "date": date_str,
                "vessel_type": "mixed",
                "anomaly": anomaly_count > 0,
                "anomaly_kind": "dwell_outlier" if anomaly_count > 0 else "none",
                "summary": True,
                "window_size": int(window_size),
                "event_id": f"{port_key}_{start_idx}_{date_str}",
            }
            chunk_id = f"C_WIN_{port_key}_{start_idx}_{date_str}"
            chunks.append({"id": chunk_id, "text": text, "metadata": _sanitize_metadata(metadata)})

    return chunks


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def build_chunks(
    processed_dir: Path,
    out_dir: Path,
    strategy: str = "all",
    hybrid_window_size: int = 5,
) -> Dict[str, Any]:
    port_calls = _read_parquet(processed_dir / "port_calls_clean.parquet")
    ais = _read_parquet(processed_dir / "ais_clean.parquet")
    port_day = _read_parquet(processed_dir / "port_day_metrics.parquet")
    ais_daily = _read_parquet(processed_dir / "ais_destination_daily.parquet")

    if port_calls.empty and ais.empty:
        raise RuntimeError("No cleaned datasets found. Run src.thesis.data_pipeline first.")

    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_norm = strategy.lower()
    stats: Dict[str, Any] = {}

    if strategy_norm in {"a", "all"}:
        a_chunks = _strategy_a_chunks(port_calls=port_calls, ais=ais)
        count = _write_jsonl(out_dir / "strategy_a_event_chunks.jsonl", a_chunks)
        stats["A"] = {"chunks": count, "path": str(out_dir / "strategy_a_event_chunks.jsonl")}

    if strategy_norm in {"b", "all"}:
        b_chunks = _strategy_b_chunks(port_day=port_day, ais_daily=ais_daily)
        count = _write_jsonl(out_dir / "strategy_b_port_day_chunks.jsonl", b_chunks)
        stats["B"] = {"chunks": count, "path": str(out_dir / "strategy_b_port_day_chunks.jsonl")}

    if strategy_norm in {"c", "all"}:
        c_chunks = _strategy_c_chunks(port_calls=port_calls, window_size=hybrid_window_size)
        count = _write_jsonl(out_dir / "strategy_c_hybrid_chunks.jsonl", c_chunks)
        stats["C"] = {"chunks": count, "path": str(out_dir / "strategy_c_hybrid_chunks.jsonl")}

    with (out_dir / "chunk_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build chunking strategies A/B/C for thesis")
    parser.add_argument("--processed_dir", default="data/thesis_processed")
    parser.add_argument("--out_dir", default="data/thesis_chunks")
    parser.add_argument("--strategy", default="all", choices=["a", "b", "c", "all"])
    parser.add_argument("--hybrid_window_size", type=int, default=5)
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    stats = build_chunks(
        processed_dir=Path(args.processed_dir),
        out_dir=Path(args.out_dir),
        strategy=args.strategy,
        hybrid_window_size=args.hybrid_window_size,
    )
    print("Chunking completed")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
