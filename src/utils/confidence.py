from __future__ import annotations

import re
from typing import List, Optional, Set, Union

from src.carbon.query import (
    CARBON_STATE_COMPUTED,
    CARBON_STATE_COMPUTED_ZERO,
    CARBON_STATE_FORECAST_ONLY,
    CARBON_STATE_NOT_COMPUTABLE,
    CARBON_STATE_RETRIEVAL_ONLY,
    CARBON_STATE_UNSUPPORTED,
    CarbonResult,
)
from src.forecast.forecast import ForecastResult
from src.kpi.query import AnalyticsResult


_INNOCUOUS_ANALYTICS_CAVEATS = (
    "combined total is the sum of per-port arrival counts in the same date window.",
    "first-arrival uses the earliest available arrival timestamp in filtered port-call dwell rows.",
    "last-arrival uses the latest available arrival timestamp in filtered port-call dwell rows.",
    "first-departure uses the earliest available departure timestamp in filtered port-call dwell rows.",
    "duration reflects arrival-to-departure time from port-call records.",
    "arrivals are filtered to `port_call` rows only.",
    "arrivals are filtered to `ais_destination_proxy` rows only.",
    "arrivals are computed from structured port-call rows for the matched scope.",
)


def _parse_rows_used(notes: List[str]) -> Optional[int]:
    for note in notes:
        match = re.search(r"Rows used:\s*([0-9,]+)", note, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                return None
    return None


def _parse_sources(notes: List[str]) -> Set[str]:
    for note in notes:
        if note.startswith("Data sources used:"):
            values = note.split(":", 1)[1]
            return {part.strip().lower() for part in values.split(",") if part.strip()}
    return set()


def _analytics_confidence_label(value: AnalyticsResult) -> str:
    if value.status != "ok":
        return "low (insufficient matched data/evidence)"

    for note in value.caveats:
        if note.lower().startswith("confidence:"):
            return note.split(":", 1)[1].strip()

    notes = list(value.coverage_notes or [])
    caveats = list(value.caveats or [])
    rows_used = _parse_rows_used(notes)
    sources = _parse_sources(notes)
    note_text = " ".join(notes).lower()
    caveat_text = " ".join(caveats).lower()

    has_scope_correction = any(note.startswith("Resolved scope correction:") for note in notes)
    has_proxy_source = "ais_destination_proxy" in sources
    has_mixed_sources = "port_call" in sources and has_proxy_source
    has_sparse_or_limited = any(token in caveat_text for token in ("limited", "sparse"))
    has_heuristic_logic = any(
        token in caveat_text
        for token in (
            "heuristic",
            "not proof",
            "jump rule",
            "proxy",
        )
    )
    has_reconstructed_logic = any(
        token in caveat_text
        for token in (
            "reconstructed voyage",
            "route-duration summary",
            "route-first uses earliest departure_time",
        )
    )
    significant_caveats = [
        caveat
        for caveat in caveats
        if caveat.lower() not in _INNOCUOUS_ANALYTICS_CAVEATS
    ]

    if rows_used is not None and rows_used <= 1 and (has_scope_correction or has_proxy_source):
        return "low (narrow deterministic match with scope fallback or proxy-only support)"

    if has_scope_correction:
        return "medium (deterministic result, but scope required fallback port resolution)"

    if has_mixed_sources:
        return "medium (deterministic result across mixed port-call and AIS proxy sources)"

    if has_proxy_source:
        return "medium (deterministic result from AIS-derived proxy data)"

    if has_sparse_or_limited:
        return "medium (deterministic result with limited underlying coverage)"

    if has_heuristic_logic:
        return "medium (deterministic heuristic over matched rows, not direct ground truth)"

    if has_reconstructed_logic:
        return "medium (deterministic result over reconstructed voyage episodes)"

    if significant_caveats:
        return "medium (deterministic result with explicit caveats in the matched scope)"

    if "no rows available" in note_text:
        return "low (insufficient matched data/evidence)"

    if rows_used is not None and rows_used >= 1:
        return "high (direct deterministic computation over matched records)"

    if value.table is not None and not value.table.empty:
        return "high (direct deterministic computation over structured result rows)"

    if "port_call" in sources:
        return "high (direct deterministic computation from port-call records)"

    return "medium (deterministic aggregation over filtered rows)"


def extract_confidence_label(
    value: Union[AnalyticsResult, ForecastResult, CarbonResult],
) -> str:
    if isinstance(value, CarbonResult):
        if value.result_state in {CARBON_STATE_NOT_COMPUTABLE, CARBON_STATE_UNSUPPORTED}:
            return "low / unavailable (deterministic carbon computation unavailable for this scope)"
        if value.result_state == CARBON_STATE_RETRIEVAL_ONLY:
            return "retrieval-only (supporting traffic evidence found, not numeric carbon source-of-truth)"
        if value.result_state == CARBON_STATE_FORECAST_ONLY:
            return "unavailable (carbon forecast requested but deterministic carbon forecast is not configured)"
        if value.result_state in {CARBON_STATE_COMPUTED, CARBON_STATE_COMPUTED_ZERO}:
            return f"{value.confidence_label} ({value.confidence_reason})"
        return "medium (carbon result returned without an explicit confidence state)"

    if isinstance(value, ForecastResult):
        if value.status != "ok":
            return "low (insufficient matched data/evidence)"
        for note in value.caveats:
            if note.lower().startswith("confidence:"):
                return note.split(":", 1)[1].strip()
        return "medium (forecast based on available historical patterns)"

    return _analytics_confidence_label(value)
