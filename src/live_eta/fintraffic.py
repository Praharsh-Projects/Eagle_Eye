"""Fintraffic Portnet + AIS adapter for live announced vessel ETAs.

This module intentionally does not predict an ETA.  Portnet is the authority
for the official scheduled ETA.  AIS supplies a vessel-reported ETA, and the
only derived delay metric is their signed announced variance:

    AIS vessel-reported ETA - Portnet official scheduled ETA

Positive minutes therefore mean the vessel announcement is later.  Every
timestamp exposed by this adapter is normalized to UTC and every query is
bounded to the next fourteen days.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

import httpx
import pandas as pd


FINTRAFFIC_BASE_URL = "https://meri.digitraffic.fi"
FINTRAFFIC_PORT_CALLS_PATH = "/api/port-call/v1/port-calls"
FINTRAFFIC_AIS_VESSEL_PATH = "/api/ais/v1/vessels/{mmsi}"
FINTRAFFIC_AIS_LOCATIONS_PATH = "/api/ais/v1/locations"
FINTRAFFIC_PORT_CALLS_URL = FINTRAFFIC_BASE_URL + FINTRAFFIC_PORT_CALLS_PATH
FINTRAFFIC_AIS_VESSELS_URL = FINTRAFFIC_BASE_URL + "/api/ais/v1/vessels"
FINTRAFFIC_AIS_LOCATIONS_URL = FINTRAFFIC_BASE_URL + FINTRAFFIC_AIS_LOCATIONS_PATH

MAX_QUERY_HORIZON_DAYS = 14
PORTNET_MAX_AGE = timedelta(minutes=5)
AIS_METADATA_MAX_AGE = timedelta(minutes=10)
AIS_LOCATION_MAX_AGE = timedelta(minutes=10)
AIS_COLLECTION_MAX_AGE = timedelta(minutes=5)
MAX_ANNOUNCED_VARIANCE = timedelta(hours=48)
FUTURE_CLOCK_TOLERANCE = timedelta(minutes=10)

LIVE_ETA_OPERATIONS = {
    "live_port_arrivals",
    "vessel_eta",
    "vessel_delay",
    "eta_comparison",
}

_FINNISH_PORT_ALIASES: Dict[str, str] = {
    "hamina": "FIKTK",
    "haminakotka": "FIKTK",
    "helsinki": "FIHEL",
    "hanko": "FIHKO",
    "kemi": "FIKEM",
    "kokkola": "FIKOK",
    "kotka": "FIKTK",
    "naantali": "FINLI",
    "oulu": "FIOUL",
    "pori": "FIPOR",
    "rauma": "FIRAU",
    "turku": "FITKU",
    "vaasa": "FIVAA",
}

# Fintraffic's open Portnet schedules cover Finland.  Its AIS receiver also
# observes vessels whose self-reported destinations are elsewhere around the
# Baltic.  These aliases are deliberately a curated port vocabulary rather
# than a claim that Fintraffic supplies official schedules for those ports.
BALTIC_PORT_ALIASES: Dict[str, str] = {
    **_FINNISH_PORT_ALIASES,
    # Sweden (kept first and broad because Eagle Eye is Sweden-oriented).
    "stockholm": "SESTO",
    "nynashamn": "SENYN",
    "visby": "SEVBY",
    "gavle": "SEGVX",
    "karlshamn": "SEKAN",
    "karlskrona": "SEKAA",
    "kalmar": "SEKLR",
    "oskarshamn": "SEOSK",
    "norrkoping": "SENRK",
    "lulea": "SELLA",
    "malmo": "SEMMA",
    "trelleborg": "SETRG",
    "ystad": "SEYST",
    "helsingborg": "SEHEL",
    "sodertalje": "SESTQ",
    "sundsvall": "SESDL",
    "umea": "SEUME",
    # Estonia, Latvia and Lithuania.
    "tallinn": "EETLL",
    "parnu": "EEPRN",
    "riga": "LVRIX",
    "ventspils": "LVVNT",
    "liepaja": "LVLPX",
    "klaipeda": "LTKLJ",
    # Poland, Germany and Denmark.
    "gdansk": "PLGDN",
    "gdynia": "PLGDY",
    "szczecin": "PLSZZ",
    "swinoujscie": "PLSWI",
    "rostock": "DERSK",
    "lubeck": "DELBC",
    "kiel": "DEKEL",
    "copenhagen": "DKCPH",
    # Eastern Baltic destinations can occur in AIS broadcasts even though no
    # official schedule authority is integrated for them.
    "kaliningrad": "RUKGD",
    "stpetersburg": "RULED",
    "saintpetersburg": "RULED",
    "ustluga": "RUULU",
}

BALTIC_COUNTRY_PREFIXES = frozenset({"SE", "FI", "EE", "LV", "LT", "PL", "DE", "DK", "RU"})
AIS_DESTINATION_GROUPS = frozenset(
    {"swedish_destination_signals", "baltic_destination_signals"}
)

# Exact curated destination tokens accepted in addition to a reported
# UN/LOCODE. AIS destination text is free-form, so the list stays explicit and
# auditable; unknown names are never guessed.
_AIS_DESTINATION_TOKENS: Dict[str, Tuple[str, ...]] = {
    locode: tuple(
        sorted(
            {
                name.upper()
                for name, candidate in BALTIC_PORT_ALIASES.items()
                if candidate == locode
            }
        )
    )
    for locode in set(BALTIC_PORT_ALIASES.values())
}

class HTTPResponse(Protocol):
    status_code: int
    headers: Any

    def json(self) -> Any:
        ...


class HTTPClient(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> HTTPResponse:
        ...


class FintrafficUnavailable(RuntimeError):
    """Raised when no current validated provider response can be obtained."""


@dataclass
class _CacheEntry:
    stored_at: float
    value: Any
    etag: Optional[str] = None


@dataclass
class LiveETAResult:
    """Executor-compatible deterministic live ETA result."""

    status: str
    answer: str
    table: Optional[pd.DataFrame]
    coverage_notes: List[str]
    caveats: List[str]
    snapshot_at: datetime
    horizon_end: Optional[datetime] = None
    data_updated_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    source_kind: str = "portnet_with_ais"

    @property
    def chart(self) -> Optional[pd.DataFrame]:
        return self.table


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _iso_utc(value: datetime) -> str:
    current = value.astimezone(timezone.utc)
    return current.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ais_window_milliseconds(snapshot: datetime) -> Tuple[int, int]:
    """Return a minute-stable ten-minute window so the 60s cache is effective."""

    to_ms = int(snapshot.timestamp() // 60 * 60 * 1000)
    from_ms = to_ms - int(AIS_METADATA_MAX_AGE.total_seconds() * 1000)
    return from_ms, to_ms


def _digits(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = re.sub(r"\D", "", str(value))
    return token or None


def _normalized_text(value: Any) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value or "")).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return re.sub(r"[^A-Z0-9]+", "", ascii_value.upper())


def normalize_finnish_port(value: Optional[str]) -> Optional[str]:
    """Resolve a conservative Finnish name/UNLOCODE scope for Fintraffic."""

    if not value:
        return None
    compact = _normalized_text(value)
    if re.fullmatch(r"FI[A-Z]{3}", compact):
        return compact
    return _FINNISH_PORT_ALIASES.get(compact.lower())


def normalize_baltic_port(value: Optional[str]) -> Optional[str]:
    """Resolve a curated Baltic port name or an explicit Baltic UN/LOCODE."""

    if not value:
        return None
    compact = _normalized_text(value)
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{3}", compact):
        return compact if compact[:2] in BALTIC_COUNTRY_PREFIXES else None
    return BALTIC_PORT_ALIASES.get(compact.lower())


def valid_mmsi(value: Optional[str]) -> bool:
    return bool(value and re.fullmatch(r"\d{9}", value))


def valid_imo(value: Optional[str]) -> bool:
    if not value or not re.fullmatch(r"\d{7}", value):
        return False
    checksum = sum(int(value[index]) * (7 - index) for index in range(6)) % 10
    return checksum == int(value[-1])


def _decode_ais_eta(encoded: Any, reference: datetime) -> Optional[datetime]:
    """Decode AIS Message 5 ETA bits and attach the nearest plausible UTC year."""

    try:
        value = int(encoded)
    except (TypeError, ValueError):
        return None
    month = (value >> 16) & 0x0F
    day = (value >> 11) & 0x1F
    hour = (value >> 6) & 0x1F
    minute = value & 0x3F
    if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    candidates: List[datetime] = []
    for year in (reference.year - 1, reference.year, reference.year + 1):
        try:
            candidates.append(datetime(year, month, day, hour, minute, tzinfo=timezone.utc))
        except ValueError:
            continue
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs((item - reference).total_seconds()))


def _destination_matches(destination: Any, locode: str) -> bool:
    destination_token = _normalized_text(destination)
    if not destination_token:
        return False
    normalized_locode = _normalized_text(locode)
    if normalized_locode == destination_token:
        return True
    return any(
        _normalized_text(token) == destination_token
        for token in _AIS_DESTINATION_TOKENS.get(normalized_locode, ())
        if len(_normalized_text(token)) >= 4
    )


def _destination_group_locode(
    destination: Any,
    aggregation: Optional[str],
) -> Optional[str]:
    """Return a group-matched destination code without interpreting routes.

    Group queries intentionally accept only a complete five-character
    country-prefixed token after whitespace/punctuation normalization. This is
    a vessel-reported code-shape check, not an assertion that the value was
    verified against the official UN/LOCODE registry. Curated free-form port
    names remain available only for an explicit port scope.
    """

    if aggregation not in AIS_DESTINATION_GROUPS:
        return None
    destination_token = _normalized_text(destination)
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{3}", destination_token):
        return None
    if aggregation == "swedish_destination_signals":
        return destination_token if destination_token.startswith("SE") else None
    return (
        destination_token
        if destination_token[:2] in BALTIC_COUNTRY_PREFIXES
        else None
    )


class FintrafficETAAdapter:
    """Read-only live ETA adapter with bounded retries and in-memory TTL cache."""

    provider = "fintraffic_digitraffic"

    def __init__(
        self,
        *,
        base_url: str = FINTRAFFIC_BASE_URL,
        user_agent: str = "EagleEye/2.0",
        timeout_seconds: float = 5.0,
        retries: int = 1,
        cache_ttl_seconds: float = 60.0,
        http_client: Optional[HTTPClient] = None,
        now_fn: Callable[[], datetime] = _utc_now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent.strip() or "EagleEye/2.0"
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.retries = max(0, min(int(retries), 4))
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.http_client = http_client or httpx.Client(follow_redirects=False)
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], _CacheEntry] = {}
        self._cache_lock = threading.Lock()

    def capabilities(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "available": True,
            "country_scope": sorted(BALTIC_COUNTRY_PREFIXES),
            "official_schedule_country_scope": ["FI"],
            "ais_destination_country_scope": sorted(BALTIC_COUNTRY_PREFIXES),
            "timezone": "UTC",
            "maximum_horizon_days": MAX_QUERY_HORIZON_DAYS,
            "operations": sorted(LIVE_ETA_OPERATIONS),
            "official_eta_authority": "Fintraffic Portnet",
            "regional_ais_scope": (
                "Fresh vessel-reported destinations, ETAs and positions observed by "
                "Fintraffic AIS; not a complete or official foreign-port arrival board."
            ),
            "announced_variance": (
                "AIS vessel-reported ETA minus Portnet official scheduled ETA; "
                "positive minutes mean later. Available for validated Finnish Portnet "
                "matches only."
            ),
            "prediction": False,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retries + 1,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }

    def query(
        self,
        *,
        operation: str,
        port: Optional[str] = None,
        ports: Optional[List[str]] = None,
        mmsi: Optional[str] = None,
        imo: Optional[str] = None,
        vessel_name: Optional[str] = None,
        target_date: Optional[str] = None,
        horizon_hours: Optional[int] = None,
        aggregation: Optional[str] = None,
        limit: int = 20,
    ) -> LiveETAResult:
        snapshot = self.now_fn().astimezone(timezone.utc)
        bounded_hours = max(
            1,
            min(
                int(horizon_hours or MAX_QUERY_HORIZON_DAYS * 24),
                MAX_QUERY_HORIZON_DAYS * 24,
            ),
        )
        horizon_end = snapshot + timedelta(hours=bounded_hours)
        if operation not in LIVE_ETA_OPERATIONS:
            return self._unavailable(
                snapshot,
                "Unsupported live ETA operation.",
                horizon_end=horizon_end,
            )

        raw_ports = list(ports or ([] if port is None else [port]))
        locodes: List[str] = []
        for raw_port in raw_ports:
            normalized_port = normalize_baltic_port(raw_port)
            if not normalized_port:
                return self._unavailable(
                    snapshot,
                    "The requested port is outside Eagle Eye's curated Baltic AIS destination coverage.",
                    horizon_end=horizon_end,
                    failure_code="coverage_unavailable",
                )
            if normalized_port not in locodes:
                locodes.append(normalized_port)
        normalized_mmsi = _digits(mmsi)
        normalized_imo = _digits(imo)
        normalized_name = re.sub(r"\s+", " ", str(vessel_name or "")).strip() or None
        if normalized_mmsi and not valid_mmsi(normalized_mmsi):
            return self._unavailable(
                snapshot,
                "MMSI must contain exactly nine digits.",
                horizon_end=horizon_end,
            )
        if normalized_imo and not valid_imo(normalized_imo):
            return self._unavailable(
                snapshot,
                "IMO must be a valid seven-digit IMO number.",
                horizon_end=horizon_end,
            )

        if target_date:
            target = _as_utc(f"{target_date}T00:00:00Z")
            if target is None:
                return self._unavailable(
                    snapshot,
                    "The requested ETA date is invalid.",
                    horizon_end=horizon_end,
                )
            if target.date() < snapshot.date() or target.date() > horizon_end.date():
                return self._unavailable(
                    snapshot,
                    f"Live ETA requests are limited to the next {MAX_QUERY_HORIZON_DAYS} days in UTC.",
                    horizon_end=horizon_end,
                    failure_code="coverage_unavailable",
                )

        non_finnish_locodes = [
            locode for locode in locodes if not locode.startswith("FI")
        ]
        destination_group = (
            aggregation
            if not locodes and aggregation in AIS_DESTINATION_GROUPS
            else None
        )
        if destination_group:
            if operation != "vessel_eta":
                return self._unavailable(
                    snapshot,
                    "Regional AIS destination groups provide vessel-reported ETA signals only; "
                    "they do not provide an official schedule or delay baseline.",
                    horizon_end=horizon_end,
                    failure_code="official_schedule_coverage_unavailable",
                    source_kind="ais_destination_only",
                )
            return self._query_regional_ais(
                locodes=[],
                destination_group=destination_group,
                snapshot=snapshot,
                horizon_end=horizon_end,
                bounded_hours=bounded_hours,
                mmsi=normalized_mmsi,
                imo=normalized_imo,
                vessel_name=normalized_name,
                target_date=target_date,
                limit=limit,
            )
        if non_finnish_locodes:
            if len(non_finnish_locodes) != len(locodes):
                return self._unavailable(
                    snapshot,
                    "Finnish official schedules and regional AIS-only destinations cannot be combined in one result.",
                    horizon_end=horizon_end,
                    failure_code="mixed_authority_scope",
                )
            if operation != "vessel_eta":
                unavailable_subject = (
                    "An official schedule baseline is not integrated for this non-Finnish Baltic port, "
                    "so announced delay or ETA variance is unavailable."
                    if operation in {"vessel_delay", "eta_comparison"}
                    else
                    "Official scheduled arrivals are not integrated for this non-Finnish Baltic port."
                )
                return self._unavailable(
                    snapshot,
                    unavailable_subject
                    + " Ask for fresh AIS-visible vessels reporting the port as their destination instead.",
                    horizon_end=horizon_end,
                    failure_code="official_schedule_coverage_unavailable",
                    source_kind="ais_destination_only",
                )
            return self._query_regional_ais(
                locodes=locodes,
                destination_group=None,
                snapshot=snapshot,
                horizon_end=horizon_end,
                bounded_hours=bounded_hours,
                mmsi=normalized_mmsi,
                imo=normalized_imo,
                vessel_name=normalized_name,
                target_date=target_date,
                limit=limit,
            )

        base_params: Dict[str, Any] = {
            "etaFrom": _iso_utc(snapshot),
            "etaTo": _iso_utc(horizon_end),
        }
        if normalized_mmsi:
            base_params["mmsi"] = int(normalized_mmsi)
        if normalized_imo:
            base_params["imo"] = int(normalized_imo)
        if normalized_name:
            base_params["vesselName"] = normalized_name

        try:
            rows: List[Dict[str, Any]] = []
            update_times: List[datetime] = []
            for query_locode in (locodes or [None]):
                params = dict(base_params)
                if query_locode:
                    params["locode"] = query_locode
                payload = self._get_json(FINTRAFFIC_PORT_CALLS_PATH, params)
                current_rows, current_updated = self._portnet_rows(
                    payload,
                    snapshot=snapshot,
                    horizon_end=horizon_end,
                    locode=query_locode,
                    mmsi=normalized_mmsi,
                    imo=normalized_imo,
                    vessel_name=normalized_name,
                )
                rows.extend(current_rows)
                update_times.append(current_updated)
            if len(set(update_times)) != 1:
                raise FintrafficUnavailable(
                    "Fintraffic returned different Portnet update snapshots for the compared ports."
                )
            data_updated = update_times[0]
        except FintrafficUnavailable as exc:
            return self._unavailable(
                snapshot,
                str(exc),
                horizon_end=horizon_end,
            )

        if target_date:
            rows = [
                row
                for row in rows
                if row["official_eta_utc"].astimezone(timezone.utc).date().isoformat() == target_date
            ]
        rows = rows[: max(1, min(int(limit), 100))]
        if not rows:
            return LiveETAResult(
                status="no_data",
                answer=(
                    "Fintraffic Portnet returned no matching upcoming official scheduled arrivals "
                    f"within the requested {bounded_hours}-hour UTC horizon."
                ),
                table=None,
                coverage_notes=[
                    f"Retrieval snapshot: {_iso_utc(snapshot)}",
                    f"Query horizon: {_iso_utc(snapshot)} to {_iso_utc(horizon_end)}",
                ],
                caveats=["No matching current-source rows were available."],
                snapshot_at=snapshot,
                horizon_end=horizon_end,
                data_updated_at=data_updated,
            )

        enriched = self._enrich_with_ais(rows, snapshot=snapshot, horizon_end=horizon_end)
        if aggregation == "positive_announced_variance":
            enriched = [
                row
                for row in enriched
                if row.get("announced_delay_minutes") is not None
                and int(row["announced_delay_minutes"]) > 0
            ]
        elif aggregation == "negative_announced_variance":
            enriched = [
                row
                for row in enriched
                if row.get("announced_delay_minutes") is not None
                and int(row["announced_delay_minutes"]) < 0
            ]
        elif aggregation == "missing_fresh_ais":
            enriched = [
                row
                for row in enriched
                if row.get("announced_delay_minutes") is None
            ]
        if not enriched:
            filter_label = {
                "positive_announced_variance": "a positive announced ETA variance",
                "negative_announced_variance": "a negative announced ETA variance",
                "missing_fresh_ais": "a missing fresh matching AIS ETA",
            }.get(aggregation, "the requested live-source condition")
            return LiveETAResult(
                status="no_data",
                answer=(
                    f"None of the {len(rows)} matching upcoming official Portnet arrival(s) "
                    f"satisfied {filter_label}. No empty selection was returned as a live result."
                ),
                table=None,
                coverage_notes=self._coverage(
                    snapshot,
                    horizon_end,
                    data_updated,
                    rows,
                ),
                caveats=[
                    "Missing announced variance remains unavailable rather than zero.",
                    "No matching current-source rows were available.",
                ],
                snapshot_at=snapshot,
                horizon_end=horizon_end,
                data_updated_at=data_updated,
                failure_reason="no_matching_live_rows",
            )
        table = self._frame(enriched)
        validated_rows = [
            row for row in enriched if row.get("announced_delay_minutes") is not None
        ]

        if operation == "vessel_delay":
            if not validated_rows:
                official = enriched[0] if enriched else rows[0]
                return LiveETAResult(
                    status="no_current_data",
                    answer=(
                        "A current announced delay cannot be calculated because no validated AIS ETA "
                        f"matches the Portnet official schedule at {_iso_utc(official['official_eta_utc'])}. "
                        "No missing variance was treated as zero."
                    ),
                    table=table,
                    coverage_notes=self._coverage(snapshot, horizon_end, data_updated, enriched),
                    caveats=[
                        "Announced delay requires an exact vessel identity, a fresh AIS position and metadata record, "
                        "a matching AIS destination, and a valid AIS ETA.",
                        "This is an announced variance, not a predicted delay.",
                    ],
                    snapshot_at=snapshot,
                    horizon_end=horizon_end,
                    data_updated_at=data_updated,
                    failure_reason="validated_ais_match_unavailable",
                )
            if not (normalized_mmsi or normalized_imo or normalized_name):
                return LiveETAResult(
                    status="ok",
                    answer=(
                        f"Found {len(validated_rows)} upcoming arrival(s) with validated announced ETA "
                        f"variance across {', '.join(locodes)} within {bounded_hours} hours UTC. "
                        "This is not a predicted delay."
                    ),
                    table=table,
                    coverage_notes=self._coverage(snapshot, horizon_end, data_updated, enriched),
                    caveats=[
                        "Announced delay is AIS vessel-reported ETA minus Portnet official scheduled ETA.",
                        "Positive minutes mean later; negative minutes mean earlier.",
                    ],
                    snapshot_at=snapshot,
                    horizon_end=horizon_end,
                    data_updated_at=data_updated,
                )
            selected = validated_rows[0]
            minutes = int(selected["announced_delay_minutes"])
            if minutes > 0:
                direction = f"{minutes} minutes later"
            elif minutes < 0:
                direction = f"{abs(minutes)} minutes earlier"
            else:
                direction = "at the same announced time"
            return LiveETAResult(
                status="ok",
                answer=(
                    f"For {selected['vessel_label']}, Portnet's official scheduled ETA at "
                    f"{selected['port_locode']} is {_iso_utc(selected['official_eta_utc'])}; "
                    f"the vessel-reported AIS ETA is {_iso_utc(selected['ais_eta_utc'])}. "
                    f"The announced variance is {direction} ({minutes:+d} minutes). "
                    "This is not a predicted delay."
                ),
                table=table,
                coverage_notes=self._coverage(snapshot, horizon_end, data_updated, enriched),
                caveats=[
                    "Announced delay is AIS vessel-reported ETA minus Portnet official scheduled ETA.",
                    "Positive minutes mean later; negative minutes mean earlier.",
                ],
                snapshot_at=snapshot,
                horizon_end=horizon_end,
                data_updated_at=data_updated,
            )

        if operation == "vessel_eta":
            selected = enriched[0]
            ais_eta = selected.get("ais_eta_utc")
            if ais_eta is None:
                answer = (
                    f"Portnet's official scheduled ETA for {selected['vessel_label']} at "
                    f"{selected['port_locode']} is {_iso_utc(selected['official_eta_utc'])}. "
                    "A validated vessel-reported AIS ETA is currently unavailable, so no announced "
                    "variance or predicted ETA is shown."
                )
                status = "ok"
            else:
                answer = (
                    f"Portnet's official scheduled ETA for {selected['vessel_label']} at "
                    f"{selected['port_locode']} is {_iso_utc(selected['official_eta_utc'])}. "
                    f"The validated vessel-reported AIS ETA is {_iso_utc(ais_eta)} UTC. "
                    "Both are announced times, not a model prediction."
                )
                status = "ok"
            return LiveETAResult(
                status=status,
                answer=answer,
                table=table,
                coverage_notes=self._coverage(snapshot, horizon_end, data_updated, enriched),
                caveats=[
                    "Portnet is the official schedule authority; AIS ETA is self-reported by the vessel.",
                    "All timestamps are UTC and the live horizon is limited to fourteen days.",
                ],
                snapshot_at=snapshot,
                horizon_end=horizon_end,
                data_updated_at=data_updated,
            )

        count = len(enriched)
        validated_count = len(validated_rows)
        locode_label = ", ".join(locodes) or "the selected Finnish vessel scope"
        if aggregation == "missing_fresh_ais":
            answer = (
                f"Of the {len(rows)} upcoming official Portnet arrival(s) examined at "
                f"{locode_label}, {count} do not have a fresh matching AIS ETA. "
                "Their AIS ETA and announced variance remain unavailable rather than zero."
            )
        elif operation == "eta_comparison":
            answer = (
                f"Compared {count} upcoming Portnet scheduled arrival(s) for {locode_label} from one "
                f"retrieval snapshot; {validated_count} have a validated AIS announced ETA and "
                f"{count - validated_count} retain schedule-only status. Missing announced variance "
                "remains unavailable rather than zero."
            )
        else:
            answer = (
                f"Fintraffic Portnet lists {count} upcoming official scheduled arrival(s) at "
                f"{locode_label} within the requested {bounded_hours}-hour UTC horizon. "
                f"{validated_count} have a validated AIS announced ETA."
            )
        return LiveETAResult(
            status="ok",
            answer=answer,
            table=table,
            coverage_notes=self._coverage(snapshot, horizon_end, data_updated, enriched),
            caveats=[
                "Rows without a validated AIS match have no numeric announced variance.",
                "No row is a model-predicted ETA or operational delay forecast.",
            ],
            snapshot_at=snapshot,
            horizon_end=horizon_end,
            data_updated_at=data_updated,
        )

    def _query_regional_ais(
        self,
        *,
        locodes: List[str],
        destination_group: Optional[str],
        snapshot: datetime,
        horizon_end: datetime,
        bounded_hours: int,
        mmsi: Optional[str],
        imo: Optional[str],
        vessel_name: Optional[str],
        target_date: Optional[str],
        limit: int,
    ) -> LiveETAResult:
        """Return a bounded AIS destination observation outside Finnish Portnet.

        This is intentionally not an arrival-board implementation.  It reports
        only fresh vessel broadcasts observed by Fintraffic whose destination
        field exactly equals a requested UN/LOCODE or an audited port alias.
        Group scopes accept only complete UN/LOCODE tokens.
        """

        from_ms, to_ms = _ais_window_milliseconds(snapshot)
        try:
            metadata_payload = self._get_json(
                "/api/ais/v1/vessels",
                {"from": from_ms, "to": to_ms},
            )
            location_payload = self._get_json(
                FINTRAFFIC_AIS_LOCATIONS_PATH,
            )
        except FintrafficUnavailable as exc:
            return self._unavailable(
                snapshot,
                str(exc),
                horizon_end=horizon_end,
                failure_code="ais_source_unavailable",
                source_kind="ais_destination_only",
            )

        metadata_rows: Any
        if isinstance(metadata_payload, list):
            metadata_rows = metadata_payload
        elif isinstance(metadata_payload, dict) and isinstance(
            metadata_payload.get("vessels"), list
        ):
            metadata_rows = metadata_payload["vessels"]
        else:
            return self._unavailable(
                snapshot,
                "Fintraffic AIS vessel metadata returned an invalid response shape.",
                horizon_end=horizon_end,
                failure_code="ais_source_unavailable",
                source_kind="ais_destination_only",
            )

        if not isinstance(location_payload, dict) or not isinstance(
            location_payload.get("features"), list
        ):
            return self._unavailable(
                snapshot,
                "Fintraffic AIS locations returned an invalid response shape.",
                horizon_end=horizon_end,
                failure_code="ais_source_unavailable",
                source_kind="ais_destination_only",
            )
        locations_updated = _as_utc(location_payload.get("dataUpdatedTime"))
        if locations_updated is None:
            return self._unavailable(
                snapshot,
                "Fintraffic AIS locations omitted the collection timestamp.",
                horizon_end=horizon_end,
                failure_code="ais_source_stale",
                source_kind="ais_destination_only",
            )
        if (
            snapshot - locations_updated > AIS_COLLECTION_MAX_AGE
            or locations_updated - snapshot > FUTURE_CLOCK_TOLERANCE
        ):
            return self._unavailable(
                snapshot,
                "Fintraffic AIS location collection is stale or has an invalid future timestamp.",
                horizon_end=horizon_end,
                failure_code="ais_source_stale",
                source_kind="ais_destination_only",
            )

        locations_by_mmsi: Dict[str, Dict[str, Any]] = {}
        for feature in location_payload["features"]:
            if not isinstance(feature, dict):
                continue
            properties = (
                feature.get("properties")
                if isinstance(feature.get("properties"), dict)
                else {}
            )
            feature_mmsi = _digits(feature.get("mmsi") or properties.get("mmsi"))
            geometry = (
                feature.get("geometry")
                if isinstance(feature.get("geometry"), dict)
                else {}
            )
            coordinates = geometry.get("coordinates")
            if not feature_mmsi or not isinstance(coordinates, list) or len(coordinates) < 2:
                continue
            try:
                longitude = float(coordinates[0])
                latitude = float(coordinates[1])
                location_time = datetime.fromtimestamp(
                    int(properties.get("timestampExternal")) / 1000.0,
                    tz=timezone.utc,
                )
            except (TypeError, ValueError, OSError):
                continue
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                continue
            if (
                snapshot - location_time > AIS_LOCATION_MAX_AGE
                or location_time - snapshot > FUTURE_CLOCK_TOLERANCE
            ):
                continue
            try:
                sog_kn = float(properties.get("sog"))
            except (TypeError, ValueError):
                sog_kn = None
            if sog_kn is not None and (sog_kn < 0 or sog_kn >= 102.3):
                sog_kn = None
            candidate = {
                "ais_location_time_utc": location_time,
                "latitude": latitude,
                "longitude": longitude,
                "sog_kn": sog_kn,
            }
            previous = locations_by_mmsi.get(feature_mmsi)
            if (
                previous is None
                or candidate["ais_location_time_utc"]
                > previous["ais_location_time_utc"]
            ):
                locations_by_mmsi[feature_mmsi] = candidate

        rows: List[Dict[str, Any]] = []
        identities_seen: Dict[str, Tuple[Any, ...]] = {}
        ambiguous_identity = False
        normalized_requested_name = _normalized_text(vessel_name)
        for metadata in metadata_rows:
            if not isinstance(metadata, dict):
                continue
            row_mmsi = _digits(metadata.get("mmsi"))
            row_imo = _digits(metadata.get("imo"))
            row_name = re.sub(
                r"\s+", " ", str(metadata.get("name") or "")
            ).strip() or None
            if not row_mmsi or not valid_mmsi(row_mmsi):
                continue
            if mmsi and row_mmsi != mmsi:
                continue
            if imo and row_imo != imo:
                continue
            if vessel_name and _normalized_text(row_name) != normalized_requested_name:
                continue

            if destination_group:
                group_locode = _destination_group_locode(
                    metadata.get("destination"),
                    destination_group,
                )
                matching_locodes = [group_locode] if group_locode else []
            else:
                matching_locodes = [
                    locode
                    for locode in locodes
                    if _destination_matches(metadata.get("destination"), locode)
                ]
            if len(matching_locodes) != 1:
                continue
            row_locode = matching_locodes[0]
            timestamp_raw = metadata.get("timestamp")
            try:
                metadata_time = datetime.fromtimestamp(
                    int(timestamp_raw) / 1000.0,
                    tz=timezone.utc,
                )
            except (TypeError, ValueError, OSError):
                continue
            if (
                snapshot - metadata_time > AIS_METADATA_MAX_AGE
                or metadata_time - snapshot > FUTURE_CLOCK_TOLERANCE
            ):
                continue
            ais_eta = _decode_ais_eta(metadata.get("eta"), snapshot)
            if (
                ais_eta is None
                or ais_eta < snapshot - FUTURE_CLOCK_TOLERANCE
                or ais_eta > horizon_end
            ):
                continue
            if target_date and ais_eta.date().isoformat() != target_date:
                continue
            location = locations_by_mmsi.get(row_mmsi)
            if location is None:
                continue

            identity_signature = (
                row_imo,
                _normalized_text(row_name),
                row_locode,
                int(metadata.get("eta")),
            )
            previous_signature = identities_seen.get(row_mmsi)
            if previous_signature is not None:
                if previous_signature != identity_signature:
                    ambiguous_identity = True
                continue
            identities_seen[row_mmsi] = identity_signature
            rows.append(
                {
                    "snapshot_time_utc": snapshot,
                    "portnet_data_updated_utc": None,
                    "port_call_id": f"ais:{row_mmsi}:{row_locode}",
                    "port_call_updated_utc": None,
                    "port_locode": row_locode,
                    "vessel_name": row_name,
                    "vessel_label": row_name or row_mmsi,
                    "mmsi": row_mmsi,
                    "imo": row_imo,
                    "official_eta_utc": None,
                    "official_eta_updated_utc": None,
                    "official_eta_source": None,
                    "port_area_code": None,
                    "port_area_name": None,
                    "berth_code": None,
                    "berth_name": None,
                    "ais_destination": str(metadata.get("destination") or "").strip(),
                    "ais_destination_match": (
                        "exact_country_prefixed_group_token"
                        if destination_group
                        else "exact_unlocode_token"
                        if _normalized_text(metadata.get("destination"))
                        == row_locode
                        else "exact_curated_port_name"
                    ),
                    "ais_eta_utc": ais_eta,
                    "ais_metadata_time_utc": metadata_time,
                    **location,
                    "announced_delay_minutes": None,
                    "variance_status": "ais_only_no_official_schedule",
                    "source_scope": "fintraffic_ais_observation",
                }
            )

        if ambiguous_identity:
            return self._unavailable(
                snapshot,
                "Fintraffic AIS returned conflicting metadata for a matching vessel identity.",
                horizon_end=horizon_end,
                failure_code="ambiguous_match",
                source_kind="ais_destination_only",
            )
        if vessel_name:
            matching_name_mmsis = {
                str(row["mmsi"])
                for row in rows
                if _normalized_text(row.get("vessel_name")) == normalized_requested_name
            }
            if len(matching_name_mmsis) > 1:
                return self._unavailable(
                    snapshot,
                    "The vessel name matches multiple fresh AIS identities; provide MMSI or IMO.",
                    horizon_end=horizon_end,
                    failure_code="ambiguous_match",
                    source_kind="ais_destination_only",
                )

        rows.sort(
            key=lambda row: (
                row["ais_eta_utc"],
                row["port_locode"],
                row["vessel_label"],
                row["mmsi"],
            )
        )
        total_count = len(rows)
        rows = rows[: max(1, min(int(limit), 100))]
        group_label = (
            "Swedish-coded destinations identified by exact five-character AIS broadcasts"
            if destination_group == "swedish_destination_signals"
            else "Baltic-coded destinations identified by exact five-character in-scope AIS broadcasts"
            if destination_group == "baltic_destination_signals"
            else ", ".join(locodes)
        )
        matching_rule = (
            "exact normalized five-character country-prefixed destination code"
            if destination_group
            else "exact UN/LOCODE or exact curated port token"
        )
        if not rows:
            return LiveETAResult(
                status="no_data",
                answer=(
                    "Fintraffic AIS currently has no fresh vessel broadcast with an exact destination "
                    f"match for {group_label} and a valid ETA inside the requested {bounded_hours}-hour UTC horizon."
                ),
                table=None,
                coverage_notes=[
                    f"Retrieval snapshot: {_iso_utc(snapshot)}",
                    f"AIS locations updated: {_iso_utc(locations_updated)}",
                    f"UTC horizon: {_iso_utc(snapshot)} to {_iso_utc(horizon_end)}",
                    f"Destination matching: {matching_rule}.",
                ],
                caveats=[
                    "Fintraffic AIS is an observed Finnish-receiver footprint, not a complete Baltic arrival board."
                ],
                snapshot_at=snapshot,
                horizon_end=horizon_end,
                data_updated_at=locations_updated,
                source_kind="ais_destination_only",
            )

        table = self._frame(rows)
        if len(rows) == 1:
            selected = rows[0]
            answer = (
                f"Fintraffic AIS shows {selected['vessel_label']} (MMSI {selected['mmsi']}) "
                f"reporting {selected['port_locode']} as its destination with a vessel-reported ETA "
                f"of {_iso_utc(selected['ais_eta_utc'])}. Its latest observed position was "
                f"{selected['latitude']:.5f}, {selected['longitude']:.5f} at "
                f"{_iso_utc(selected['ais_location_time_utc'])}. This is not an official port "
                "schedule or an Eagle Eye prediction."
            )
        else:
            displayed = len(rows)
            answer = (
                f"Fintraffic AIS contains {total_count} fresh vessel broadcast(s) with an exact "
                f"destination match for {group_label} and a valid ETA inside the requested "
                f"{bounded_hours}-hour UTC horizon"
                + (
                    f"; the first {displayed} by vessel-reported ETA are shown"
                    if displayed < total_count
                    else ""
                )
                + ". This is an observed, non-exhaustive AIS set—not an official port arrival board "
                "or a prediction."
            )
        return LiveETAResult(
            status="ok",
            answer=answer,
            table=table,
            coverage_notes=[
                f"Retrieval snapshot: {_iso_utc(snapshot)}",
                f"AIS locations updated: {_iso_utc(locations_updated)}",
                f"UTC horizon: {_iso_utc(snapshot)} to {_iso_utc(horizon_end)}",
                f"Fresh exact destination matches: {total_count}",
                f"Destination matching: {matching_rule}.",
            ],
            caveats=[
                "AIS ETA and destination are self-reported by each vessel.",
                "Fintraffic AIS is an observed Finnish-receiver footprint, not a complete Baltic arrival board.",
                "No official schedule, announced delay, route, or predicted ETA is inferred.",
            ],
            snapshot_at=snapshot,
            horizon_end=horizon_end,
            data_updated_at=locations_updated,
            source_kind="ais_destination_only",
        )

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        normalized_params = tuple(
            sorted((str(key), str(value)) for key, value in (params or {}).items())
        )
        cache_key = (path, normalized_params)
        monotonic_now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and monotonic_now - cached.stored_at <= self.cache_ttl_seconds:
                return cached.value

        last_error = "request failed"
        request_headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Digitraffic-User": self.user_agent,
        }
        if cached and cached.etag:
            request_headers["If-None-Match"] = cached.etag

        for attempt in range(self.retries + 1):
            try:
                response = self.http_client.get(
                    self.base_url + path,
                    params=dict(params or {}),
                    headers=request_headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 304 and cached is not None:
                    with self._cache_lock:
                        self._cache[cache_key] = _CacheEntry(
                            stored_at=time.monotonic(),
                            value=cached.value,
                            etag=cached.etag,
                        )
                    return cached.value
                if response.status_code == 200:
                    value = response.json()
                    response_headers = getattr(response, "headers", {}) or {}
                    etag = response_headers.get("etag") or response_headers.get("ETag")
                    with self._cache_lock:
                        self._cache[cache_key] = _CacheEntry(
                            stored_at=time.monotonic(),
                            value=value,
                            etag=str(etag) if etag else None,
                        )
                    return value
                last_error = f"provider returned HTTP {response.status_code}"
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except (httpx.HTTPError, OSError, TimeoutError, ValueError) as exc:
                last_error = f"provider request failed ({type(exc).__name__})"
            if attempt < self.retries:
                self.sleep_fn(0.1 * (2**attempt))
        raise FintrafficUnavailable(
            f"Fintraffic current data is unavailable: {last_error}."
        )

    def _portnet_rows(
        self,
        payload: Any,
        *,
        snapshot: datetime,
        horizon_end: datetime,
        locode: Optional[str],
        mmsi: Optional[str],
        imo: Optional[str],
        vessel_name: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], datetime]:
        if not isinstance(payload, dict) or not isinstance(payload.get("portCalls"), list):
            raise FintrafficUnavailable("Fintraffic Portnet returned an invalid response shape.")
        data_updated = _as_utc(payload.get("dataUpdatedTime"))
        if data_updated is None:
            raise FintrafficUnavailable("Fintraffic Portnet omitted its data-updated timestamp.")
        age = snapshot - data_updated
        if age > PORTNET_MAX_AGE or data_updated - snapshot > FUTURE_CLOCK_TOLERANCE:
            raise FintrafficUnavailable(
                "Fintraffic Portnet data is stale or has an invalid future timestamp."
            )

        rows: List[Dict[str, Any]] = []
        for call in payload["portCalls"]:
            if not isinstance(call, dict):
                continue
            call_locode = normalize_finnish_port(str(call.get("portToVisit") or ""))
            call_mmsi = _digits(call.get("mmsi"))
            call_imo = _digits(call.get("imoLloyds"))
            call_name = re.sub(r"\s+", " ", str(call.get("vesselName") or "")).strip()
            if locode and call_locode != locode:
                continue
            if mmsi and call_mmsi != mmsi:
                continue
            if imo and call_imo != imo:
                continue
            if vessel_name and _normalized_text(call_name) != _normalized_text(vessel_name):
                continue
            if not call_locode:
                continue

            candidates: List[Dict[str, Any]] = []
            for detail in call.get("portAreaDetails") or []:
                if not isinstance(detail, dict):
                    continue
                eta = _as_utc(detail.get("eta"))
                if eta is None or eta < snapshot or eta > horizon_end:
                    continue
                candidates.append(
                    {
                        "official_eta_utc": eta,
                        "official_eta_updated_utc": _as_utc(detail.get("etaTimestamp")),
                        "official_eta_source": str(detail.get("etaSource") or "Portnet"),
                        "port_area_code": detail.get("portAreaCode"),
                        "port_area_name": detail.get("portAreaName"),
                        "berth_code": detail.get("berthCode"),
                        "berth_name": detail.get("berthName"),
                    }
                )
            if not candidates:
                continue
            selected = min(candidates, key=lambda item: item["official_eta_utc"])
            port_call_id = str(call.get("portCallId") or "").strip()
            if not port_call_id:
                continue
            rows.append(
                {
                    "snapshot_time_utc": snapshot,
                    "portnet_data_updated_utc": data_updated,
                    "port_call_id": port_call_id,
                    "port_call_updated_utc": _as_utc(call.get("portCallTimestamp")),
                    "port_locode": call_locode,
                    "vessel_name": call_name or None,
                    "vessel_label": call_name or call_mmsi or call_imo or "Unknown vessel",
                    "mmsi": call_mmsi,
                    "imo": call_imo,
                    **selected,
                    "ais_eta_utc": None,
                    "ais_metadata_time_utc": None,
                    "ais_location_time_utc": None,
                    "latitude": None,
                    "longitude": None,
                    "sog_kn": None,
                    "announced_delay_minutes": None,
                    "variance_status": "ais_unavailable",
                }
            )
        deduped_rows: List[Dict[str, Any]] = []
        by_call_id: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            existing = by_call_id.get(row["port_call_id"])
            if existing is None:
                by_call_id[row["port_call_id"]] = row
                deduped_rows.append(row)
                continue
            identity = (
                row["port_locode"],
                row["mmsi"],
                row["imo"],
                row["official_eta_utc"],
            )
            existing_identity = (
                existing["port_locode"],
                existing["mmsi"],
                existing["imo"],
                existing["official_eta_utc"],
            )
            if identity != existing_identity:
                raise FintrafficUnavailable(
                    "Fintraffic Portnet returned conflicting duplicate port-call identifiers."
                )
        deduped_rows.sort(
            key=lambda row: (
                row["official_eta_utc"],
                row["port_locode"],
                row["vessel_label"],
                row["port_call_id"],
            )
        )
        return deduped_rows, data_updated

    def _enrich_with_ais(
        self,
        rows: List[Dict[str, Any]],
        *,
        snapshot: datetime,
        horizon_end: datetime,
    ) -> List[Dict[str, Any]]:
        by_mmsi: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            if row.get("mmsi"):
                by_mmsi.setdefault(str(row["mmsi"]), []).append(row)

        expected_imo_by_mmsi: Dict[str, Optional[str]] = {}
        identity_conflicts: set[str] = set()
        for mmsi, vessel_rows in by_mmsi.items():
            expected_imos = {
                str(row["imo"])
                for row in vessel_rows
                if row.get("imo")
            }
            if len(expected_imos) > 1:
                identity_conflicts.add(mmsi)
                for row in vessel_rows:
                    row["variance_status"] = "portnet_identity_conflict"
                continue
            expected_imo_by_mmsi[mmsi] = next(iter(expected_imos), None)

        # A schedule page can include many vessels. Fetch independent AIS
        # records concurrently so a cold live query is bounded by a small
        # number of provider round trips rather than two serial calls per row.
        valid_mmsis = [
            mmsi for mmsi in by_mmsi if mmsi not in identity_conflicts
        ]
        validations: Dict[
            str,
            Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str],
        ] = {}
        if valid_mmsis:
            worker_count = min(8, len(valid_mmsis))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="eagle-eye-ais",
            ) as executor:
                futures = {
                    mmsi: executor.submit(
                        self._validated_ais,
                        mmsi,
                        expected_imo=expected_imo_by_mmsi[mmsi],
                        snapshot=snapshot,
                    )
                    for mmsi in valid_mmsis
                }
                for mmsi in valid_mmsis:
                    validations[mmsi] = futures[mmsi].result()

        for mmsi, vessel_rows in by_mmsi.items():
            if mmsi in identity_conflicts:
                continue
            metadata, location, reason = validations[mmsi]
            if metadata is None or location is None:
                for row in vessel_rows:
                    row["variance_status"] = reason
                continue
            destination = metadata["destination"]
            matching_rows = [
                row
                for row in vessel_rows
                if _destination_matches(destination, str(row["port_locode"]))
            ]
            if not matching_rows:
                for row in vessel_rows:
                    row["variance_status"] = "ais_destination_mismatch"
                continue
            plausible_rows: List[Tuple[Dict[str, Any], datetime, timedelta]] = []
            for row in matching_rows:
                # AIS carries MMDDHHMM without a year.  Portnet's scheduled
                # ETA is the semantic year anchor for this exact call.
                decoded = _decode_ais_eta(
                    metadata["encoded_eta"],
                    row["official_eta_utc"],
                )
                if decoded is None:
                    continue
                variance = decoded - row["official_eta_utc"]
                if abs(variance) > MAX_ANNOUNCED_VARIANCE:
                    continue
                if decoded < snapshot or decoded > horizon_end:
                    continue
                plausible_rows.append((row, decoded, variance))
            if len(plausible_rows) != 1:
                status = (
                    "ais_schedule_conflict"
                    if len(plausible_rows) > 1
                    else "ais_eta_not_matchable"
                )
                for row in matching_rows:
                    row["variance_status"] = status
                continue
            selected, ais_eta, variance = plausible_rows[0]
            for row in vessel_rows:
                row["ais_metadata_time_utc"] = metadata["ais_metadata_time_utc"]
                row["ais_location_time_utc"] = location["ais_location_time_utc"]
                row["latitude"] = location["latitude"]
                row["longitude"] = location["longitude"]
                row["sog_kn"] = location["sog_kn"]
                if row is selected:
                    row["ais_eta_utc"] = ais_eta
                    row["announced_delay_minutes"] = int(round(variance.total_seconds() / 60.0))
                    row["variance_status"] = "validated"
                elif row["variance_status"] == "ais_unavailable":
                    row["variance_status"] = "not_next_announced_call"
        return rows

    def _validated_ais(
        self,
        mmsi: str,
        *,
        expected_imo: Optional[str],
        snapshot: datetime,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
        try:
            metadata_payload = self._get_json(
                FINTRAFFIC_AIS_VESSEL_PATH.format(mmsi=mmsi)
            )
            from_ms, to_ms = _ais_window_milliseconds(snapshot)
            location_payload = self._get_json(
                FINTRAFFIC_AIS_LOCATIONS_PATH,
                {"mmsi": int(mmsi), "from": from_ms, "to": to_ms},
            )
        except FintrafficUnavailable:
            return None, None, "ais_unavailable"

        if not isinstance(metadata_payload, dict) or _digits(metadata_payload.get("mmsi")) != mmsi:
            return None, None, "ais_identity_mismatch"
        if expected_imo and _digits(metadata_payload.get("imo")) != expected_imo:
            return None, None, "ais_imo_mismatch"
        metadata_time = _as_utc(
            datetime.fromtimestamp(
                int(metadata_payload.get("timestamp", 0)) / 1000.0,
                tz=timezone.utc,
            )
            if metadata_payload.get("timestamp")
            else None
        )
        if metadata_time is None:
            return None, None, "ais_metadata_time_missing"
        if snapshot - metadata_time > AIS_METADATA_MAX_AGE or metadata_time - snapshot > FUTURE_CLOCK_TOLERANCE:
            return None, None, "ais_metadata_stale"
        encoded_eta = metadata_payload.get("eta")
        try:
            int(encoded_eta)
        except (TypeError, ValueError):
            return None, None, "ais_eta_invalid"

        if not isinstance(location_payload, dict) or not isinstance(location_payload.get("features"), list):
            return None, None, "ais_location_unavailable"
        location_data_updated = _as_utc(location_payload.get("dataUpdatedTime"))
        if location_data_updated is None:
            return None, None, "ais_location_snapshot_missing"
        if (
            snapshot - location_data_updated > AIS_COLLECTION_MAX_AGE
            or location_data_updated - snapshot > FUTURE_CLOCK_TOLERANCE
        ):
            return None, None, "ais_location_snapshot_stale"
        valid_locations: List[Dict[str, Any]] = []
        for feature in location_payload["features"]:
            if not isinstance(feature, dict) or _digits(feature.get("mmsi")) != mmsi:
                continue
            properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
            geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
            coordinates = geometry.get("coordinates")
            if not isinstance(coordinates, list) or len(coordinates) < 2:
                continue
            try:
                longitude = float(coordinates[0])
                latitude = float(coordinates[1])
            except (TypeError, ValueError):
                continue
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                continue
            external_ms = properties.get("timestampExternal")
            if external_ms is None:
                continue
            try:
                location_time = datetime.fromtimestamp(
                    int(external_ms) / 1000.0,
                    tz=timezone.utc,
                )
            except (TypeError, ValueError, OSError):
                continue
            if snapshot - location_time > AIS_LOCATION_MAX_AGE:
                continue
            if location_time - snapshot > FUTURE_CLOCK_TOLERANCE:
                continue
            sog_raw = properties.get("sog")
            try:
                sog_kn = float(sog_raw)
            except (TypeError, ValueError):
                sog_kn = None
            if sog_kn is not None and (sog_kn < 0 or sog_kn >= 102.3):
                sog_kn = None
            valid_locations.append(
                {
                    "ais_location_time_utc": location_time,
                    "latitude": latitude,
                    "longitude": longitude,
                    "sog_kn": sog_kn,
                }
            )
        if not valid_locations:
            return None, None, "ais_location_stale"
        latest_location = max(valid_locations, key=lambda item: item["ais_location_time_utc"])
        return (
            {
                "ais_metadata_time_utc": metadata_time,
                "encoded_eta": encoded_eta,
                "destination": metadata_payload.get("destination"),
            },
            latest_location,
            "validated",
        )

    @staticmethod
    def _frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        columns = [
            "snapshot_time_utc",
            "portnet_data_updated_utc",
            "port_call_id",
            "port_call_updated_utc",
            "port_locode",
            "vessel_name",
            "mmsi",
            "imo",
            "official_eta_utc",
            "official_eta_updated_utc",
            "official_eta_source",
            "port_area_code",
            "port_area_name",
            "berth_code",
            "berth_name",
            "ais_destination",
            "ais_destination_match",
            "ais_eta_utc",
            "ais_metadata_time_utc",
            "ais_location_time_utc",
            "latitude",
            "longitude",
            "sog_kn",
            "announced_delay_minutes",
            "variance_status",
            "source_scope",
        ]
        frame = pd.DataFrame(list(rows))
        if frame.empty:
            return pd.DataFrame(columns=columns)
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        return frame[columns].sort_values(
            ["official_eta_utc", "ais_eta_utc", "port_locode", "vessel_name"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

    @staticmethod
    def _coverage(
        snapshot: datetime,
        horizon_end: datetime,
        data_updated: datetime,
        rows: Iterable[Dict[str, Any]],
    ) -> List[str]:
        materialized = list(rows)
        validated = sum(
            1 for row in materialized if row.get("announced_delay_minutes") is not None
        )
        return [
            f"Retrieval snapshot: {_iso_utc(snapshot)}",
            f"Portnet data updated: {_iso_utc(data_updated)}",
            f"UTC live horizon: {_iso_utc(snapshot)} to {_iso_utc(horizon_end)}",
            f"Portnet rows: {len(materialized)}",
            f"Validated AIS matches: {validated}",
            "Announced variance: AIS ETA minus Portnet official scheduled ETA.",
        ]

    @staticmethod
    def _unavailable(
        snapshot: datetime,
        reason: str,
        *,
        horizon_end: Optional[datetime] = None,
        failure_code: Optional[str] = None,
        source_kind: str = "portnet_with_ais",
    ) -> LiveETAResult:
        return LiveETAResult(
            status="no_current_data",
            answer=reason,
            table=None,
            coverage_notes=[f"Retrieval snapshot: {_iso_utc(snapshot)}"],
            caveats=["Current-source data is unavailable for this request."],
            snapshot_at=snapshot,
            horizon_end=horizon_end,
            failure_reason=failure_code or reason,
            source_kind=source_kind,
        )
