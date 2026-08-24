"""Backend-only AISStream collector and deterministic operational queries.

AISStream is a live AIS broadcast source, not an official port schedule and
not a prediction service.  This module therefore keeps vessel-reported ETA,
destination and movement observations explicitly separate from official ETA
semantics.

The collector owns one WebSocket connection, sends the API key only in the
subscription frame, normalizes the small subset of AIS messages needed for
vessel monitoring, and persists a bounded 24-hour observation history.  It is
deliberately independent from the canonical query planner/service so it can be
started and integrated by the application lifecycle in a later change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import pandas as pd

from .fintraffic import BALTIC_PORT_ALIASES


AISSTREAM_PROVIDER = "aisstream"
AISSTREAM_WEBSOCKET_URL = "wss://stream.aisstream.io/v0/stream"

# AISStream documents bounding-box corners as [latitude, longitude].  This
# conservative box covers the Baltic/Kattegat operating region without opening
# a whole-world stream.
BALTIC_BOUNDING_BOX: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
    ((53.0, 9.0), (66.0, 31.5)),
)

AISSTREAM_MESSAGE_TYPES: Tuple[str, ...] = (
    "PositionReport",
    "ShipStaticData",
)

POSITION_MESSAGE_TYPES = frozenset({"PositionReport"})

AISSTREAM_QUERY_OPERATIONS = frozenset(
    {
        "inbound_watchlist",
        "vessel_status",
        "low_speed",
        "destination_load",
        "eta_revisions",
        "stale_missing",
        "shift_handover",
    }
)

DEFAULT_HISTORY_HOURS = 24
DEFAULT_STALE_AFTER_MINUTES = 15
DEFAULT_HORIZON_HOURS = 24
DEFAULT_SPEED_THRESHOLD_KN = 2.0
DEFAULT_ETA_CHANGE_THRESHOLD_MINUTES = 30
DEFAULT_CHANGE_WINDOW_MINUTES = 60
MAX_QUERY_HORIZON_HOURS = 48
MAX_QUERY_LIMIT = 500


class AISStreamSocket(Protocol):
    """Minimal socket surface used by the collector and frozen tests."""

    async def send(self, message: str) -> None:
        ...

    def __aiter__(self) -> "AISStreamSocket":
        ...

    async def __anext__(self) -> Any:
        ...

    async def close(self) -> None:
        ...


TransportFactory = Callable[[str], AsyncContextManager[AISStreamSocket]]
Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]


class AISStreamError(RuntimeError):
    """Base class for safe, non-secret collector failures."""


class AISStreamConfigurationError(AISStreamError):
    """Raised when the collector cannot create a valid subscription."""


class AISStreamProtocolError(AISStreamError):
    """Raised for provider error frames or invalid envelope contracts."""


@dataclass(frozen=True)
class AISStreamSourceHealth:
    """Secret-free snapshot of collector and source state."""

    provider: str
    status: str
    connected: bool
    api_key_configured: bool
    last_connected_at: Optional[datetime]
    last_message_at: Optional[datetime]
    last_error_at: Optional[datetime]
    last_error_code: Optional[str]
    consecutive_failures: int
    reconnect_attempts: int
    next_retry_seconds: Optional[float]
    accepted_messages: int
    duplicate_messages: int
    out_of_order_messages: int
    invalid_messages: int
    cached_vessels: int
    history_hours: int


@dataclass
class AISStreamVesselState:
    """Latest normalized, null-preserving state for one MMSI."""

    mmsi: str
    vessel_name: Optional[str] = None
    imo: Optional[str] = None
    call_sign: Optional[str] = None
    ship_type: Optional[int] = None
    destination_raw: Optional[str] = None
    destination_locode: Optional[str] = None
    destination_name: Optional[str] = None
    destination_match: Optional[str] = None
    eta_utc: Optional[datetime] = None
    eta_observed_at: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    sog_kn: Optional[float] = None
    cog_deg: Optional[float] = None
    heading_deg: Optional[int] = None
    navigational_status: Optional[int] = None
    position_observed_at: Optional[datetime] = None
    static_observed_at: Optional[datetime] = None
    last_seen_utc: Optional[datetime] = None
    last_message_type: Optional[str] = None

    def public_row(
        self,
        *,
        snapshot: datetime,
        stale_after: timedelta,
    ) -> Dict[str, Any]:
        position_age = _age_minutes(snapshot, self.position_observed_at)
        static_age = _age_minutes(snapshot, self.static_observed_at)
        eta_minutes = (
            int(round((self.eta_utc - snapshot).total_seconds() / 60.0))
            if self.eta_utc is not None
            else None
        )
        return {
            "snapshot_time_utc": _iso_utc(snapshot),
            "mmsi": self.mmsi,
            "imo": self.imo,
            "vessel_name": self.vessel_name,
            "call_sign": self.call_sign,
            "ship_type": self.ship_type,
            "destination_raw": self.destination_raw,
            "destination_locode": self.destination_locode,
            "destination_name": self.destination_name,
            "destination_match": self.destination_match,
            "eta_utc": _optional_iso(self.eta_utc),
            "eta_observed_at_utc": _optional_iso(self.eta_observed_at),
            "time_to_eta_minutes": eta_minutes,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "sog_kn": self.sog_kn,
            "cog_deg": self.cog_deg,
            "heading_deg": self.heading_deg,
            "navigational_status": self.navigational_status,
            "position_observed_at_utc": _optional_iso(self.position_observed_at),
            "static_observed_at_utc": _optional_iso(self.static_observed_at),
            "last_seen_utc": _optional_iso(self.last_seen_utc),
            "position_age_minutes": position_age,
            "static_age_minutes": static_age,
            "position_stale": (
                self.position_observed_at is None
                or snapshot - self.position_observed_at > stale_after
            ),
            "static_stale": (
                self.static_observed_at is None
                or snapshot - self.static_observed_at > stale_after
            ),
            "last_message_type": self.last_message_type,
            "source_scope": "aisstream_baltic_broadcast",
        }


@dataclass
class AISStreamQueryResult:
    """Executor-friendly deterministic result for AISStream operations."""

    status: str
    operation: str
    answer: str
    table: Optional[pd.DataFrame]
    snapshot_at: datetime
    data_updated_at: Optional[datetime]
    horizon_end: Optional[datetime]
    summary: Dict[str, Any] = field(default_factory=dict)
    sections: Dict[str, pd.DataFrame] = field(default_factory=dict)
    coverage_notes: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    source_kind: str = "aisstream_observation"
    health: Optional[AISStreamSourceHealth] = None

    @property
    def chart(self) -> Optional[pd.DataFrame]:
        return self.table


@dataclass(frozen=True)
class _Destination:
    raw: Optional[str]
    locode: Optional[str]
    name: Optional[str]
    match: Optional[str]


@dataclass(frozen=True)
class _NormalizedMessage:
    event_id: str
    message_type: str
    ordering_key: str
    mmsi: str
    observed_at: datetime
    received_at: datetime
    body: Dict[str, Any]
    metadata: Dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return _ensure_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_iso(value: Optional[datetime]) -> Optional[str]:
    return _iso_utc(value) if value is not None else None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, (int, float)):
        try:
            raw = float(value)
            if abs(raw) > 10_000_000_000:
                raw /= 1000.0
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    token = str(value).strip()
    if not token:
        return None
    token = re.sub(r"\s+UTC$", "", token, flags=re.IGNORECASE).strip()
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        return _ensure_utc(datetime.fromisoformat(token))
    except ValueError:
        pass
    for pattern in (
        "%Y-%m-%d %H:%M:%S.%f %z",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return _ensure_utc(datetime.strptime(token, pattern))
        except ValueError:
            continue
    return None


def _age_minutes(snapshot: datetime, observed_at: Optional[datetime]) -> Optional[int]:
    if observed_at is None:
        return None
    return max(0, int(round((snapshot - observed_at).total_seconds() / 60.0)))


def _normalized_text(value: Any) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value or "")).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return re.sub(r"[^A-Z0-9]+", "", ascii_value.upper())


def _clean_ais_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).replace("@", " ")
    token = re.sub(r"\s+", " ", token).strip()
    return token or None


def _digits(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = re.sub(r"\D", "", str(value))
    return token or None


def _valid_mmsi(value: Any) -> Optional[str]:
    token = _digits(value)
    return token if token and re.fullmatch(r"\d{9}", token) else None


def _optional_imo(value: Any) -> Optional[str]:
    token = _digits(value)
    return token if token and re.fullmatch(r"\d{7}", token) and token != "0000000" else None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(result):
        return None
    return result


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_AISSTREAM_EXTRA_ALIASES: Dict[str, str] = {
    # Gothenburg is a core Eagle Eye operating port but is absent from the
    # existing Fintraffic foreign-destination vocabulary.
    "gothenburg": "SEGOT",
    "goteborg": "SEGOT",
}
_ETA_WATCH_COUNTRY_PREFIXES = frozenset(
    {"SE", "FI", "EE", "LV", "LT", "PL", "DE", "DK"}
)
AISSTREAM_DESTINATION_ALIASES: Dict[str, str] = {
    **{
        name: locode
        for name, locode in BALTIC_PORT_ALIASES.items()
        if locode[:2] in _ETA_WATCH_COUNTRY_PREFIXES
    },
    **_AISSTREAM_EXTRA_ALIASES,
}
_ALIAS_BY_NORMALIZED_NAME: Dict[str, str] = {
    _normalized_text(name): locode
    for name, locode in AISSTREAM_DESTINATION_ALIASES.items()
}
_CURATED_LOCODE_SET = frozenset(AISSTREAM_DESTINATION_ALIASES.values())
_PREFERRED_DESTINATION_NAMES: Dict[str, str] = {
    "SEGOT": "Gothenburg",
}
for _alias, _locode in AISSTREAM_DESTINATION_ALIASES.items():
    _PREFERRED_DESTINATION_NAMES.setdefault(_locode, _alias.title())


def normalize_aisstream_destination(value: Any) -> _Destination:
    """Conservatively normalize an exact curated port token.

    Route-like strings, unknown five-character shapes and partial names remain
    unverified.  Their source text is preserved, while the locode stays null.
    """

    raw = _clean_ais_text(value)
    if raw is None:
        return _Destination(None, None, None, None)
    compact = _normalized_text(raw)
    if compact in _CURATED_LOCODE_SET:
        return _Destination(
            raw,
            compact,
            _PREFERRED_DESTINATION_NAMES.get(compact),
            "exact_curated_unlocode",
        )
    locode = _ALIAS_BY_NORMALIZED_NAME.get(compact)
    if locode:
        return _Destination(
            raw,
            locode,
            _PREFERRED_DESTINATION_NAMES.get(locode),
            "exact_curated_port_name",
        )
    return _Destination(raw, None, None, None)


def infer_ais_eta(
    eta: Any,
    reference: datetime,
) -> Optional[datetime]:
    """Attach the nearest plausible UTC year to AIS type-5 MMDDHHMM fields."""

    if not isinstance(eta, dict):
        return None
    month = _int_or_none(eta.get("Month"))
    day = _int_or_none(eta.get("Day"))
    hour = _int_or_none(eta.get("Hour"))
    minute = _int_or_none(eta.get("Minute"))
    if None in {month, day, hour, minute}:
        return None
    assert month is not None
    assert day is not None
    assert hour is not None
    assert minute is not None
    if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    anchor = _ensure_utc(reference)
    candidates: List[datetime] = []
    for year in (anchor.year - 1, anchor.year, anchor.year + 1):
        try:
            candidates.append(
                datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            )
        except ValueError:
            continue
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs((item - anchor).total_seconds()))


def _default_transport_factory(
    url: str,
) -> AsyncContextManager[AISStreamSocket]:
    import websockets

    return websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10,
        max_size=None,
    )


def _frame(rows: Iterable[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    materialized = list(rows)
    if not materialized:
        return None
    return pd.DataFrame(materialized)


def _frame_or_empty(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    frame = _frame(rows)
    return frame if frame is not None else pd.DataFrame()


class AISStreamCollector:
    """Single-connection async AISStream collector with bounded local history."""

    provider = AISSTREAM_PROVIDER

    def __init__(
        self,
        sqlite_path: str | Path,
        *,
        api_key: Optional[str] = None,
        clock: Clock = _utc_now,
        transport_factory: Optional[TransportFactory] = None,
        sleep_fn: Sleep = asyncio.sleep,
        history_hours: int = DEFAULT_HISTORY_HOURS,
        stale_after_minutes: int = DEFAULT_STALE_AFTER_MINUTES,
        reconnect_base_seconds: float = 1.0,
        reconnect_max_seconds: float = 60.0,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._api_key = str(
            api_key if api_key is not None else os.getenv("AISSTREAM_API_KEY", "")
        ).strip()
        self.clock = clock
        self.transport_factory = transport_factory or _default_transport_factory
        self.sleep_fn = sleep_fn
        self.history_hours = max(1, min(int(history_hours), 24))
        self.stale_after = timedelta(
            minutes=max(1, int(stale_after_minutes))
        )
        self.reconnect_base_seconds = max(0.0, float(reconnect_base_seconds))
        self.reconnect_max_seconds = max(
            self.reconnect_base_seconds,
            float(reconnect_max_seconds),
        )

        self._lock = threading.RLock()
        self._states: Dict[str, AISStreamVesselState] = {}
        self._last_event_by_key: Dict[Tuple[str, str], datetime] = {}
        self._seen_event_ids: set[str] = set()
        self._stop_event = asyncio.Event()
        self._active_socket: Optional[AISStreamSocket] = None

        self._connected = False
        self._run_started = False
        self._stop_requested = False
        self._last_connected_at: Optional[datetime] = None
        self._last_message_at: Optional[datetime] = None
        self._last_error_at: Optional[datetime] = None
        self._last_error_code: Optional[str] = None
        self._consecutive_failures = 0
        self._reconnect_attempts = 0
        self._next_retry_seconds: Optional[float] = None
        self._accepted_messages = 0
        self._duplicate_messages = 0
        self._out_of_order_messages = 0
        self._invalid_messages = 0

        self._initialize_database()
        self._restore_cache()

    def capabilities(self) -> Dict[str, Any]:
        health = self.source_health()
        return {
            "provider": self.provider,
            "available": bool(self._api_key),
            "source_health": health.status,
            "transport": "backend_websocket",
            "websocket_url": AISSTREAM_WEBSOCKET_URL,
            "bounding_boxes": [
                [[corner[0], corner[1]] for corner in box]
                for box in BALTIC_BOUNDING_BOX
            ],
            "message_types": list(AISSTREAM_MESSAGE_TYPES),
            "operations": sorted(AISSTREAM_QUERY_OPERATIONS),
            "history_hours": self.history_hours,
            "maximum_horizon_hours": MAX_QUERY_HORIZON_HOURS,
            "default_stale_after_minutes": int(
                self.stale_after.total_seconds() // 60
            ),
            "maximum_subscription_mmsis": 50,
            "official_schedule": False,
            "official_schedule_country_scope": [],
            "country_scope": sorted(_ETA_WATCH_COUNTRY_PREFIXES),
            "ais_destination_country_scope": sorted(
                _ETA_WATCH_COUNTRY_PREFIXES
            ),
            "timezone": "UTC",
            "source_kind": "vessel_reported_ais",
            "prediction": False,
            "official_eta_authority": None,
            "announced_variance": None,
            "regional_ais_scope": (
                "Fresh, non-exhaustive vessel-reported AIS destinations, ETAs, "
                "positions, speeds, and ETA revisions for curated Baltic ports."
            ),
            "api_key_configured": bool(self._api_key),
            "api_key_exposed": False,
            "reason": (
                None
                if self._api_key
                else "AISSTREAM_API_KEY is not configured in this runtime."
            ),
        }

    def source_health(self) -> AISStreamSourceHealth:
        snapshot = _ensure_utc(self.clock())
        with self._lock:
            has_fresh_position = any(
                state.position_observed_at is not None
                and snapshot - state.position_observed_at <= self.stale_after
                for state in self._states.values()
            )
            has_fresh_static = any(
                state.static_observed_at is not None
                and snapshot - state.static_observed_at <= self.stale_after
                for state in self._states.values()
            )
            field_families_ready = has_fresh_position and has_fresh_static
            if self._stop_requested:
                status = "unavailable"
            elif self._connected:
                if self._last_message_at is None:
                    status = "warming"
                elif snapshot - self._last_message_at > self.stale_after:
                    status = "stale"
                elif not field_families_ready:
                    status = "warming"
                else:
                    status = "live"
            elif (
                field_families_ready
                and self._last_message_at is not None
                and snapshot - self._last_message_at <= self.stale_after
            ):
                status = "live"
            elif (
                self._last_message_at is not None
                and snapshot - self._last_message_at <= self.stale_after
            ):
                status = "warming"
            elif self._consecutive_failures:
                status = "connecting"
            elif self._run_started:
                status = "connecting"
            elif not self._api_key:
                status = "unavailable"
            else:
                status = "warming"
            return AISStreamSourceHealth(
                provider=self.provider,
                status=status,
                connected=self._connected,
                api_key_configured=bool(self._api_key),
                last_connected_at=self._last_connected_at,
                last_message_at=self._last_message_at,
                last_error_at=self._last_error_at,
                last_error_code=self._last_error_code,
                consecutive_failures=self._consecutive_failures,
                reconnect_attempts=self._reconnect_attempts,
                next_retry_seconds=self._next_retry_seconds,
                accepted_messages=self._accepted_messages,
                duplicate_messages=self._duplicate_messages,
                out_of_order_messages=self._out_of_order_messages,
                invalid_messages=self._invalid_messages,
                cached_vessels=len(self._states),
                history_hours=self.history_hours,
            )

    def request_stop(self) -> None:
        """Request shutdown without awaiting the active socket close."""

        self._stop_requested = True
        self._stop_event.set()

    async def stop(self) -> None:
        """Stop the collector and close the current socket when possible."""

        self.request_stop()
        socket = self._active_socket
        if socket is not None:
            try:
                await socket.close()
            except Exception:
                # Shutdown is best-effort.  Do not surface provider/socket text
                # that could accidentally include connection details.
                pass

    async def run(self, *, max_connections: Optional[int] = None) -> None:
        """Continuously connect, subscribe, drain messages and reconnect.

        ``max_connections`` is a deterministic test/diagnostic bound. Production
        lifecycle code should omit it.
        """

        if not self._api_key:
            raise AISStreamConfigurationError(
                "AISSTREAM_API_KEY is required to start the AISStream collector."
            )
        if self._run_started and not self._stop_requested:
            raise AISStreamConfigurationError(
                "This AISStreamCollector already has an active run loop."
            )

        self._run_started = True
        self._stop_requested = False
        self._stop_event.clear()
        connection_count = 0
        try:
            while not self._stop_event.is_set():
                if max_connections is not None and connection_count >= max_connections:
                    break
                if connection_count:
                    with self._lock:
                        self._reconnect_attempts += 1
                connection_count += 1
                received_on_connection = False
                try:
                    async with self.transport_factory(
                        AISSTREAM_WEBSOCKET_URL
                    ) as socket:
                        self._active_socket = socket
                        with self._lock:
                            self._connected = True
                            self._last_connected_at = _ensure_utc(self.clock())
                            self._last_error_code = None
                            self._next_retry_seconds = None
                        await socket.send(
                            json.dumps(
                                self._subscription_payload(),
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                        )
                        async for frame in socket:
                            if self._stop_event.is_set():
                                break
                            accepted = self.ingest_message(frame)
                            received_on_connection = received_on_connection or accepted
                    if self._stop_event.is_set():
                        break
                    if max_connections is not None and connection_count >= max_connections:
                        break
                    raise AISStreamProtocolError("stream_closed")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    with self._lock:
                        self._connected = False
                        self._active_socket = None
                        self._last_error_at = _ensure_utc(self.clock())
                        self._last_error_code = _safe_error_code(exc)
                        self._consecutive_failures += 1
                    if self._stop_event.is_set():
                        break
                    if max_connections is not None and connection_count >= max_connections:
                        break
                    delay = min(
                        self.reconnect_max_seconds,
                        self.reconnect_base_seconds
                        * (2 ** max(0, self._consecutive_failures - 1)),
                    )
                    with self._lock:
                        self._next_retry_seconds = delay
                    await self.sleep_fn(delay)
                else:
                    with self._lock:
                        if received_on_connection:
                            self._consecutive_failures = 0
                finally:
                    with self._lock:
                        self._connected = False
                        self._active_socket = None
        finally:
            with self._lock:
                self._connected = False
                self._active_socket = None
                self._next_retry_seconds = None
            self._run_started = False

    def ingest_message(self, payload: str | bytes | Dict[str, Any]) -> bool:
        """Normalize and persist one AISStream event.

        Returns ``True`` only when the event advances the accepted observation
        history. Exact duplicates and older family-specific messages return
        ``False`` and never overwrite latest state.
        """

        received_at = _ensure_utc(self.clock())
        try:
            normalized = self._normalize_message(payload, received_at=received_at)
        except AISStreamProtocolError:
            with self._lock:
                self._invalid_messages += 1
            raise
        except Exception:
            with self._lock:
                self._invalid_messages += 1
            return False
        if normalized is None:
            with self._lock:
                self._invalid_messages += 1
            return False

        cutoff = received_at - timedelta(hours=self.history_hours)
        if normalized.observed_at < cutoff:
            with self._lock:
                self._out_of_order_messages += 1
            return False
        if normalized.observed_at > received_at + timedelta(minutes=10):
            with self._lock:
                self._invalid_messages += 1
            return False

        with self._lock:
            if normalized.event_id in self._seen_event_ids:
                self._duplicate_messages += 1
                return False
            ordering = (normalized.mmsi, normalized.ordering_key)
            previous_time = self._last_event_by_key.get(ordering)
            if previous_time is not None and normalized.observed_at < previous_time:
                self._out_of_order_messages += 1
                return False

            current = self._states.get(normalized.mmsi)
            state = replace(current) if current is not None else AISStreamVesselState(
                mmsi=normalized.mmsi
            )
            eta_reported, destination_reported = self._apply_update(
                state,
                normalized,
            )
            self._persist_observation(
                normalized,
                state,
                eta_reported=eta_reported,
                destination_reported=destination_reported,
            )
            self._states[normalized.mmsi] = state
            self._seen_event_ids.add(normalized.event_id)
            self._last_event_by_key[ordering] = normalized.observed_at
            self._accepted_messages += 1
            self._last_message_at = max(
                filter(
                    lambda value: value is not None,
                    (self._last_message_at, normalized.observed_at),
                )
            )
            self._consecutive_failures = 0
            self._last_error_code = None
            self._next_retry_seconds = None
            self._purge_history(received_at)
            self._prune_cache(received_at)
        return True

    def latest_state(
        self,
        mmsi: Optional[str] = None,
    ) -> AISStreamVesselState | Dict[str, AISStreamVesselState] | None:
        """Return defensive copies of latest state."""

        with self._lock:
            if mmsi is not None:
                token = _valid_mmsi(mmsi)
                state = self._states.get(token or "")
                return replace(state) if state is not None else None
            return {
                key: replace(value)
                for key, value in sorted(self._states.items())
            }

    def query(
        self,
        operation: str,
        *,
        ports: Optional[List[str]] = None,
        mmsi: Optional[str] = None,
        imo: Optional[str] = None,
        vessel_name: Optional[str] = None,
        mmsis: Optional[List[str]] = None,
        horizon_hours: Optional[int] = None,
        limit: int = 20,
        speed_threshold_kn: Optional[float] = None,
        eta_change_threshold_minutes: Optional[int] = None,
        change_window_minutes: Optional[int] = None,
        include_stale: bool = False,
    ) -> AISStreamQueryResult:
        """Execute a deterministic operational view over latest state/history."""

        snapshot = _ensure_utc(self.clock())
        operation = str(operation or "").strip().lower()
        health = self.source_health()
        if operation not in AISSTREAM_QUERY_OPERATIONS:
            return self._result(
                status="no_current_data",
                operation=operation or "unsupported",
                answer="Unsupported AISStream operational query.",
                rows=[],
                snapshot=snapshot,
                health=health,
                failure_reason="unsupported_operation",
            )

        bounded_horizon = max(
            1,
            min(
                int(horizon_hours or DEFAULT_HORIZON_HOURS),
                MAX_QUERY_HORIZON_HOURS,
            ),
        )
        horizon_end = snapshot + timedelta(hours=bounded_horizon)
        bounded_limit = max(1, min(int(limit), MAX_QUERY_LIMIT))
        speed_threshold = max(
            0.0,
            float(
                DEFAULT_SPEED_THRESHOLD_KN
                if speed_threshold_kn is None
                else speed_threshold_kn
            ),
        )
        revision_threshold = max(
            0,
            int(
                DEFAULT_ETA_CHANGE_THRESHOLD_MINUTES
                if eta_change_threshold_minutes is None
                else eta_change_threshold_minutes
            ),
        )
        revision_window = max(
            1,
            min(
                int(
                    DEFAULT_CHANGE_WINDOW_MINUTES
                    if change_window_minutes is None
                    else change_window_minutes
                ),
                self.history_hours * 60,
            ),
        )
        stale_after = self.stale_after

        selected_mmsis = {
            token
            for token in [
                *(_valid_mmsi(item) for item in (mmsis or [])),
                _valid_mmsi(mmsi),
            ]
            if token
        }
        selected_imo = _optional_imo(imo)
        selected_name = _normalized_text(vessel_name) if vessel_name else None
        selected_ports = {
            destination.locode
            for destination in (
                normalize_aisstream_destination(value) for value in (ports or [])
            )
            if destination.locode
        }
        invalid_ports = [
            str(value)
            for value in (ports or [])
            if normalize_aisstream_destination(value).locode is None
        ]
        if invalid_ports:
            return self._result(
                status="no_current_data",
                operation=operation,
                answer=(
                    "The requested destination is outside the curated AISStream "
                    "Baltic destination vocabulary."
                ),
                rows=[],
                snapshot=snapshot,
                health=health,
                horizon_end=horizon_end,
                failure_reason="destination_coverage_unavailable",
            )

        with self._lock:
            states = [
                replace(state)
                for _, state in sorted(self._states.items())
            ]
        states = [
            state
            for state in states
            if (not selected_mmsis or state.mmsi in selected_mmsis)
            and (not selected_imo or state.imo == selected_imo)
            and (
                not selected_name
                or _normalized_text(state.vessel_name) == selected_name
            )
            and (
                not selected_ports
                or state.destination_locode in selected_ports
            )
        ]

        if operation == "vessel_status":
            rows = [
                state.public_row(snapshot=snapshot, stale_after=stale_after)
                for state in states
            ]
            rows.sort(key=lambda row: (row["mmsi"], row.get("vessel_name") or ""))
            total = len(rows)
            return self._result(
                status=self._status_for_rows(rows, health),
                operation=operation,
                answer=(
                    f"AISStream has current or retained status for {total} selected vessel"
                    f"{'' if total == 1 else 's'}."
                ),
                rows=rows[:bounded_limit],
                snapshot=snapshot,
                health=health,
                horizon_end=horizon_end,
                summary={"matched_vessels": total, "displayed_vessels": min(total, bounded_limit)},
            )

        inbound = self._inbound_rows(
            states,
            snapshot=snapshot,
            horizon_end=horizon_end,
            stale_after=stale_after,
            include_stale=include_stale,
        )
        low_speed = [
            row
            for row in inbound
            if row.get("sog_kn") is not None
            and float(row["sog_kn"]) < speed_threshold
        ]
        low_speed.sort(
            key=lambda row: (
                row.get("eta_utc") or "",
                float(row.get("sog_kn") or 0),
                row["mmsi"],
            )
        )
        destination_load = self._destination_load_rows(inbound)
        revisions = self._eta_revision_rows(
            states=states,
            snapshot=snapshot,
            window_minutes=revision_window,
            threshold_minutes=revision_threshold,
        )
        stale_missing = self._stale_missing_rows(
            states,
            snapshot=snapshot,
            stale_after=stale_after,
        )

        if operation == "inbound_watchlist":
            total = len(inbound)
            return self._result(
                status=self._status_for_rows(inbound, health),
                operation=operation,
                answer=(
                    f"{total} AISStream vessel-reported inbound signal"
                    f"{'' if total == 1 else 's'} meet the {bounded_horizon}-hour "
                    "destination, ETA and freshness checks."
                ),
                rows=inbound[:bounded_limit],
                snapshot=snapshot,
                health=health,
                horizon_end=horizon_end,
                summary={
                    "matched_vessels": total,
                    "displayed_vessels": min(total, bounded_limit),
                    "horizon_hours": bounded_horizon,
                },
            )
        if operation == "low_speed":
            total = len(low_speed)
            return self._result(
                status=self._status_for_rows(low_speed, health),
                operation=operation,
                answer=(
                    f"{total} inbound vessel-reported signal"
                    f"{'' if total == 1 else 's'} are below {speed_threshold:g} knots. "
                    "This is a monitoring condition, not proof of delay."
                ),
                rows=low_speed[:bounded_limit],
                snapshot=snapshot,
                health=health,
                horizon_end=horizon_end,
                summary={
                    "matched_vessels": total,
                    "displayed_vessels": min(total, bounded_limit),
                    "speed_threshold_kn": speed_threshold,
                },
            )
        if operation == "destination_load":
            total_vessels = sum(int(row["vessel_count"]) for row in destination_load)
            return self._result(
                status=self._status_for_rows(destination_load, health),
                operation=operation,
                answer=(
                    f"{total_vessels} fresh inbound AISStream signal"
                    f"{'' if total_vessels == 1 else 's'} are grouped across "
                    f"{len(destination_load)} curated destinations."
                ),
                rows=destination_load[:bounded_limit],
                snapshot=snapshot,
                health=health,
                horizon_end=horizon_end,
                summary={
                    "destination_count": len(destination_load),
                    "matched_vessels": total_vessels,
                },
            )
        if operation == "eta_revisions":
            total = len(revisions)
            return self._result(
                status=self._status_for_rows(revisions, health),
                operation=operation,
                answer=(
                    f"{total} vessel-reported ETA revision"
                    f"{'' if total == 1 else 's'} meet the "
                    f"{revision_threshold}-minute threshold in the last "
                    f"{revision_window} minutes."
                ),
                rows=revisions[:bounded_limit],
                snapshot=snapshot,
                health=health,
                horizon_end=horizon_end,
                summary={
                    "matched_vessels": total,
                    "eta_change_threshold_minutes": revision_threshold,
                    "change_window_minutes": revision_window,
                },
            )
        if operation == "stale_missing":
            total = len(stale_missing)
            return self._result(
                status=self._status_for_rows(stale_missing, health),
                operation=operation,
                answer=(
                    f"{total} selected vessel state"
                    f"{'' if total == 1 else 's'} have stale, missing or "
                    "unrecognized operational fields."
                ),
                rows=stale_missing[:bounded_limit],
                snapshot=snapshot,
                health=health,
                horizon_end=horizon_end,
                summary={
                    "matched_vessels": total,
                    "stale_after_minutes": int(stale_after.total_seconds() // 60),
                },
            )

        # shift_handover
        sections = {
            "inbound_watchlist": _frame_or_empty(inbound[:bounded_limit]),
            "low_speed": _frame_or_empty(low_speed[:bounded_limit]),
            "destination_load": _frame_or_empty(
                destination_load[:bounded_limit]
            ),
            "eta_revisions": _frame_or_empty(revisions[:bounded_limit]),
            "stale_missing": _frame_or_empty(stale_missing[:bounded_limit]),
        }
        summary = {
            "inbound_vessels": len(inbound),
            "low_speed_vessels": len(low_speed),
            "destination_count": len(destination_load),
            "eta_revision_vessels": len(revisions),
            "stale_or_missing_vessels": len(stale_missing),
            "horizon_hours": bounded_horizon,
            "speed_threshold_kn": speed_threshold,
            "eta_change_threshold_minutes": revision_threshold,
            "change_window_minutes": revision_window,
        }
        primary_rows = inbound[:bounded_limit]
        has_any = any(summary[key] for key in (
            "inbound_vessels",
            "low_speed_vessels",
            "destination_count",
            "eta_revision_vessels",
            "stale_or_missing_vessels",
        ))
        return self._result(
            status=(
                "ok"
                if has_any
                else self._status_for_rows([], health)
            ),
            operation=operation,
            answer=(
                "Shift handover: "
                f"{len(inbound)} inbound, {len(low_speed)} below {speed_threshold:g} knots, "
                f"{len(revisions)} ETA revisions, and {len(stale_missing)} stale or missing "
                "vessel states."
            ),
            rows=primary_rows,
            snapshot=snapshot,
            health=health,
            horizon_end=horizon_end,
            summary=summary,
            sections=sections,
        )

    def _subscription_payload(self) -> Dict[str, Any]:
        if not self._api_key:
            raise AISStreamConfigurationError(
                "AISSTREAM_API_KEY is required to create a subscription."
            )
        return {
            "APIKey": self._api_key,
            "BoundingBoxes": [
                [[corner[0], corner[1]] for corner in box]
                for box in BALTIC_BOUNDING_BOX
            ],
            "FilterMessageTypes": list(AISSTREAM_MESSAGE_TYPES),
        }

    def _normalize_message(
        self,
        payload: str | bytes | Dict[str, Any],
        *,
        received_at: datetime,
    ) -> Optional[_NormalizedMessage]:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            parsed = json.loads(payload)
        else:
            parsed = dict(payload)
        if not isinstance(parsed, dict):
            return None
        if "error" in parsed:
            # Never retain or repeat provider-supplied error text. It can include
            # authentication detail.
            raise AISStreamProtocolError("provider_error")

        message_type = str(parsed.get("MessageType") or "").strip()
        if message_type not in AISSTREAM_MESSAGE_TYPES:
            return None
        message = parsed.get("Message")
        if not isinstance(message, dict):
            return None
        body = message.get(message_type)
        if not isinstance(body, dict):
            return None
        if body.get("Valid") is False:
            return None
        metadata = parsed.get("MetaData")
        if not isinstance(metadata, dict):
            # Accept the website's inconsistent spelling defensively, while the
            # subscription and fixtures retain canonical OpenAPI spelling.
            metadata = parsed.get("Metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        mmsi = _valid_mmsi(
            body.get("UserID")
            or metadata.get("MMSI")
            or metadata.get("mmsi")
        )
        if not mmsi:
            return None
        observed_at = (
            _parse_datetime(
                metadata.get("time_utc")
                or metadata.get("TimeUTC")
                or metadata.get("timestamp")
            )
            or received_at
        )
        if message_type in POSITION_MESSAGE_TYPES:
            ordering_key = "position"
        elif message_type == "ShipStaticData":
            ordering_key = "voyage_static"
        else:
            part = "a" if body.get("PartNumber") is False else "b"
            ordering_key = f"class_b_static_{part}"
        canonical = json.dumps(parsed, separators=(",", ":"), sort_keys=True)
        event_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return _NormalizedMessage(
            event_id=event_id,
            message_type=message_type,
            ordering_key=ordering_key,
            mmsi=mmsi,
            observed_at=observed_at,
            received_at=received_at,
            body=body,
            metadata=metadata,
        )

    def _apply_update(
        self,
        state: AISStreamVesselState,
        message: _NormalizedMessage,
    ) -> Tuple[bool, bool]:
        body = message.body
        observed_at = message.observed_at
        eta_reported = False
        destination_reported = False

        metadata_name = _clean_ais_text(
            message.metadata.get("ShipName")
            or message.metadata.get("ship_name")
        )
        if metadata_name and not state.vessel_name:
            state.vessel_name = metadata_name

        if message.message_type in POSITION_MESSAGE_TYPES:
            latitude = _float_or_none(body.get("Latitude"))
            longitude = _float_or_none(body.get("Longitude"))
            if (
                latitude is None
                or longitude is None
                or not (-90.0 <= latitude <= 90.0)
                or not (-180.0 <= longitude <= 180.0)
                or not _inside_baltic(latitude, longitude)
            ):
                latitude = None
                longitude = None
            state.latitude = latitude
            state.longitude = longitude
            sog = _float_or_none(body.get("Sog"))
            state.sog_kn = sog if sog is not None and 0 <= sog < 102.3 else None
            cog = _float_or_none(body.get("Cog"))
            state.cog_deg = cog if cog is not None and 0 <= cog < 360 else None
            heading = _int_or_none(body.get("TrueHeading"))
            state.heading_deg = (
                heading if heading is not None and 0 <= heading < 360 else None
            )
            nav_status = _int_or_none(body.get("NavigationalStatus"))
            state.navigational_status = nav_status
            state.position_observed_at = observed_at
        elif message.message_type == "ShipStaticData":
            state.vessel_name = _clean_ais_text(body.get("Name"))
            state.imo = _optional_imo(body.get("ImoNumber"))
            state.call_sign = _clean_ais_text(body.get("CallSign"))
            state.ship_type = _int_or_none(body.get("Type"))
            destination = normalize_aisstream_destination(body.get("Destination"))
            state.destination_raw = destination.raw
            state.destination_locode = destination.locode
            state.destination_name = destination.name
            state.destination_match = destination.match
            destination_reported = True
            state.eta_utc = infer_ais_eta(body.get("Eta"), observed_at)
            state.eta_observed_at = observed_at
            eta_reported = True
            state.static_observed_at = observed_at
        else:
            report_a = body.get("ReportA")
            report_b = body.get("ReportB")
            if isinstance(report_a, dict) and report_a.get("Valid") is not False:
                if "Name" in report_a:
                    state.vessel_name = _clean_ais_text(report_a.get("Name"))
            if isinstance(report_b, dict) and report_b.get("Valid") is not False:
                if "CallSign" in report_b:
                    state.call_sign = _clean_ais_text(report_b.get("CallSign"))
                if "ShipType" in report_b:
                    state.ship_type = _int_or_none(report_b.get("ShipType"))
            state.static_observed_at = max(
                filter(
                    lambda value: value is not None,
                    (state.static_observed_at, observed_at),
                )
            )

        state.last_seen_utc = max(
            filter(
                lambda value: value is not None,
                (state.last_seen_utc, observed_at),
            )
        )
        state.last_message_type = message.message_type
        return eta_reported, destination_reported

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS aisstream_history (
                    event_id TEXT PRIMARY KEY,
                    observed_at_utc TEXT NOT NULL,
                    received_at_utc TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    ordering_key TEXT NOT NULL,
                    mmsi TEXT NOT NULL,
                    vessel_name TEXT,
                    imo TEXT,
                    call_sign TEXT,
                    ship_type INTEGER,
                    destination_raw TEXT,
                    destination_locode TEXT,
                    destination_name TEXT,
                    destination_match TEXT,
                    eta_utc TEXT,
                    eta_observed_at_utc TEXT,
                    eta_reported INTEGER NOT NULL,
                    destination_reported INTEGER NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    sog_kn REAL,
                    cog_deg REAL,
                    heading_deg INTEGER,
                    navigational_status INTEGER,
                    position_observed_at_utc TEXT,
                    static_observed_at_utc TEXT,
                    last_seen_utc TEXT,
                    last_message_type TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_aisstream_history_mmsi_time
                    ON aisstream_history(mmsi, observed_at_utc);

                CREATE INDEX IF NOT EXISTS idx_aisstream_history_eta
                    ON aisstream_history(eta_reported, observed_at_utc);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path, timeout=15.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _restore_cache(self) -> None:
        snapshot = _ensure_utc(self.clock())
        self._purge_history(snapshot)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM aisstream_history
                ORDER BY observed_at_utc ASC, rowid ASC
                """
            ).fetchall()
        with self._lock:
            for row in rows:
                state = _state_from_sqlite_row(row)
                self._states[state.mmsi] = state
                observed = _parse_datetime(row["observed_at_utc"])
                if observed is not None:
                    key = (state.mmsi, str(row["ordering_key"]))
                    previous = self._last_event_by_key.get(key)
                    if previous is None or observed > previous:
                        self._last_event_by_key[key] = observed
                    if self._last_message_at is None or observed > self._last_message_at:
                        self._last_message_at = observed
                self._seen_event_ids.add(str(row["event_id"]))
            self._prune_cache(snapshot)

    def _persist_observation(
        self,
        message: _NormalizedMessage,
        state: AISStreamVesselState,
        *,
        eta_reported: bool,
        destination_reported: bool,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO aisstream_history(
                    event_id, observed_at_utc, received_at_utc, message_type,
                    ordering_key, mmsi, vessel_name, imo, call_sign, ship_type,
                    destination_raw, destination_locode, destination_name,
                    destination_match, eta_utc, eta_observed_at_utc, eta_reported,
                    destination_reported, latitude, longitude, sog_kn, cog_deg,
                    heading_deg, navigational_status, position_observed_at_utc,
                    static_observed_at_utc, last_seen_utc, last_message_type
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    message.event_id,
                    _iso_utc(message.observed_at),
                    _iso_utc(message.received_at),
                    message.message_type,
                    message.ordering_key,
                    state.mmsi,
                    state.vessel_name,
                    state.imo,
                    state.call_sign,
                    state.ship_type,
                    state.destination_raw,
                    state.destination_locode,
                    state.destination_name,
                    state.destination_match,
                    _optional_iso(state.eta_utc),
                    _optional_iso(state.eta_observed_at),
                    int(eta_reported),
                    int(destination_reported),
                    state.latitude,
                    state.longitude,
                    state.sog_kn,
                    state.cog_deg,
                    state.heading_deg,
                    state.navigational_status,
                    _optional_iso(state.position_observed_at),
                    _optional_iso(state.static_observed_at),
                    _optional_iso(state.last_seen_utc),
                    state.last_message_type,
                ),
            )

    def _purge_history(self, snapshot: datetime) -> None:
        cutoff = _iso_utc(
            _ensure_utc(snapshot) - timedelta(hours=self.history_hours)
        )
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM aisstream_history WHERE observed_at_utc < ?",
                (cutoff,),
            )

    def _prune_cache(self, snapshot: datetime) -> None:
        cutoff = _ensure_utc(snapshot) - timedelta(hours=self.history_hours)
        expired = [
            mmsi
            for mmsi, state in self._states.items()
            if state.last_seen_utc is None or state.last_seen_utc < cutoff
        ]
        for mmsi in expired:
            self._states.pop(mmsi, None)
            for key in [
                item for item in self._last_event_by_key if item[0] == mmsi
            ]:
                self._last_event_by_key.pop(key, None)
        if expired:
            retained_ids: set[str] = set()
            with self._connect() as connection:
                for row in connection.execute(
                    "SELECT event_id FROM aisstream_history"
                ).fetchall():
                    retained_ids.add(str(row["event_id"]))
            self._seen_event_ids.intersection_update(retained_ids)

    def _inbound_rows(
        self,
        states: Sequence[AISStreamVesselState],
        *,
        snapshot: datetime,
        horizon_end: datetime,
        stale_after: timedelta,
        include_stale: bool,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for state in states:
            if state.destination_locode is None or state.eta_utc is None:
                continue
            if state.eta_utc < snapshot - timedelta(minutes=10):
                continue
            if state.eta_utc > horizon_end:
                continue
            if (
                not include_stale
                and (
                    state.position_observed_at is None
                    or snapshot - state.position_observed_at > stale_after
                    or state.static_observed_at is None
                    or snapshot - state.static_observed_at > stale_after
                )
            ):
                continue
            rows.append(
                state.public_row(snapshot=snapshot, stale_after=stale_after)
            )
        rows.sort(
            key=lambda row: (
                row.get("eta_utc") or "",
                row.get("destination_locode") or "",
                row.get("vessel_name") or "",
                row["mmsi"],
            )
        )
        return rows

    @staticmethod
    def _destination_load_rows(
        inbound: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_destination: Dict[str, List[Dict[str, Any]]] = {}
        for row in inbound:
            locode = str(row.get("destination_locode") or "")
            if not locode:
                continue
            by_destination.setdefault(locode, []).append(row)
        output: List[Dict[str, Any]] = []
        for locode, rows in by_destination.items():
            rows = sorted(
                rows,
                key=lambda row: (
                    row.get("eta_utc") or "",
                    row["mmsi"],
                ),
            )
            output.append(
                {
                    "destination_locode": locode,
                    "destination_name": rows[0].get("destination_name"),
                    "vessel_count": len(rows),
                    "next_eta_utc": rows[0].get("eta_utc"),
                    "next_vessel_mmsi": rows[0].get("mmsi"),
                    "next_vessel_name": rows[0].get("vessel_name"),
                    "mmsis": [row["mmsi"] for row in rows],
                    "source_scope": "aisstream_baltic_broadcast",
                }
            )
        output.sort(
            key=lambda row: (
                -int(row["vessel_count"]),
                row.get("next_eta_utc") or "",
                row["destination_locode"],
            )
        )
        return output

    def _eta_revision_rows(
        self,
        *,
        states: Sequence[AISStreamVesselState],
        snapshot: datetime,
        window_minutes: int,
        threshold_minutes: int,
    ) -> List[Dict[str, Any]]:
        selected = {state.mmsi for state in states}
        if not selected:
            return []
        history_cutoff = snapshot - timedelta(hours=self.history_hours)
        window_cutoff = snapshot - timedelta(minutes=window_minutes)
        placeholders = ",".join("?" for _ in selected)
        params: List[Any] = [
            _iso_utc(history_cutoff),
            *sorted(selected),
        ]
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT mmsi, vessel_name, destination_locode, destination_name,
                       eta_utc, observed_at_utc
                FROM aisstream_history
                WHERE eta_reported = 1
                  AND eta_utc IS NOT NULL
                  AND observed_at_utc >= ?
                  AND mmsi IN ({placeholders})
                ORDER BY mmsi ASC, observed_at_utc ASC, rowid ASC
                """,
                params,
            ).fetchall()
        previous_eta: Dict[str, datetime] = {}
        latest_revision: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            observed_at = _parse_datetime(row["observed_at_utc"])
            eta = _parse_datetime(row["eta_utc"])
            mmsi = str(row["mmsi"])
            if observed_at is None or eta is None:
                continue
            previous = previous_eta.get(mmsi)
            if previous is not None and eta != previous and observed_at >= window_cutoff:
                delta = int(round((eta - previous).total_seconds() / 60.0))
                if abs(delta) >= threshold_minutes:
                    latest_revision[mmsi] = {
                        "snapshot_time_utc": _iso_utc(snapshot),
                        "mmsi": mmsi,
                        "vessel_name": row["vessel_name"],
                        "destination_locode": row["destination_locode"],
                        "destination_name": row["destination_name"],
                        "previous_eta_utc": _iso_utc(previous),
                        "current_eta_utc": _iso_utc(eta),
                        "eta_revision_minutes": delta,
                        "revision_observed_at_utc": _iso_utc(observed_at),
                        "source_scope": "aisstream_baltic_broadcast",
                    }
            previous_eta[mmsi] = eta
        output = list(latest_revision.values())
        output.sort(
            key=lambda row: (
                -abs(int(row["eta_revision_minutes"])),
                row["revision_observed_at_utc"],
                row["mmsi"],
            )
        )
        return output

    @staticmethod
    def _stale_missing_rows(
        states: Sequence[AISStreamVesselState],
        *,
        snapshot: datetime,
        stale_after: timedelta,
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for state in states:
            reasons: List[str] = []
            if (
                state.position_observed_at is None
                or state.latitude is None
                or state.longitude is None
            ):
                reasons.append("missing_position")
            elif snapshot - state.position_observed_at > stale_after:
                reasons.append("stale_position")
            if state.static_observed_at is None:
                reasons.append("missing_static")
            elif snapshot - state.static_observed_at > stale_after:
                reasons.append("stale_static")
            if state.destination_raw is None:
                reasons.append("missing_destination")
            elif state.destination_locode is None:
                reasons.append("unrecognized_destination")
            if state.eta_utc is None:
                reasons.append("missing_eta")
            if not reasons:
                continue
            row = state.public_row(
                snapshot=snapshot,
                stale_after=stale_after,
            )
            row["validation_status"] = "stale_or_missing"
            row["validation_reasons"] = reasons
            output.append(row)
        output.sort(
            key=lambda row: (
                -len(row["validation_reasons"]),
                row.get("last_seen_utc") or "",
                row["mmsi"],
            )
        )
        return output

    def _result(
        self,
        *,
        status: str,
        operation: str,
        answer: str,
        rows: Sequence[Dict[str, Any]],
        snapshot: datetime,
        health: AISStreamSourceHealth,
        horizon_end: Optional[datetime] = None,
        summary: Optional[Dict[str, Any]] = None,
        sections: Optional[Dict[str, pd.DataFrame]] = None,
        failure_reason: Optional[str] = None,
    ) -> AISStreamQueryResult:
        data_updated = max(
            (
                state.last_seen_utc
                for state in self._states.values()
                if state.last_seen_utc is not None
            ),
            default=None,
        )
        coverage = [
            f"Retrieval snapshot: {_iso_utc(snapshot)}",
            (
                "Latest accepted AISStream message: "
                f"{_iso_utc(data_updated)}"
                if data_updated is not None
                else "Latest accepted AISStream message: unavailable"
            ),
            (
                "Baltic subscription bounding box: "
                f"{BALTIC_BOUNDING_BOX[0][0]} to {BALTIC_BOUNDING_BOX[0][1]}"
            ),
            f"SQLite observation retention: {self.history_hours} hours",
            f"Source health: {health.status}",
        ]
        caveats = [
            "AISStream data is vessel-broadcast and receiver-dependent; it is not an official port schedule.",
            "AIS ETA and destination are self-reported and are not an Eagle Eye prediction.",
            "AISStream is a beta source without a published uptime SLA.",
        ]
        return AISStreamQueryResult(
            status=status,
            operation=operation,
            answer=answer,
            table=_frame(rows),
            summary=dict(summary or {}),
            sections=dict(sections or {}),
            coverage_notes=coverage,
            caveats=caveats,
            snapshot_at=snapshot,
            data_updated_at=data_updated,
            horizon_end=horizon_end,
            failure_reason=failure_reason,
            health=health,
        )

    @staticmethod
    def _status_for_rows(
        rows: Sequence[Dict[str, Any]],
        health: AISStreamSourceHealth,
    ) -> str:
        if rows:
            return "ok"
        if health.cached_vessels == 0 and health.status in {
            "connecting",
            "warming",
            "stale",
            "unavailable",
        }:
            return "no_current_data"
        return "no_data"


def _inside_baltic(latitude: float, longitude: float) -> bool:
    for first, second in BALTIC_BOUNDING_BOX:
        lat_min, lat_max = sorted((first[0], second[0]))
        lon_min, lon_max = sorted((first[1], second[1]))
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            return True
    return False


def _safe_error_code(exc: Exception) -> str:
    if isinstance(exc, AISStreamProtocolError):
        token = str(exc).strip()
        return token if token in {"provider_error", "stream_closed"} else "protocol_error"
    if isinstance(exc, AISStreamConfigurationError):
        return "configuration_error"
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        type(exc).__name__.lower(),
    ).strip("_") or "connection_error"


def _state_from_sqlite_row(row: sqlite3.Row) -> AISStreamVesselState:
    return AISStreamVesselState(
        mmsi=str(row["mmsi"]),
        vessel_name=row["vessel_name"],
        imo=row["imo"],
        call_sign=row["call_sign"],
        ship_type=_int_or_none(row["ship_type"]),
        destination_raw=row["destination_raw"],
        destination_locode=row["destination_locode"],
        destination_name=row["destination_name"],
        destination_match=row["destination_match"],
        eta_utc=_parse_datetime(row["eta_utc"]),
        eta_observed_at=_parse_datetime(row["eta_observed_at_utc"]),
        latitude=_float_or_none(row["latitude"]),
        longitude=_float_or_none(row["longitude"]),
        sog_kn=_float_or_none(row["sog_kn"]),
        cog_deg=_float_or_none(row["cog_deg"]),
        heading_deg=_int_or_none(row["heading_deg"]),
        navigational_status=_int_or_none(row["navigational_status"]),
        position_observed_at=_parse_datetime(row["position_observed_at_utc"]),
        static_observed_at=_parse_datetime(row["static_observed_at_utc"]),
        last_seen_utc=_parse_datetime(row["last_seen_utc"]),
        last_message_type=row["last_message_type"],
    )


__all__ = [
    "AISSTREAM_MESSAGE_TYPES",
    "AISSTREAM_PROVIDER",
    "AISSTREAM_QUERY_OPERATIONS",
    "AISSTREAM_WEBSOCKET_URL",
    "AISStreamCollector",
    "AISStreamConfigurationError",
    "AISStreamProtocolError",
    "AISStreamQueryResult",
    "AISStreamSourceHealth",
    "AISStreamVesselState",
    "BALTIC_BOUNDING_BOX",
    "infer_ais_eta",
    "normalize_aisstream_destination",
]
