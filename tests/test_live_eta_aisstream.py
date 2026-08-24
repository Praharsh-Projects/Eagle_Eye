from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd
import pytest

from src.live_eta import (
    AISSTREAM_MESSAGE_TYPES,
    AISSTREAM_WEBSOCKET_URL,
    AISStreamCollector,
    AISStreamConfigurationError,
    AISStreamProtocolError,
    BALTIC_BOUNDING_BOX,
    infer_ais_eta,
    normalize_aisstream_destination,
)


FIXTURES = Path(__file__).parent / "fixtures" / "aisstream"
FIXED_NOW = datetime(2026, 7, 27, 12, 45, tzinfo=timezone.utc)


def _messages() -> List[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (FIXTURES / "stream_messages.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def _collector(tmp_path: Path, **kwargs: Any) -> AISStreamCollector:
    return AISStreamCollector(
        tmp_path / "aisstream.sqlite3",
        clock=lambda: FIXED_NOW,
        **kwargs,
    )


def _replay(collector: AISStreamCollector) -> None:
    for message in _messages():
        assert collector.ingest_message(message) is True


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class _FixtureSocket:
    def __init__(self, frames: List[Any]) -> None:
        self.frames = list(frames)
        self.sent: List[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def __aiter__(self) -> "_FixtureSocket":
        return self

    async def __anext__(self) -> Any:
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)

    async def close(self) -> None:
        self.closed = True


class _SocketContext:
    def __init__(self, socket: _FixtureSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> _FixtureSocket:
        return self.socket

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        return None


class _FailingContext:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aenter__(self) -> _FixtureSocket:
        raise self.error

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        return None


class _SequenceTransport:
    def __init__(self, contexts: List[Any]) -> None:
        self.contexts = list(contexts)
        self.urls: List[str] = []

    def __call__(self, url: str) -> Any:
        self.urls.append(url)
        return self.contexts.pop(0)


def test_normalizes_latest_state_nulls_and_eta_year_rollover(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path)
    _replay(collector)

    state = collector.latest_state("265111000")
    assert state is not None
    assert state.mmsi == "265111000"
    assert state.imo == "9543756"
    assert state.destination_raw == "STOCKHOLM"
    assert state.destination_locode == "SESTO"
    assert state.destination_name == "Stockholm"
    assert state.eta_utc == datetime(2026, 7, 27, 19, 0, tzinfo=timezone.utc)
    assert state.sog_kn == 8.5

    unresolved = collector.latest_state("265444000")
    assert unresolved is not None
    assert unresolved.imo is None
    assert unresolved.destination_raw == "UNDERWAY"
    assert unresolved.destination_locode is None
    assert unresolved.eta_utc is None
    assert unresolved.latitude is None
    assert unresolved.longitude is None
    assert unresolved.sog_kn is None

    exact = normalize_aisstream_destination("SE STO")
    assert (exact.locode, exact.match) == ("SESTO", "exact_curated_unlocode")
    gothenburg = normalize_aisstream_destination("Göteborg")
    assert (gothenburg.locode, gothenburg.name) == ("SEGOT", "Gothenburg")
    assert normalize_aisstream_destination("Stockholm via Tallinn").locode is None

    rollover_message = json.loads(
        (FIXTURES / "year_rollover.json").read_text(encoding="utf-8")
    )
    rollover = AISStreamCollector(
        tmp_path / "rollover.sqlite3",
        clock=lambda: datetime(2026, 12, 31, 23, 56, tzinfo=timezone.utc),
    )
    assert rollover.ingest_message(rollover_message) is True
    rollover_state = rollover.latest_state("265555000")
    assert rollover_state is not None
    assert rollover_state.eta_utc == datetime(
        2027, 1, 1, 1, 0, tzinfo=timezone.utc
    )
    assert infer_ais_eta(
        {"Month": 0, "Day": 0, "Hour": 24, "Minute": 60},
        FIXED_NOW,
    ) is None


def test_deduplicates_rejects_out_of_order_and_retains_only_24_hours(
    tmp_path: Path,
) -> None:
    clock = _MutableClock(FIXED_NOW)
    database = tmp_path / "history.sqlite3"
    collector = AISStreamCollector(database, clock=clock)
    messages = _messages()
    _replay(collector)

    assert collector.ingest_message(messages[0]) is False

    older_position = copy.deepcopy(messages[0])
    older_position["MetaData"]["time_utc"] = (
        "2026-07-27 12:37:00.000000 +0000 UTC"
    )
    older_position["Message"]["PositionReport"]["Sog"] = 12.0
    assert collector.ingest_message(older_position) is False
    state = collector.latest_state("265111000")
    assert state is not None
    assert state.sog_kn == 8.5
    health = collector.source_health()
    assert health.duplicate_messages == 1
    assert health.out_of_order_messages == 1

    restored = AISStreamCollector(database, clock=clock)
    restored_state = restored.latest_state("265111000")
    assert restored_state is not None
    assert restored_state.eta_utc == datetime(
        2026, 7, 27, 19, 0, tzinfo=timezone.utc
    )
    assert restored.ingest_message(messages[0]) is False
    assert restored.source_health().duplicate_messages == 1

    clock.value = datetime(2026, 7, 28, 12, 46, tzinfo=timezone.utc)
    current = copy.deepcopy(messages[0])
    current["MetaData"].update(
        {
            "MMSI": 265999000,
            "ShipName": "CURRENT SAMPLE",
            "time_utc": "2026-07-28 12:46:00.000000 +0000 UTC",
        }
    )
    current["Message"]["PositionReport"]["UserID"] = 265999000
    assert restored.ingest_message(current) is True
    all_states = restored.latest_state()
    assert isinstance(all_states, dict)
    assert set(all_states) == {"265999000"}
    with sqlite3.connect(database) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM aisstream_history"
        ).fetchone()[0]
    assert count == 1


def test_all_operational_queries_are_deterministic_and_capped_at_48_hours(
    tmp_path: Path,
) -> None:
    collector = _collector(tmp_path)
    _replay(collector)

    inbound = collector.query(
        "inbound_watchlist",
        ports=["Stockholm", "Gothenburg"],
        horizon_hours=999,
    )
    assert inbound.status == "ok"
    assert inbound.horizon_end == datetime(
        2026, 7, 29, 12, 45, tzinfo=timezone.utc
    )
    assert inbound.summary == {
        "matched_vessels": 2,
        "displayed_vessels": 2,
        "horizon_hours": 48,
    }
    assert inbound.table is not None
    assert inbound.table["mmsi"].tolist() == ["265222000", "265111000"]
    assert inbound.chart is inbound.table

    status = collector.query("vessel_status", imo="9543756")
    assert status.table is not None
    assert status.table["mmsi"].tolist() == ["265111000"]

    low_speed = collector.query(
        "low_speed",
        ports=["SESTO", "SEGOT"],
        speed_threshold_kn=2,
    )
    assert low_speed.table is not None
    assert low_speed.table["mmsi"].tolist() == ["265222000"]
    assert low_speed.table.iloc[0]["sog_kn"] == 1.2

    destination_load = collector.query(
        "destination_load",
        ports=["SESTO", "SEGOT"],
    )
    assert destination_load.table is not None
    assert set(destination_load.table["destination_locode"]) == {
        "SEGOT",
        "SESTO",
    }
    assert destination_load.summary["matched_vessels"] == 2

    revisions = collector.query(
        "eta_revisions",
        eta_change_threshold_minutes=30,
        change_window_minutes=60,
    )
    assert revisions.table is not None
    assert revisions.table["mmsi"].tolist() == ["265111000"]
    assert revisions.table.iloc[0]["eta_revision_minutes"] == 60

    stale_missing = collector.query("stale_missing")
    assert stale_missing.table is not None
    assert set(stale_missing.table["mmsi"]) == {"265333000", "265444000"}
    unresolved_row = stale_missing.table[
        stale_missing.table["mmsi"] == "265444000"
    ].iloc[0]
    assert "unrecognized_destination" in unresolved_row["validation_reasons"]
    assert pd.isna(unresolved_row["latitude"])
    assert pd.isna(unresolved_row["sog_kn"])

    handover = collector.query("shift_handover")
    assert handover.status == "ok"
    assert handover.summary["inbound_vessels"] == 2
    assert handover.summary["low_speed_vessels"] == 1
    assert handover.summary["eta_revision_vessels"] == 1
    assert handover.summary["stale_or_missing_vessels"] == 2
    assert set(handover.sections) == {
        "inbound_watchlist",
        "low_speed",
        "destination_load",
        "eta_revisions",
        "stale_missing",
    }
    assert handover.sections["stale_missing"].shape[0] == 2

    repeated = collector.query("shift_handover")
    assert repeated.answer == handover.answer
    assert repeated.summary == handover.summary
    assert repeated.table is not None
    assert handover.table is not None
    assert repeated.table.to_dict("records") == handover.table.to_dict("records")

    outside_coverage = collector.query(
        "inbound_watchlist",
        ports=["Port on Mars"],
    )
    assert outside_coverage.status == "no_current_data"
    assert outside_coverage.failure_reason == "destination_coverage_unavailable"
    assert outside_coverage.table is None


def test_single_websocket_subscription_reconnect_and_secret_hygiene(
    tmp_path: Path,
) -> None:
    api_key = "fixture-secret-key"
    socket = _FixtureSocket([_messages()[0]])
    transport = _SequenceTransport(
        [
            _FailingContext(OSError(f"connection failed for {api_key}")),
            _SocketContext(socket),
        ]
    )
    sleeps: List[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    collector = _collector(
        tmp_path,
        api_key=api_key,
        transport_factory=transport,
        sleep_fn=record_sleep,
        reconnect_base_seconds=1,
        reconnect_max_seconds=8,
    )
    asyncio.run(collector.run(max_connections=2))

    assert transport.urls == [
        AISSTREAM_WEBSOCKET_URL,
        AISSTREAM_WEBSOCKET_URL,
    ]
    assert sleeps == [1.0]
    assert len(socket.sent) == 1
    subscription = json.loads(socket.sent[0])
    assert subscription == {
        "APIKey": api_key,
        "BoundingBoxes": [
            [[corner[0], corner[1]] for corner in box]
            for box in BALTIC_BOUNDING_BOX
        ],
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }
    assert tuple(subscription["FilterMessageTypes"]) == AISSTREAM_MESSAGE_TYPES

    health = collector.source_health()
    assert health.status == "warming"
    assert health.connected is False
    assert health.reconnect_attempts == 1
    assert health.accepted_messages == 1
    assert api_key not in repr(health)
    capabilities = collector.capabilities()
    assert capabilities["provider"] == "aisstream"
    assert capabilities["source_health"] == "warming"
    assert capabilities["maximum_horizon_hours"] == 48
    assert capabilities["official_schedule"] is False
    assert capabilities["country_scope"] == [
        "DE",
        "DK",
        "EE",
        "FI",
        "LT",
        "LV",
        "PL",
        "SE",
    ]
    assert api_key not in repr(capabilities)
    assert api_key not in collector.query("vessel_status").answer


def test_configuration_provider_errors_and_unsupported_frames_are_safe(
    tmp_path: Path,
) -> None:
    missing_key = _collector(tmp_path)
    with pytest.raises(
        AISStreamConfigurationError,
        match="AISSTREAM_API_KEY is required",
    ):
        asyncio.run(missing_key.run(max_connections=1))

    api_key = "do-not-expose-this"
    error_socket = _FixtureSocket(
        [json.dumps({"error": f"invalid API key: {api_key}"})]
    )
    collector = AISStreamCollector(
        tmp_path / "provider-error.sqlite3",
        api_key=api_key,
        clock=lambda: FIXED_NOW,
        transport_factory=_SequenceTransport([_SocketContext(error_socket)]),
    )
    asyncio.run(collector.run(max_connections=1))
    health = collector.source_health()
    assert health.last_error_code == "provider_error"
    assert health.invalid_messages == 1
    assert api_key not in repr(health)

    unsupported = {
        "MessageType": "StandardClassBPositionReport",
        "MetaData": {
            "MMSI": 265111000,
            "time_utc": "2026-07-27 12:44:00.000000 +0000 UTC",
        },
        "Message": {
            "StandardClassBPositionReport": {
                "UserID": 265111000,
                "Valid": True,
            }
        },
    }
    assert collector.ingest_message(unsupported) is False
    assert collector.source_health().invalid_messages == 2

    with pytest.raises(AISStreamProtocolError, match="provider_error") as error:
        collector.ingest_message({"error": api_key})
    assert api_key not in str(error.value)
