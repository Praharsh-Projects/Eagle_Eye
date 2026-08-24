"""Durable, structured conversation state for contextual query resolution."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AnswerEnvelope, QueryPlan
from src.utils.redaction import redact_sensitive_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ConversationContext:
    conversation_id: str
    previous_plan: Optional[QueryPlan]
    previous_envelope: Optional[AnswerEnvelope]
    recent_turns: List[Dict[str, Any]]


class ConversationStore:
    """Small SQLite repository; plans and envelopes are stored as validated JSON."""

    def __init__(self, path: str | Path, max_history_turns: int = 8) -> None:
        self.path = Path(path)
        self.max_history_turns = max(2, int(max_history_turns))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, turn_id),
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_conversation_created
                    ON turns(conversation_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    note TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_trace
                    ON feedback(trace_id, created_at DESC);
                """
            )

    def save_turn(self, envelope: AnswerEnvelope) -> None:
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations(conversation_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id)
                DO UPDATE SET updated_at = excluded.updated_at
                """,
                (envelope.conversation_id, now, now),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO turns(
                    conversation_id, turn_id, created_at, question, plan_json, envelope_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.conversation_id,
                    envelope.turn_id,
                    now,
                    envelope.question,
                    envelope.plan.model_dump_json(),
                    envelope.model_dump_json(),
                ),
            )
            connection.execute(
                """
                DELETE FROM turns
                WHERE conversation_id = ?
                  AND rowid NOT IN (
                    SELECT rowid FROM turns
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                  )
                """,
                (envelope.conversation_id, envelope.conversation_id, self.max_history_turns),
            )

    def get_context(self, conversation_id: str) -> ConversationContext:
        clean_id = str(conversation_id or "").strip()
        if not clean_id:
            return ConversationContext("", None, None, [])
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT turn_id, question, plan_json, envelope_json, created_at
                FROM turns
                WHERE conversation_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (clean_id, self.max_history_turns),
            ).fetchall()

        previous_plan: Optional[QueryPlan] = None
        previous_envelope: Optional[AnswerEnvelope] = None
        recent: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            try:
                plan = QueryPlan.model_validate_json(row["plan_json"])
                envelope = AnswerEnvelope.model_validate_json(row["envelope_json"])
            except Exception:
                continue
            if index == 0:
                previous_plan = plan
                previous_envelope = envelope
            recent.append(
                {
                    "turn_id": row["turn_id"],
                    "question": row["question"],
                    "answer": envelope.answer,
                    "mode": envelope.mode.value,
                    "state": envelope.state.value,
                    "plan": plan.model_dump(mode="json"),
                    "created_at": row["created_at"],
                }
            )
        recent.reverse()
        return ConversationContext(clean_id, previous_plan, previous_envelope, recent)

    def get_envelope(self, conversation_id: str, turn_id: str) -> Optional[AnswerEnvelope]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT envelope_json FROM turns
                WHERE conversation_id = ? AND turn_id = ?
                """,
                (str(conversation_id).strip(), str(turn_id).strip()),
            ).fetchone()
        if row is None:
            return None
        try:
            return AnswerEnvelope.model_validate_json(row["envelope_json"])
        except Exception:
            return None

    def count_conversations(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM conversations").fetchone()
        return int(row["count"] if row else 0)

    def save_feedback(self, *, trace_id: str, prompt: str, note: Optional[str] = None) -> str:
        """Persist only user-visible feedback fields; traces and secrets are not copied."""
        import uuid

        feedback_id = f"feedback_{uuid.uuid4().hex[:20]}"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback(feedback_id, created_at, trace_id, prompt, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    _utc_now(),
                    redact_sensitive_text(trace_id).strip(),
                    redact_sensitive_text(prompt).strip(),
                    redact_sensitive_text(note).strip() or None,
                ),
            )
        return feedback_id

    def export_debug_snapshot(self, conversation_id: str) -> Dict[str, Any]:
        """Return structured state without exposing raw SQLite internals."""
        context = self.get_context(conversation_id)
        return {
            "conversation_id": context.conversation_id,
            "turns": json.loads(json.dumps(context.recent_turns, default=str)),
        }
