from __future__ import annotations

import sqlite3

from src.query.context import ConversationStore
from src.utils.redaction import redact_sensitive_text, redact_sensitive_value


SECRET = "sk-" + "exampleCredentialValue1234567890"


def test_output_redactor_covers_keys_headers_and_nested_metadata() -> None:
    payload = {
        "message": f"OPENAI_API_KEY={SECRET}",
        "nested": [f"Bearer {SECRET}", f"https://example.test/source?token={SECRET}"],
    }
    redacted = redact_sensitive_value(payload)
    assert SECRET not in str(redacted)
    assert "[REDACTED]" in str(redacted)
    assert SECRET not in redact_sensitive_text(f"failure: {SECRET}")


def test_feedback_store_never_persists_supplied_credentials(tmp_path) -> None:
    path = tmp_path / "feedback.sqlite3"
    store = ConversationStore(path)
    store.save_feedback(trace_id="trace_1", prompt=f"wrong {SECRET}", note=f"token={SECRET}")
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT prompt, note FROM feedback").fetchone()
    assert row is not None
    assert SECRET not in " ".join(str(value) for value in row)
