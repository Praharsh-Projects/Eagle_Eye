"""Small output-boundary redactor for credentials and authorization values."""

from __future__ import annotations

import re
from typing import Any


_SENSITIVE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"(?i)\b(?:OPENAI_API_KEY|API_KEY|TOKEN|SECRET)\s*[=:]\s*[^\s,;]+"),
    re.compile(r"(?i)(?:\?|&)(?:api_key|key|token|secret)=[^&#\s]+"),
)


def redact_sensitive_text(value: Any) -> str:
    text = str(value or "")
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def redact_sensitive_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): redact_sensitive_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_sensitive_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_value(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value
