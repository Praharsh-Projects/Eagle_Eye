from __future__ import annotations

from typing import Any


class TransientJobError(Exception):
    """Error type that allows retry with backoff."""


def compute_backoff(attempt: int, base_seconds: int) -> int:
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return base_seconds * (2 ** (attempt - 1))


def execute_payload(payload: dict[str, Any]) -> dict[str, Any]:
    operation = payload.get("operation", "echo")

    if payload.get("force_transient_failure"):
        raise TransientJobError("simulated transient failure")

    if payload.get("force_failure"):
        raise RuntimeError("simulated terminal failure")

    if operation == "sum_numbers":
        numbers = payload.get("numbers", [])
        if not isinstance(numbers, list) or not all(isinstance(x, (int, float)) for x in numbers):
            raise ValueError("numbers must be a list of ints/floats")
        return {"operation": operation, "result": sum(numbers)}

    if operation == "uppercase":
        text = payload.get("text", "")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return {"operation": operation, "result": text.upper()}

    return {"operation": "echo", "result": payload}
