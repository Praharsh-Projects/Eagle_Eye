import pytest

from worker.executor import TransientJobError, compute_backoff, execute_payload


def test_compute_backoff_exponential() -> None:
    assert compute_backoff(1, 2) == 2
    assert compute_backoff(2, 2) == 4
    assert compute_backoff(3, 2) == 8


def test_compute_backoff_rejects_zero_attempt() -> None:
    with pytest.raises(ValueError):
        compute_backoff(0, 2)


def test_execute_payload_sum_numbers() -> None:
    result = execute_payload({"operation": "sum_numbers", "numbers": [1, 2, 3]})
    assert result["result"] == 6


def test_execute_payload_transient_failure() -> None:
    with pytest.raises(TransientJobError):
        execute_payload({"force_transient_failure": True})
