from db.models import JobStatus
from db.state_machine import can_transition, ensure_transition


def test_valid_transitions() -> None:
    assert can_transition(JobStatus.PENDING, JobStatus.RUNNING)
    assert can_transition(JobStatus.RUNNING, JobStatus.RETRYING)
    assert can_transition(JobStatus.RETRYING, JobStatus.RUNNING)
    assert can_transition(JobStatus.RUNNING, JobStatus.COMPLETED)


def test_invalid_transition_raises() -> None:
    try:
        ensure_transition(JobStatus.COMPLETED, JobStatus.RUNNING)
    except ValueError as exc:
        assert "invalid transition" in str(exc)
    else:
        raise AssertionError("expected ValueError")
