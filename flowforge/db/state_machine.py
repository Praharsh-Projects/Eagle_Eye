from __future__ import annotations

from db.models import JobStatus


ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.RUNNING, JobStatus.CANCELED},
    JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.RETRYING, JobStatus.CANCELED},
    JobStatus.RETRYING: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELED: set(),
}


TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def ensure_transition(current: JobStatus, target: JobStatus) -> None:
    if not can_transition(current, target):
        raise ValueError(f"invalid transition: {current.value} -> {target.value}")
