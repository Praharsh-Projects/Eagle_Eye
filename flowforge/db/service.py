from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from db.models import Job, JobStatus, WorkerHeartbeat
from db.state_machine import TERMINAL_STATUSES, ensure_transition


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_job(session: Session, payload: dict, priority: int = 0, webhook_url: str | None = None) -> Job:
    job = Job(
        id=str(uuid.uuid4()),
        payload=payload,
        status=JobStatus.PENDING,
        priority=priority,
        webhook_url=webhook_url,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_job(session: Session, job_id: str) -> Job | None:
    return session.get(Job, job_id)


def list_jobs(
    session: Session,
    status: JobStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[int, list[Job]]:
    base_query: Select[tuple[Job]] = select(Job)
    count_query = select(func.count()).select_from(Job)

    if status is not None:
        base_query = base_query.where(Job.status == status)
        count_query = count_query.where(Job.status == status)

    total = session.execute(count_query).scalar_one()
    items = (
        session.execute(base_query.order_by(Job.created_at.desc()).offset(offset).limit(limit))
        .scalars()
        .all()
    )
    return total, items


def cancel_job(session: Session, job: Job) -> Job:
    if job.status in TERMINAL_STATUSES:
        return job

    ensure_transition(job.status, JobStatus.CANCELED)
    job.status = JobStatus.CANCELED
    job.updated_at = _now()
    job.completed_at = _now()
    session.commit()
    session.refresh(job)
    return job


def mark_running(session: Session, job: Job) -> Job:
    if job.status == JobStatus.CANCELED:
        return job

    ensure_transition(job.status, JobStatus.RUNNING)
    now = _now()
    job.status = JobStatus.RUNNING
    job.started_at = now
    job.updated_at = now
    session.commit()
    session.refresh(job)
    return job


def mark_retrying(session: Session, job: Job, error_message: str) -> Job:
    ensure_transition(job.status, JobStatus.RETRYING)
    job.status = JobStatus.RETRYING
    job.error_message = error_message
    job.updated_at = _now()
    session.commit()
    session.refresh(job)
    return job


def mark_completed(session: Session, job: Job, result: dict) -> Job:
    ensure_transition(job.status, JobStatus.COMPLETED)
    now = _now()
    job.status = JobStatus.COMPLETED
    job.result = result
    job.error_message = None
    job.updated_at = now
    job.completed_at = now
    session.commit()
    session.refresh(job)
    return job


def mark_failed(session: Session, job: Job, error_message: str) -> Job:
    ensure_transition(job.status, JobStatus.FAILED)
    now = _now()
    job.status = JobStatus.FAILED
    job.error_message = error_message
    job.updated_at = now
    job.completed_at = now
    session.commit()
    session.refresh(job)
    return job


def bump_attempt(session: Session, job: Job, error_message: str | None = None) -> Job:
    job.attempts += 1
    job.error_message = error_message
    job.updated_at = _now()
    session.commit()
    session.refresh(job)
    return job


def upsert_worker_heartbeat(
    session: Session,
    worker_id: str,
    hostname: str,
    current_job_id: str | None,
) -> WorkerHeartbeat:
    heartbeat = session.get(WorkerHeartbeat, worker_id)
    if heartbeat is None:
        heartbeat = WorkerHeartbeat(id=worker_id, hostname=hostname)
        session.add(heartbeat)

    heartbeat.hostname = hostname
    heartbeat.current_job_id = current_job_id
    heartbeat.last_heartbeat = _now()
    session.commit()
    session.refresh(heartbeat)
    return heartbeat
