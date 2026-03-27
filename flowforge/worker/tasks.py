from __future__ import annotations

import socket

from celery.exceptions import MaxRetriesExceededError

from core.config import settings
from core.database import SessionLocal, init_db
from db.models import JobStatus
from db.service import (
    bump_attempt,
    get_job,
    mark_completed,
    mark_failed,
    mark_retrying,
    mark_running,
    upsert_worker_heartbeat,
)
from worker.celery_app import celery_app
from worker.executor import TransientJobError, compute_backoff, execute_payload
from worker.webhook import dispatch_webhook


@celery_app.task(bind=True, name="flowforge.execute_job", max_retries=settings.max_job_retries)
def execute_job(self, job_id: str) -> dict:
    init_db()
    worker_id = self.request.hostname or socket.gethostname()
    hostname = socket.gethostname()

    with SessionLocal() as session:
        job = get_job(session, job_id)
        if job is None:
            return {"job_id": job_id, "status": "not_found"}

        upsert_worker_heartbeat(session, worker_id=worker_id, hostname=hostname, current_job_id=job_id)

        if job.status in {JobStatus.CANCELED, JobStatus.COMPLETED, JobStatus.FAILED}:
            return {"job_id": job_id, "status": job.status.value}

        if job.status in {JobStatus.PENDING, JobStatus.RETRYING}:
            mark_running(session, job)

        try:
            bump_attempt(session, job)
            result = execute_payload(job.payload)
            mark_completed(session, job, result=result)

            if job.webhook_url:
                dispatch_webhook(
                    webhook_url=job.webhook_url,
                    payload={"job_id": job.id, "status": job.status.value, "result": result},
                )

            return {"job_id": job.id, "status": job.status.value, "result": result}

        except TransientJobError as exc:
            refreshed = get_job(session, job_id)
            if refreshed is None:
                return {"job_id": job_id, "status": "not_found"}

            if refreshed.attempts <= settings.max_job_retries:
                mark_retrying(session, refreshed, error_message=str(exc))
                delay_seconds = compute_backoff(refreshed.attempts, settings.base_backoff_seconds)
                try:
                    raise self.retry(exc=exc, countdown=delay_seconds)
                except MaxRetriesExceededError:
                    latest = get_job(session, job_id)
                    if latest is not None and latest.status != JobStatus.FAILED:
                        mark_failed(session, latest, error_message=str(exc))
                    return {"job_id": job_id, "status": JobStatus.FAILED.value, "error": str(exc)}

            mark_failed(session, refreshed, error_message=str(exc))
            return {"job_id": refreshed.id, "status": refreshed.status.value, "error": str(exc)}

        except Exception as exc:  # noqa: BLE001
            refreshed = get_job(session, job_id)
            if refreshed is not None and refreshed.status != JobStatus.FAILED:
                mark_failed(session, refreshed, error_message=str(exc))
            return {"job_id": job_id, "status": JobStatus.FAILED.value, "error": str(exc)}

        finally:
            with SessionLocal() as heartbeat_session:
                upsert_worker_heartbeat(
                    heartbeat_session,
                    worker_id=worker_id,
                    hostname=hostname,
                    current_job_id=None,
                )
