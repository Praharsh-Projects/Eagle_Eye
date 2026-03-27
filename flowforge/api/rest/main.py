from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.database import get_session, init_db
from db.models import JobStatus
from db.schemas import JobCreate, JobListResponse, JobRead, JobSubmitResponse
from db.service import cancel_job, create_job, get_job, list_jobs
from worker.celery_app import celery_app


app = FastAPI(title="FlowForge REST API", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_job(payload: JobCreate, session: Session = Depends(get_session)) -> JobSubmitResponse:
    job = create_job(session, payload=payload.payload, priority=payload.priority, webhook_url=payload.webhook_url)
    celery_app.send_task("flowforge.execute_job", args=[job.id])
    return JobSubmitResponse(id=job.id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobRead)
def get_job_by_id(job_id: str, session: Session = Depends(get_session)) -> JobRead:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return JobRead.model_validate(job)


@app.get("/jobs", response_model=JobListResponse)
def list_jobs_endpoint(
    status_filter: JobStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> JobListResponse:
    total, items = list_jobs(session, status=status_filter, limit=limit, offset=offset)
    return JobListResponse(total=total, items=[JobRead.model_validate(item) for item in items])


@app.delete("/jobs/{job_id}", response_model=JobRead)
def cancel_job_endpoint(job_id: str, session: Session = Depends(get_session)) -> JobRead:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    updated = cancel_job(session, job)
    return JobRead.model_validate(updated)


@app.get("/metrics")
def metrics(session: Session = Depends(get_session)) -> dict[str, int]:
    _, pending = list_jobs(session, status=JobStatus.PENDING, limit=1000)
    _, running = list_jobs(session, status=JobStatus.RUNNING, limit=1000)
    _, retrying = list_jobs(session, status=JobStatus.RETRYING, limit=1000)
    _, completed = list_jobs(session, status=JobStatus.COMPLETED, limit=1000)
    _, failed = list_jobs(session, status=JobStatus.FAILED, limit=1000)

    return {
        "pending": len(pending),
        "running": len(running),
        "retrying": len(retrying),
        "completed": len(completed),
        "failed": len(failed),
    }
