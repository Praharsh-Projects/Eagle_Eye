from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import grpc

from core.config import settings
from core.database import SessionLocal, init_db
from db.models import JobStatus
from db.service import create_job, get_job
from worker.celery_app import celery_app

from api.grpc.generated import flowforge_pb2, flowforge_pb2_grpc


def _to_epoch_seconds(dt: datetime | None) -> int:
    if dt is None:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _build_status_response(job_id: str, status: str, attempts: int = 0, error: str = "", result: str = "", updated: int = 0) -> flowforge_pb2.JobStatusResponse:
    return flowforge_pb2.JobStatusResponse(
        job_id=job_id,
        status=status,
        attempts=attempts,
        error_message=error,
        result_json=result,
        updated_epoch_seconds=updated,
    )


class FlowForgeServicer(flowforge_pb2_grpc.FlowForgeServicer):
    async def SubmitJob(self, request: flowforge_pb2.SubmitJobRequest, context: grpc.aio.ServicerContext) -> flowforge_pb2.JobResponse:  # noqa: N802
        init_db()
        payload_raw = request.payload_json or "{}"
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must be valid JSON")

        with SessionLocal() as session:
            job = create_job(
                session,
                payload=payload,
                priority=max(request.priority, 0),
                webhook_url=request.webhook_url or None,
            )

        celery_app.send_task("flowforge.execute_job", args=[job.id])
        return flowforge_pb2.JobResponse(job_id=job.id, status=job.status.value)

    async def GetJobStatus(self, request: flowforge_pb2.GetJobStatusRequest, context: grpc.aio.ServicerContext) -> flowforge_pb2.JobStatusResponse:  # noqa: N802
        with SessionLocal() as session:
            job = get_job(session, request.job_id)
            if job is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "job not found")

            result_json = ""
            if job.result is not None:
                result_json = json.dumps(job.result)

            return _build_status_response(
                job_id=job.id,
                status=job.status.value,
                attempts=job.attempts,
                error=job.error_message or "",
                result=result_json,
                updated=_to_epoch_seconds(job.updated_at),
            )

    async def StreamJobUpdates(self, request: flowforge_pb2.GetJobStatusRequest, context: grpc.aio.ServicerContext):  # noqa: N802
        seen_status: str | None = None
        while True:
            if not context.is_active():
                return

            with SessionLocal() as session:
                job = get_job(session, request.job_id)
                if job is None:
                    await context.abort(grpc.StatusCode.NOT_FOUND, "job not found")

                result_json = json.dumps(job.result) if job.result is not None else ""
                if seen_status != job.status.value:
                    seen_status = job.status.value
                    yield _build_status_response(
                        job_id=job.id,
                        status=job.status.value,
                        attempts=job.attempts,
                        error=job.error_message or "",
                        result=result_json,
                        updated=_to_epoch_seconds(job.updated_at),
                    )

                if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}:
                    return

            await asyncio.sleep(1.0)


async def serve() -> None:
    init_db()
    server = grpc.aio.server()
    flowforge_pb2_grpc.add_FlowForgeServicer_to_server(FlowForgeServicer(), server)
    address = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(address)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
