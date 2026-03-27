from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from db.models import JobStatus


class JobCreate(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=10)
    webhook_url: str | None = Field(default=None, max_length=512)


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: JobStatus
    priority: int
    attempts: int
    error_message: str | None
    result: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class JobListResponse(BaseModel):
    total: int
    items: list[JobRead]


class JobSubmitResponse(BaseModel):
    id: str
    status: JobStatus
