from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "FlowForge"
    environment: str = os.getenv("FLOWFORGE_ENV", "development")
    database_url: str = os.getenv("FLOWFORGE_DATABASE_URL", "sqlite:///./flowforge.db")
    redis_url: str = os.getenv("FLOWFORGE_REDIS_URL", "redis://localhost:6379/0")
    max_job_retries: int = int(os.getenv("FLOWFORGE_MAX_JOB_RETRIES", "3"))
    base_backoff_seconds: int = int(os.getenv("FLOWFORGE_BASE_BACKOFF_SECONDS", "2"))
    webhook_timeout_seconds: float = float(os.getenv("FLOWFORGE_WEBHOOK_TIMEOUT_SECONDS", "5"))
    grpc_host: str = os.getenv("FLOWFORGE_GRPC_HOST", "0.0.0.0")
    grpc_port: int = int(os.getenv("FLOWFORGE_GRPC_PORT", "50051"))


settings = Settings()
