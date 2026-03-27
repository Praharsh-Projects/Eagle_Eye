from __future__ import annotations

import logging
from typing import Any

import httpx

from core.config import settings


logger = logging.getLogger(__name__)


def dispatch_webhook(webhook_url: str, payload: dict[str, Any]) -> None:
    try:
        with httpx.Client(timeout=settings.webhook_timeout_seconds) as client:
            client.post(webhook_url, json=payload)
    except httpx.HTTPError as exc:
        logger.warning("webhook dispatch failed for %s: %s", webhook_url, exc)
