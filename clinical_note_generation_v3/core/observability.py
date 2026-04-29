"""
Structured observability helpers for the v3 pipeline.
"""

from __future__ import annotations

from typing import Any

import structlog

from clinical_note_generation_v3.core.log import get_logger


class PipelineTraceCollector:
    """
    Captures per-stage structured events when verbose logging is enabled.
    """

    def __init__(
        self,
        *,
        correlation_id: str,
        enabled: bool,
        logger_name: str,
    ) -> None:
        self._correlation_id = correlation_id
        self._enabled = enabled
        self._events: list[dict[str, Any]] = []
        self._logger = get_logger(logger_name) if enabled else None
        if enabled:
            structlog.threadlocal.bind_threadlocal(correlation_id=correlation_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self,
        *,
        event_type: str,
        stage_number: int | None = None,
        stage_name: str | None = None,
        substage: str | None = None,
        status: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return

        event = {
            "event_type": event_type,
            "stage_number": stage_number,
            "stage_name": stage_name,
            "substage": substage,
            "status": status,
            "correlation_id": self._correlation_id,
            "payload": payload or {},
        }
        self._events.append(event)
        logger = self._logger
        if logger is None:
            return
        logger.info(event_type, **event)

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def close(self) -> None:
        if self._enabled:
            structlog.threadlocal.clear_threadlocal()
