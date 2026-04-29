"""
Centralized exception types for the v3 pipeline.
"""

from __future__ import annotations

from typing import Any


class ClinicalNoteGenerationError(RuntimeError):
    """
    Raised when a pipeline stage fails unexpectedly.
    """

    def __init__(
        self,
        message: str,
        *,
        correlation_id: str | None = None,
        stage_number: int | None = None,
        stage_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.correlation_id = correlation_id
        self.stage_number = stage_number
        self.stage_name = stage_name
        self.details = details or {}

        stage_label = ""
        if stage_number is not None and stage_name:
            stage_label = f" stage={stage_number}:{stage_name}"
        elif stage_name:
            stage_label = f" stage={stage_name}"

        correlation_label = f" correlation_id={correlation_id}" if correlation_id else ""
        super().__init__(f"{message}{stage_label}{correlation_label}")
