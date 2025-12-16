"""
Validation Layer - Clinical Note Validation

This layer validates that generated clinical notes properly support
their assigned ICD-10 codes.

Submodules:
    note_validator.py → Rule-based and LLM validation

Dependency Rule:
    This layer depends on: core, clients (for LLM validation)
    This layer is used by: pipeline (orchestrator)

Author: Shubham Singh
Date: December 2025
"""

from clinical_note_generation.validation.note_validator import (
    NoteValidator,
    RuleBasedChecks,
)

__all__ = [
    "NoteValidator",
    "RuleBasedChecks",
]
