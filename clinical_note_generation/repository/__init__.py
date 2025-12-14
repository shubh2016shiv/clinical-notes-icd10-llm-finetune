"""
Repository Layer - ICD-10 Data Access

This layer provides a clean abstraction over ICD-10 data sources,
whether from local JSON files or external APIs. The repository pattern:
    1. Hides data source implementation details
    2. Provides consistent interface for code lookup/search
    3. Enables easy switching between file and API sources

Submodules:
    icd10_repository.py → Main repository protocol + implementations

Dependency Rule:
    This layer depends on: core (models, exceptions)
    This layer is used by: selection, validation

Author: Shubham Singh
Date: December 2025
"""

from clinical_note_generation.repository.icd10_repository import (
    ICD10Repository,
    FileBasedICD10Repository,
)

__all__ = [
    "ICD10Repository",
    "FileBasedICD10Repository",
]
