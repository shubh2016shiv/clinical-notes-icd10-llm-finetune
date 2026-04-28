"""
ICD code repository port (interface) for the v3 pipeline.

LAYER: core/ports
ARCHITECTURE:
  core/services and core/retrieval depend on this protocol.
  The concrete implementation lives in infrastructure/data_preprocessing/.

  ICDCodeRepositoryPort (Protocol)
      <- OfficialICDCodeRepository (infrastructure/data_preprocessing/)

DATA FLOW:
  ICD code string -> ICDCodeRepositoryPort -> ICDCodeRecord | bool | list

DEPENDENCIES:
  - core/models/icd_codes.py (ICDCodeRecord)
  - typing (Protocol, runtime_checkable)
"""

from typing import Protocol, runtime_checkable

from clinical_note_generation_v3.core.models.icd_codes import ICDCodeRecord


@runtime_checkable
class ICDCodeRepositoryPort(Protocol):
    """
    Structural protocol for the official ICD-10-CM code repository.

    Defines the minimal interface required by core/services/ and core/retrieval/
    so those layers do not depend on the concrete infrastructure repository class.
    OfficialICDCodeRepository in infrastructure/data_preprocessing/ implements
    all of these methods.

    Note:
        The billable_records property is required by HybridICDCandidateRetriever
        to initialize the BM25 index at construction time.
    """

    @property
    def billable_records(self) -> list[ICDCodeRecord]:
        """Return all billable ICD-10-CM diagnosis records."""
        ...

    def normalize_code(self, code: str) -> str:
        """
        Normalize an ICD code to uppercase undotted form for consistent lookup.

        Args:
            code: ICD code with or without a decimal point.

        Returns:
            Uppercase code without decimal (e.g., "E11.9" -> "E119").
        """
        ...

    def get_code(self, code: str) -> ICDCodeRecord | None:
        """
        Return the ICDCodeRecord for a given dotted or undotted code.

        Args:
            code: ICD-10-CM code.

        Returns:
            Matching ICDCodeRecord, or None if not found.
        """
        ...

    def is_existing_code(self, code: str) -> bool:
        """Return True when the code exists in the official order file."""
        ...

    def is_billable_code(self, code: str) -> bool:
        """Return True when the code exists and is marked as billable."""
        ...
