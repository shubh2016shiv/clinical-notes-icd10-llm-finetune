"""
Official ICD-10-CM in-memory repository.

LAYER: infrastructure/data_preprocessing
ARCHITECTURE:
  icd_order_file_parser.iter_icd_order_records()
      -> OfficialICDCodeRepository.__init__()  (loads all records into memory)
      -> dict lookup by undotted code and by normalized (dotted) code
      -> get_code(), is_billable_code(), search_descriptions()

  Implements ICDCodeRepositoryPort from core/ports/, so it can be injected
  into SelectionValidator and HybridICDCandidateRetriever without those core
  components depending on this infrastructure module.

DATA FLOW:
  Path (order file) -> OfficialICDCodeRepository -> ICDCodeRecord lookups

DEPENDENCIES:
  - core/models/icd_codes.py (ICDCodeRecord)
  - core/ports/icd_repository_port.py (ICDCodeRepositoryPort — structural, not inherited)
  - infrastructure/data_preprocessing/icd_order_file_parser.py (iter_icd_order_records)
  - stdlib (pathlib)
"""

from pathlib import Path

from clinical_note_generation_v3.core.models.icd_codes import ICDCodeRecord
from clinical_note_generation_v3.infrastructure.data_preprocessing.icd_order_file_parser import (
    iter_icd_order_records,
)


class OfficialICDCodeRepository:
    """
    In-memory repository backed by the official ICD-10-CM order file.

    Loads all records into two lookup dicts at construction time:
    one keyed by undotted code ("E119") and one by normalized code ("E11.9").
    Provides exact lookup, billable filtering, and simple keyword search.

    Args:
        order_file_path: Absolute path to the official ICD-10-CM order file.

    Returns:
        Loaded repository with O(1) code lookup.

    Raises:
        FileNotFoundError: If the official order file does not exist.
        ValueError: If the order file contains malformed rows.

    Example:
        >>> from clinical_note_generation_v3.config.settings import V3PipelineSettings
        >>> settings = V3PipelineSettings()
        >>> repo = OfficialICDCodeRepository(settings.official_icd_order_path)
        >>> repo.get_code("E11.9").is_billable
        True
    """

    def __init__(self, order_file_path: Path) -> None:
        self._order_file_path = order_file_path
        self._records_by_undotted_code: dict[str, ICDCodeRecord] = {
            record.code: record for record in iter_icd_order_records(order_file_path)
        }
        self._records_by_normalized_code: dict[str, ICDCodeRecord] = {
            record.normalized_code.upper(): record
            for record in self._records_by_undotted_code.values()
        }

    @classmethod
    def from_default_settings(cls) -> "OfficialICDCodeRepository":
        """
        Create a repository using default V3 pipeline settings.

        Returns:
            Repository loaded from the project-root official ICD-10-CM folder.

        Raises:
            FileNotFoundError: If the configured order file is missing.

        Example:
            >>> OfficialICDCodeRepository.from_default_settings().total_records > 0
            True
        """
        from clinical_note_generation_v3.config.settings import V3PipelineSettings

        settings = V3PipelineSettings()
        return cls(settings.official_icd_order_path)

    @property
    def total_records(self) -> int:
        """Return total official rows loaded, including non-billable header codes."""
        return len(self._records_by_undotted_code)

    @property
    def billable_records(self) -> list[ICDCodeRecord]:
        """Return all billable ICD-10-CM diagnosis records."""
        return [record for record in self._records_by_undotted_code.values() if record.is_billable]

    def normalize_code(self, code: str) -> str:
        """
        Normalize an ICD code to uppercase undotted form for consistent lookup.

        Args:
            code: ICD code with or without a decimal point.

        Returns:
            Uppercase code without decimal (e.g., "E11.9" -> "E119").

        Raises:
            None.

        Example:
            >>> repo.normalize_code("E11.9")
            'E119'
        """
        return code.strip().upper().replace(".", "")

    def get_code(self, code: str) -> ICDCodeRecord | None:
        """
        Return the ICDCodeRecord for a given dotted or undotted code.

        Args:
            code: ICD-10-CM code (with or without decimal point).

        Returns:
            Matching ICDCodeRecord, or None if not found in the official file.

        Raises:
            None.

        Example:
            >>> repo.get_code("E11.9").normalized_code
            'E11.9'
        """
        normalized = self.normalize_code(code)
        return self._records_by_undotted_code.get(normalized)

    def is_existing_code(self, code: str) -> bool:
        """Return True when the code exists anywhere in the official order file."""
        return self.get_code(code) is not None

    def is_billable_code(self, code: str) -> bool:
        """Return True when the code exists and is marked as a billable diagnosis code."""
        record = self.get_code(code)
        return bool(record and record.is_billable)

    def search_descriptions(self, query: str, *, limit: int = 25) -> list[ICDCodeRecord]:
        """
        Perform simple case-insensitive keyword lookup over billable descriptions.

        Scores each billable record by how many query terms appear in the combined
        long + short description. Returns records sorted by descending score, then
        by order_number for deterministic tie-breaking.

        Args:
            query: Search text (space-separated terms).
            limit: Maximum number of records to return.

        Returns:
            Billable records whose descriptions contain at least one query term,
            sorted by match score descending.

        Raises:
            None.

        Example:
            >>> any(r.normalized_code == "E78.2" for r in repo.search_descriptions("mixed hyperlipidemia"))
            True
        """
        search_terms = _expand_search_terms_from_query(query)
        if not search_terms:
            return []

        scored_matches: list[tuple[int, ICDCodeRecord]] = []
        for record in self.billable_records:
            haystack = f"{record.long_description} {record.short_description}".lower()
            match_score = sum(1 for term in search_terms if term in haystack)
            if match_score:
                scored_matches.append((match_score, record))

        scored_matches.sort(key=lambda item: (-item[0], item[1].order_number or 0))
        return [record for _, record in scored_matches[:limit]]


def _expand_search_terms_from_query(query: str) -> list[str]:
    raw_terms = [term.lower().strip(" ,.;:()[]") for term in query.split() if term.strip()]
    return list(dict.fromkeys(term for term in raw_terms if len(term) > 1))
