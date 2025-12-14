"""
ICD-10 Repository - Data Access Abstraction

This module provides a clean interface for accessing ICD-10 code data,
abstracting away whether codes come from a local JSON file or external API.

Architecture:
    ICD10Repository (Protocol)
    ├── FileBasedICD10Repository  → Loads from local JSON file
    └── APIBasedICD10Repository   → Fetches from NLM Clinical Tables API (future)

Why Repository Pattern:
    1. Single responsibility: data access logic in one place
    2. Testability: can mock repository for unit tests
    3. Flexibility: switch data sources without changing business logic
    4. Caching: centralized caching logic

Pipeline Position:
    Config → [Repository] → Selection → Generation → Validation → Output
              ^^^^^^^^^^^
              You are here

Usage:
    from clinical_note_generation.repository import FileBasedICD10Repository

    repo = FileBasedICD10Repository("path/to/icd10_dataset.json")
    code = repo.get_code("E11.9")
    codes = repo.search_codes("diabetes", max_results=10)

Author: Shubham Singh
Date: December 2025
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger

from clinical_note_generation.core.models import ICD10Code
from clinical_note_generation.core.exceptions import (
    CodeNotFoundError,
    InsufficientCodesError,
    DatasetLoadError,
)


# =============================================================================
# STAGE 1: REPOSITORY PROTOCOL (INTERFACE)
# =============================================================================
# Defines the contract that all repository implementations must follow.


@runtime_checkable
class ICD10Repository(Protocol):
    """
    Protocol defining the interface for ICD-10 code repositories.

    What it does:
        Defines the contract for any ICD-10 data source, enabling
        dependency injection and easy testing with mock implementations.

    Why it exists:
        1. Decouples business logic from data access implementation
        2. Enables switching between file/API without code changes
        3. Makes repository mockable for unit testing

    When to use:
        - As type hint for dependency injection
        - When implementing new data sources
        - When mocking for tests

    Required Methods:
        get_code(code_id)     → Get single code by ID
        search_codes(query)   → Search codes by text
        get_billable_codes()  → Get codes suitable for billing
        get_codes_by_category() → Filter by diagnosis category
    """

    def get_code(self, code_id: str) -> Optional[ICD10Code]:
        """
        Retrieve a single ICD-10 code by its identifier.

        Args:
            code_id: The ICD-10 code string (e.g., "E11.9")

        Returns:
            ICD10Code object if found, None otherwise
        """
        ...

    def search_codes(self, query: str, max_results: int = 100) -> List[ICD10Code]:
        """
        Search for codes matching a text query.

        Args:
            query: Search term (e.g., "diabetes", "fracture")
            max_results: Maximum number of results to return

        Returns:
            List of matching ICD10Code objects
        """
        ...

    def get_billable_codes(
        self, count: Optional[int] = None, category_filter: Optional[str] = None
    ) -> List[ICD10Code]:
        """
        Get billable ICD-10 codes, optionally filtered by category.

        Args:
            count: Number of codes to return (None = all)
            category_filter: Optional category to filter by

        Returns:
            List of billable ICD10Code objects
        """
        ...

    def get_random_billable_codes(
        self, count: int, category_filter: Optional[str] = None
    ) -> List[ICD10Code]:
        """
        Get random selection of billable codes.

        Args:
            count: Number of codes to select
            category_filter: Optional category to filter by

        Returns:
            Randomly selected list of ICD10Code objects

        Raises:
            InsufficientCodesError: If not enough codes available
        """
        ...

    @property
    def total_codes(self) -> int:
        """Total number of codes in repository."""
        ...

    @property
    def billable_code_count(self) -> int:
        """Number of billable codes in repository."""
        ...


# =============================================================================
# STAGE 2: FILE-BASED REPOSITORY IMPLEMENTATION
# =============================================================================
# Loads ICD-10 codes from a local JSON file with in-memory indexing.


class FileBasedICD10Repository:
    """
    ICD-10 repository backed by a local JSON file.

    What it does:
        Loads ICD-10 codes from a JSON file into memory and provides
        fast lookups using dictionary-based indexing.

    Why it exists:
        1. Works offline without API dependencies
        2. Fast lookups after initial load
        3. Supports the large 100MB+ ICD-10 dataset

    When to use:
        - When you have the ICD-10 dataset JSON file available
        - When API access is not needed or not available
        - For development and testing

    How it works:
        STAGE 2.1: Load JSON file into memory
        STAGE 2.2: Index codes by ID for O(1) lookup
        STAGE 2.3: Build billable codes list
        STAGE 2.4: Index by category for filtered queries

    Performance:
        - Initial load: O(n) where n = number of codes (~100k)
        - Lookup by ID: O(1) dictionary access
        - Search by text: O(n) scan (consider vector search for better perf)
        - Memory: ~200MB for full dataset with indices

    Example:
        >>> repo = FileBasedICD10Repository("icd10_dataset.json")
        >>> code = repo.get_code("E11.9")
        >>> codes = repo.search_codes("diabetes", max_results=10)
    """

    # Class-level cache to avoid reloading large dataset
    _dataset_cache: Dict[str, Dict] = {}

    def __init__(self, dataset_path: str):
        """
        Initialize repository from JSON file.

        STAGE 2.1: Validate file path
        STAGE 2.2: Load or retrieve from cache
        STAGE 2.3: Build lookup indices

        Args:
            dataset_path: Path to ICD-10 dataset JSON file

        Raises:
            DatasetLoadError: If file cannot be loaded
        """
        # =====================================================================
        # STAGE 2.1: VALIDATE FILE PATH
        # =====================================================================
        self._dataset_path = Path(dataset_path)

        if not self._dataset_path.exists():
            raise DatasetLoadError(str(self._dataset_path), "File not found")

        # =====================================================================
        # STAGE 2.2: INITIALIZE DATA STRUCTURES
        # =====================================================================
        # Primary index: code_id → ICD10Code
        self._codes_by_id: Dict[str, ICD10Code] = {}

        # Secondary indices for fast filtering
        self._billable_codes: List[ICD10Code] = []
        self._codes_by_category: Dict[str, List[ICD10Code]] = {}
        self._codes_by_first_char: Dict[str, List[ICD10Code]] = {}

        # =====================================================================
        # STAGE 2.3: LOAD AND INDEX DATA
        # =====================================================================
        self._load_dataset()

        logger.info(
            f"FileBasedICD10Repository initialized | "
            f"Total: {self.total_codes:,} | "
            f"Billable: {self.billable_code_count:,}"
        )

    # =========================================================================
    # STAGE 3: DATASET LOADING
    # =========================================================================

    def _load_dataset(self) -> None:
        """
        Load dataset from JSON and build indices.

        STAGE 3.1: Check cache for previously loaded data
        STAGE 3.2: Load from file if not cached
        STAGE 3.3: Parse each record into ICD10Code
        STAGE 3.4: Build primary index (by code ID)
        STAGE 3.5: Build secondary indices (billable, category, chapter)
        STAGE 3.6: Cache raw data for future instances
        """
        cache_key = str(self._dataset_path.absolute())

        # -----------------------------------------------------------------
        # 3.1: Check cache
        # -----------------------------------------------------------------
        if cache_key in self._dataset_cache:
            logger.debug(f"Using cached dataset: {cache_key}")
            raw_data = self._dataset_cache[cache_key]
        else:
            # -----------------------------------------------------------------
            # 3.2: Load from file
            # -----------------------------------------------------------------
            logger.info(f"Loading ICD-10 dataset from: {self._dataset_path}")
            try:
                with open(self._dataset_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                # 3.6: Cache for future use
                self._dataset_cache[cache_key] = raw_data
                logger.info(f"Loaded {len(raw_data):,} records from dataset")

            except json.JSONDecodeError as e:
                raise DatasetLoadError(str(self._dataset_path), f"Invalid JSON: {e}")
            except PermissionError:
                raise DatasetLoadError(str(self._dataset_path), "Permission denied")
            except Exception as e:
                raise DatasetLoadError(str(self._dataset_path), str(e))

        # -----------------------------------------------------------------
        # 3.3-3.4: Parse and build primary index (deduplication)
        # -----------------------------------------------------------------
        for record in raw_data:
            code = ICD10Code.from_dict(record)
            # This handles deduplication by overwriting existing keys
            self._codes_by_id[code.code] = code

        # -----------------------------------------------------------------
        # 3.5: Build secondary indices from UNIQUE codes
        # -----------------------------------------------------------------
        for code in self._codes_by_id.values():
            if code.is_billable:
                self._billable_codes.append(code)

            # Index by category
            category = code.diagnosis_category or "Uncategorized"
            if category not in self._codes_by_category:
                self._codes_by_category[category] = []
            self._codes_by_category[category].append(code)

            # Index by first character (chapter)
            first_char = code.first_character
            if first_char not in self._codes_by_first_char:
                self._codes_by_first_char[first_char] = []
            self._codes_by_first_char[first_char].append(code)

    # =========================================================================
    # STAGE 4: PUBLIC API - CODE RETRIEVAL
    # =========================================================================

    def get_code(self, code_id: str) -> Optional[ICD10Code]:
        """
        Retrieve a single ICD-10 code by its identifier.

        Time Complexity: O(1) dictionary lookup

        Args:
            code_id: The ICD-10 code string (e.g., "E11.9")

        Returns:
            ICD10Code object if found, None otherwise
        """
        return self._codes_by_id.get(code_id)

    def get_code_or_raise(self, code_id: str) -> ICD10Code:
        """
        Retrieve a single ICD-10 code, raising error if not found.

        Args:
            code_id: The ICD-10 code string

        Returns:
            ICD10Code object

        Raises:
            CodeNotFoundError: If code doesn't exist
        """
        code = self.get_code(code_id)
        if code is None:
            raise CodeNotFoundError(code_id)
        return code

    # =========================================================================
    # STAGE 5: PUBLIC API - CODE SEARCH
    # =========================================================================

    def search_codes(
        self, query: str, max_results: int = 100, billable_only: bool = True
    ) -> List[ICD10Code]:
        """
        Search for codes matching a text query.

        Algorithm:
            1. Normalize query to lowercase
            2. Scan all codes checking name and category
            3. Return matches up to max_results

        Time Complexity: O(n) where n = number of codes

        Args:
            query: Search term (e.g., "diabetes", "fracture")
            max_results: Maximum number of results to return
            billable_only: If True, only return billable codes

        Returns:
            List of matching ICD10Code objects
        """
        query_lower = query.lower()
        results = []

        search_pool = self._billable_codes if billable_only else self._codes_by_id.values()

        for code in search_pool:
            # Check if query matches code name or category
            if query_lower in code.name.lower() or (
                code.category_name and query_lower in code.category_name.lower()
            ):
                results.append(code)
                if len(results) >= max_results:
                    break

        return results

    # =========================================================================
    # STAGE 6: PUBLIC API - BILLABLE CODE SELECTION
    # =========================================================================

    def get_billable_codes(
        self, count: Optional[int] = None, category_filter: Optional[str] = None
    ) -> List[ICD10Code]:
        """
        Get billable ICD-10 codes, optionally filtered by category.

        Args:
            count: Number of codes to return (None = all)
            category_filter: Optional category to filter by

        Returns:
            List of billable ICD10Code objects
        """
        if category_filter:
            codes = [c for c in self._billable_codes if c.diagnosis_category == category_filter]
        else:
            codes = self._billable_codes

        if count is not None:
            return codes[:count]
        return codes

    def get_random_billable_codes(
        self, count: int, category_filter: Optional[str] = None
    ) -> List[ICD10Code]:
        """
        Get random selection of billable codes.

        Args:
            count: Number of codes to select
            category_filter: Optional category to filter by

        Returns:
            Randomly selected list of ICD10Code objects

        Raises:
            InsufficientCodesError: If not enough codes available
        """
        available = self.get_billable_codes(category_filter=category_filter)

        if len(available) < count:
            raise InsufficientCodesError(
                requested=count,
                available=len(available),
                criteria=f"category={category_filter}" if category_filter else None,
            )

        return random.sample(available, count)

    def get_codes_by_chapter(self, chapter: str, billable_only: bool = True) -> List[ICD10Code]:
        """
        Get codes by ICD-10 chapter (first character).

        Args:
            chapter: First character of codes (e.g., "E" for endocrine)
            billable_only: If True, only return billable codes

        Returns:
            List of codes in that chapter
        """
        chapter_upper = chapter.upper()
        codes = self._codes_by_first_char.get(chapter_upper, [])

        if billable_only:
            return [c for c in codes if c.is_billable]
        return codes

    # =========================================================================
    # STAGE 7: PROPERTIES
    # =========================================================================

    @property
    def total_codes(self) -> int:
        """Total number of codes in repository."""
        return len(self._codes_by_id)

    @property
    def billable_code_count(self) -> int:
        """Number of billable codes in repository."""
        return len(self._billable_codes)

    @property
    def categories(self) -> List[str]:
        """List of all diagnosis categories."""
        return sorted(self._codes_by_category.keys())

    @property
    def chapters(self) -> List[str]:
        """List of all ICD-10 chapters (first characters)."""
        return sorted(self._codes_by_first_char.keys())
