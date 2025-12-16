"""
Selection Strategies - Pluggable Code Selection Algorithms

This module implements the Strategy Pattern for ICD-10 code selection,
allowing different selection algorithms to be used interchangeably.

Strategy Hierarchy:
    SelectionStrategyBase (Abstract)
    ├── RandomSelectionStrategy     → Fast, random selection
    ├── ProfileBasedSelectionStrategy → Profile-guided selection
    └── (Future) SemanticSelectionStrategy → Co-occurrence based

Why Strategy Pattern:
    1. Open/Closed Principle: Add new strategies without modifying existing code
    2. Runtime flexibility: Switch strategies based on configuration
    3. Testability: Each strategy is independently testable
    4. Single Responsibility: Each strategy has one selection algorithm

Pipeline Position:
    Config → Repository → [Selection Strategies] → Generation → Validation
                          ^^^^^^^^^^^^^^^^^^^^^^
                          You are here

Author: Shubham Singh
Date: December 2025
"""

import random
from abc import ABC, abstractmethod
from typing import List

from loguru import logger

from clinical_note_generation.core.models import ICD10Code
from clinical_note_generation.core.constants import GENERATION_PROFILES
from clinical_note_generation.repository.icd10_repository import ICD10Repository


# =============================================================================
# STAGE 1: ABSTRACT BASE STRATEGY
# =============================================================================
# Defines the contract for all selection strategies.


class SelectionStrategyBase(ABC):
    """
    Abstract base class for ICD-10 code selection strategies.

    What it does:
        Defines the interface that all selection strategies must implement,
        enabling polymorphic strategy switching.

    Why it exists:
        1. Enforces consistent interface across strategies
        2. Enables type hinting for strategy injection
        3. Provides common utility methods

    When to extend:
        - Adding a new selection algorithm (e.g., vector-based)
        - Implementing domain-specific selection logic

    Template Method Pattern:
        Subclasses implement `_do_selection()`, base class handles common logic
        in `select()` (validation, logging, error handling).
    """

    def __init__(self, repository: ICD10Repository):
        """
        Initialize strategy with repository dependency.

        Args:
            repository: ICD10Repository for accessing code data
        """
        self._repository = repository

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name of this strategy."""
        ...

    def select(self, count: int, profile: str = "default", **kwargs) -> List[ICD10Code]:
        """
        Select ICD-10 codes using this strategy.

        Template Method Pattern:
            1. Validate inputs
            2. Delegate to strategy-specific _do_selection()
            3. Validate outputs
            4. Log results

        Args:
            count: Number of codes to select
            profile: Generation profile name
            **kwargs: Strategy-specific parameters

        Returns:
            List of selected ICD10Code objects

        Raises:
            InsufficientCodesError: If not enough codes available
            ValueError: If count is invalid
        """
        # Step 1: Validate inputs
        if count <= 0:
            raise ValueError(f"Count must be positive, got {count}")

        # Step 2: Delegate to strategy-specific implementation
        selected_codes = self._do_selection(count, profile, **kwargs)

        # Step 3: Validate we got enough codes
        if len(selected_codes) < count:
            logger.warning(
                f"Strategy {self.strategy_name} returned {len(selected_codes)} "
                f"codes, requested {count}"
            )

        # Step 4: Log results
        logger.debug(
            f"[{self.strategy_name}] Selected {len(selected_codes)} codes for "
            f"profile '{profile}': {[c.code for c in selected_codes]}"
        )

        return selected_codes

    @abstractmethod
    def _do_selection(self, count: int, profile: str, **kwargs) -> List[ICD10Code]:
        """
        Strategy-specific selection implementation.

        Args:
            count: Number of codes to select
            profile: Generation profile name
            **kwargs: Additional parameters

        Returns:
            List of selected ICD10Code objects
        """
        ...


# =============================================================================
# STAGE 2: RANDOM SELECTION STRATEGY
# =============================================================================
# Simple random selection from billable codes. Fast but low medical coherence.


class RandomSelectionStrategy(SelectionStrategyBase):
    """
    Random selection from billable ICD-10 codes.

    What it does:
        Randomly samples from all available billable codes without
        considering medical coherence or relationships.

    Why it exists:
        1. Baseline strategy for testing
        2. Fast selection when coherence not important
        3. Stress testing with diverse code combinations

    When to use:
        - Testing/debugging the pipeline
        - When speed matters more than medical accuracy
        - Generating diverse, random training examples

    Performance:
        - Time: O(n) for sampling where n = billable codes
        - Medical Coherence: Low (random combinations)
    """

    @property
    def strategy_name(self) -> str:
        return "RandomSelection"

    def _do_selection(self, count: int, profile: str, **kwargs) -> List[ICD10Code]:
        """
        Select random billable codes.

        Algorithm:
            1. Get all billable codes from repository
            2. Randomly sample `count` codes

        Args:
            count: Number of codes to select
            profile: Ignored for random selection

        Returns:
            Randomly selected ICD10Code objects
        """
        return self._repository.get_random_billable_codes(count)


# =============================================================================
# STAGE 3: PROFILE-BASED SELECTION STRATEGY
# =============================================================================
# Selects codes based on patient profile (chronic_care, acute_injury, etc.)


class ProfileBasedSelectionStrategy(SelectionStrategyBase):
    """
    Profile-guided selection for medically coherent code combinations.

    What it does:
        Uses predefined patient profiles (chronic_care, acute_injury, etc.)
        to select codes that commonly appear together. Searches for primary
        conditions first, then adds related comorbidities.

    Why it exists:
        1. Produces more realistic training data
        2. Mimics real clinical documentation patterns
        3. Balances speed with medical coherence

    When to use:
        - Production dataset generation
        - When medical coherence matters
        - Generating themed clinical notes

    How it works:
        STAGE 3.1: Lookup profile configuration
        STAGE 3.2: Search for primary condition codes
        STAGE 3.3: Search for comorbidity codes
        STAGE 3.4: Combine and deduplicate

    Performance:
        - Time: O(n * search_count) for text searches
        - Medical Coherence: Medium-High (profile-guided)
    """

    @property
    def strategy_name(self) -> str:
        return "ProfileBasedSelection"

    def _do_selection(self, count: int, profile: str, **kwargs) -> List[ICD10Code]:
        """
        Select codes based on patient profile.

        Algorithm:
            1. Get profile configuration (primary conditions, comorbidities)
            2. Calculate primary vs comorbidity split
            3. Search for primary condition codes
            4. Search for comorbidity codes
            5. Fill remaining slots randomly if needed
            6. Combine and deduplicate

        Args:
            count: Number of codes to select
            profile: Profile name (e.g., "chronic_care")

        Returns:
            Profile-guided selection of ICD10Code objects
        """
        # =====================================================================
        # STAGE 3.1: LOOKUP PROFILE CONFIGURATION
        # =====================================================================
        profile_config = GENERATION_PROFILES.get(profile, GENERATION_PROFILES["default"])

        primary_terms = profile_config.get("primary_search_terms", [])
        comorbidity_terms = profile_config.get("comorbidity_terms", [])
        comorbidity_ratio = profile_config.get("comorbidity_ratio", 0.0)

        # If no primary terms, fall back to random selection
        if not primary_terms:
            logger.debug(f"Profile '{profile}' has no primary terms, using random selection")
            return self._repository.get_random_billable_codes(count)

        # =====================================================================
        # STAGE 3.2: CALCULATE SPLIT
        # =====================================================================
        # How many codes should be comorbidities vs primary conditions
        comorbidity_count = int(count * comorbidity_ratio)
        primary_count = count - comorbidity_count

        # =====================================================================
        # STAGE 3.3: SEARCH FOR PRIMARY CONDITION CODES
        # =====================================================================
        primary_codes: List[ICD10Code] = []

        # Shuffle terms for variety
        shuffled_terms = primary_terms.copy()
        random.shuffle(shuffled_terms)

        for term in shuffled_terms:
            if len(primary_codes) >= primary_count:
                break

            # Search for codes matching this term
            matches = self._repository.search_codes(term, max_results=10, billable_only=True)

            # Add matches that aren't already selected
            for code in matches:
                if code not in primary_codes and len(primary_codes) < primary_count:
                    primary_codes.append(code)

        # =====================================================================
        # STAGE 3.4: SEARCH FOR COMORBIDITY CODES
        # =====================================================================
        comorbidity_codes: List[ICD10Code] = []

        if comorbidity_count > 0 and comorbidity_terms:
            shuffled_comorbidities = comorbidity_terms.copy()
            random.shuffle(shuffled_comorbidities)

            for term in shuffled_comorbidities:
                if len(comorbidity_codes) >= comorbidity_count:
                    break

                matches = self._repository.search_codes(term, max_results=5, billable_only=True)

                for code in matches:
                    if (
                        code not in primary_codes
                        and code not in comorbidity_codes
                        and len(comorbidity_codes) < comorbidity_count
                    ):
                        comorbidity_codes.append(code)

        # =====================================================================
        # STAGE 3.5: COMBINE AND FILL REMAINING
        # =====================================================================
        selected = primary_codes + comorbidity_codes

        # If we don't have enough, fill with random codes
        remaining_needed = count - len(selected)
        if remaining_needed > 0:
            logger.debug(
                f"Profile '{profile}' only found {len(selected)} codes, "
                f"filling {remaining_needed} with random"
            )
            try:
                # Get random codes that aren't already selected
                all_billable = self._repository.get_billable_codes()
                available = [c for c in all_billable if c not in selected]

                if available:
                    fill_count = min(remaining_needed, len(available))
                    selected.extend(random.sample(available, fill_count))
            except Exception as e:
                logger.warning(f"Could not fill remaining codes: {e}")

        return selected[:count]  # Ensure we don't exceed requested count
