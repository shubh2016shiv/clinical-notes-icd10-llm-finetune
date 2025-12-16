"""
Unified Code Selector - Facade for Selection Strategies

This module provides a single entry point for ICD-10 code selection,
abstracting away the complexity of multiple selection strategies.

BEFORE (4 different classes with overlapping logic):
    codes = api_manager.get_themed_icd10_codes(...)
    codes = semantic_selector.select_medically_coherent_codes(...)
    codes = vector_selector.select_codes(...)
    codes = dataset_manager.get_random_billable_codes(...)

AFTER (One unified selector):
    selector = CodeSelector(repository)
    codes = selector.select(count=5, profile="chronic_care")

Why Facade Pattern:
    1. Simplifies client code (one method to call)
    2. Hides strategy complexity
    3. Single place to configure selection behavior
    4. Easy to add new strategies without changing clients

Pipeline Position:
    Config → Repository → [CodeSelector] → Generation → Validation
                          ^^^^^^^^^^^^^^
                          You are here

Author: Shubham Singh
Date: December 2025
"""

from typing import List, Optional, Dict, Type

from loguru import logger

from clinical_note_generation.core.models import ICD10Code
from clinical_note_generation.core.enums import SelectionStrategy
from clinical_note_generation.core.exceptions import InsufficientCodesError
from clinical_note_generation.repository.icd10_repository import ICD10Repository
from clinical_note_generation.selection.strategies import (
    SelectionStrategyBase,
    RandomSelectionStrategy,
    ProfileBasedSelectionStrategy,
)


# =============================================================================
# STAGE 1: STRATEGY REGISTRY
# =============================================================================
# Maps strategy enum to implementation class.

STRATEGY_REGISTRY: Dict[SelectionStrategy, Type[SelectionStrategyBase]] = {
    SelectionStrategy.RANDOM: RandomSelectionStrategy,
    SelectionStrategy.PROFILE_BASED: ProfileBasedSelectionStrategy,
    # Future strategies can be added here:
    # SelectionStrategy.SEMANTIC: SemanticSelectionStrategy,
    # SelectionStrategy.VECTOR: VectorSelectionStrategy,
}


# =============================================================================
# STAGE 2: UNIFIED CODE SELECTOR
# =============================================================================
# Main façade that provides a simple interface to all selection strategies.


class CodeSelector:
    """
    Unified interface for ICD-10 code selection.

    What it does:
        Provides a single, simple API for selecting ICD-10 codes,
        hiding the complexity of multiple selection strategies behind
        a clean interface.

    Why it exists:
        1. Consolidates 4+ overlapping selection mechanisms into one
        2. Simplifies client code (one method to call)
        3. Enables easy strategy switching via configuration
        4. Centralizes selection logic for maintainability

    When to use:
        - Almost always - this is the primary code selection interface
        - When you need ICD-10 codes for clinical note generation

    How it works:
        STAGE 2.1: Initialize with repository and default strategy
        STAGE 2.2: On select(), lookup strategy implementation
        STAGE 2.3: Delegate to strategy with parameters
        STAGE 2.4: Return selected codes

    Example:
        >>> repo = FileBasedICD10Repository("icd10_dataset.json")
        >>> selector = CodeSelector(repo)
        >>>
        >>> # Simple selection
        >>> codes = selector.select(count=5)
        >>>
        >>> # Profile-based selection
        >>> codes = selector.select(
        ...     count=5,
        ...     profile="chronic_care",
        ...     strategy=SelectionStrategy.PROFILE_BASED
        ... )
    """

    def __init__(
        self,
        repository: ICD10Repository,
        default_strategy: SelectionStrategy = SelectionStrategy.PROFILE_BASED,
    ):
        """
        Initialize the code selector.

        Args:
            repository: ICD10Repository for code data access
            default_strategy: Strategy to use when none specified
        """
        # =====================================================================
        # STAGE 2.1: STORE DEPENDENCIES
        # =====================================================================
        self._repository = repository
        self._default_strategy = default_strategy

        # =====================================================================
        # STAGE 2.2: INITIALIZE STRATEGY CACHE
        # =====================================================================
        # Lazy-load strategies on first use
        self._strategy_cache: Dict[SelectionStrategy, SelectionStrategyBase] = {}

        logger.debug(f"CodeSelector initialized | " f"Default strategy: {default_strategy.value}")

    # =========================================================================
    # STAGE 3: MAIN SELECTION API
    # =========================================================================

    def select(
        self,
        count: int,
        profile: str = "default",
        strategy: Optional[SelectionStrategy] = None,
        **kwargs,
    ) -> List[ICD10Code]:
        """
        Select ICD-10 codes for clinical note generation.

        This is the main entry point for code selection. It:
            1. Determines which strategy to use
            2. Gets or creates the strategy instance
            3. Delegates selection to the strategy
            4. Returns the selected codes

        Args:
            count: Number of codes to select (must be > 0)
            profile: Generation profile (e.g., "chronic_care", "acute_injury")
            strategy: Selection strategy to use (defaults to configured default)
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of selected ICD10Code objects

        Raises:
            ValueError: If count is not positive
            InsufficientCodesError: If not enough codes available

        Example:
            >>> codes = selector.select(count=5, profile="chronic_care")
            >>> for code in codes:
            ...     print(f"{code.code}: {code.name}")
        """
        # =====================================================================
        # STAGE 3.1: VALIDATE INPUTS
        # =====================================================================
        if count <= 0:
            raise ValueError(f"Count must be positive, got {count}")

        # =====================================================================
        # STAGE 3.2: DETERMINE STRATEGY
        # =====================================================================
        selected_strategy = strategy or self._default_strategy

        # =====================================================================
        # STAGE 3.3: GET OR CREATE STRATEGY INSTANCE
        # =====================================================================
        strategy_instance = self._get_strategy(selected_strategy)

        # =====================================================================
        # STAGE 3.4: DELEGATE TO STRATEGY
        # =====================================================================
        logger.info(
            f"Selecting {count} codes | "
            f"Profile: {profile} | "
            f"Strategy: {selected_strategy.value}"
        )

        try:
            selected_codes = strategy_instance.select(count=count, profile=profile, **kwargs)

            logger.info(
                f"Selected {len(selected_codes)} codes: " f"{[c.code for c in selected_codes]}"
            )

            return selected_codes

        except InsufficientCodesError:
            # Re-raise with more context
            raise
        except Exception as e:
            logger.error(f"Selection failed with {selected_strategy.value}: {e}")
            raise

    def select_random(self, count: int) -> List[ICD10Code]:
        """
        Convenience method for random selection.

        Args:
            count: Number of codes to select

        Returns:
            Randomly selected codes
        """
        return self.select(count=count, strategy=SelectionStrategy.RANDOM)

    def select_for_profile(self, count: int, profile: str) -> List[ICD10Code]:
        """
        Convenience method for profile-based selection.

        Args:
            count: Number of codes to select
            profile: Profile name (e.g., "chronic_care")

        Returns:
            Profile-guided selection of codes
        """
        return self.select(count=count, profile=profile, strategy=SelectionStrategy.PROFILE_BASED)

    # =========================================================================
    # STAGE 4: STRATEGY MANAGEMENT
    # =========================================================================

    def _get_strategy(self, strategy: SelectionStrategy) -> SelectionStrategyBase:
        """
        Get or create strategy instance (lazy loading with cache).

        Args:
            strategy: Strategy enum value

        Returns:
            Initialized strategy instance

        Raises:
            ValueError: If strategy not in registry
        """
        # Check cache first
        if strategy in self._strategy_cache:
            return self._strategy_cache[strategy]

        # Look up in registry
        if strategy not in STRATEGY_REGISTRY:
            available = [s.value for s in STRATEGY_REGISTRY.keys()]
            raise ValueError(f"Unknown strategy: {strategy.value}. " f"Available: {available}")

        # Create and cache
        strategy_class = STRATEGY_REGISTRY[strategy]
        instance = strategy_class(self._repository)
        self._strategy_cache[strategy] = instance

        logger.debug(f"Created strategy instance: {strategy.value}")
        return instance

    def register_strategy(
        self, strategy_enum: SelectionStrategy, strategy_class: Type[SelectionStrategyBase]
    ) -> None:
        """
        Register a new selection strategy at runtime.

        This allows extending the selector with custom strategies
        without modifying this module.

        Args:
            strategy_enum: Strategy enum value
            strategy_class: Class implementing SelectionStrategyBase

        Example:
            >>> selector.register_strategy(
            ...     SelectionStrategy.VECTOR,
            ...     VectorSelectionStrategy
            ... )
        """
        STRATEGY_REGISTRY[strategy_enum] = strategy_class
        # Clear cache to force re-creation
        self._strategy_cache.pop(strategy_enum, None)
        logger.info(f"Registered strategy: {strategy_enum.value}")

    # =========================================================================
    # STAGE 5: PROPERTIES
    # =========================================================================

    @property
    def available_strategies(self) -> List[str]:
        """List of available strategy names."""
        return [s.value for s in STRATEGY_REGISTRY.keys()]

    @property
    def repository(self) -> ICD10Repository:
        """Access to underlying repository."""
        return self._repository
