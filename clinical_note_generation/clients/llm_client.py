"""
LLM Client Protocol and Base Implementation

This module defines the interface for LLM clients and provides a base
class with common functionality (rate limiting, error handling).

Protocol Pattern:
    - LLMClientProtocol defines the interface
    - BaseLLMClient provides common implementation
    - Concrete clients (GeminiClient, OpenAIClient) extend base

Why This Design:
    1. Dependency Inversion: Business logic depends on protocol, not concrete class
    2. Open/Closed: Add new providers without modifying existing code
    3. Liskov Substitution: Any client implementing protocol is interchangeable

Author: Shubham Singh
Date: December 2025
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

from loguru import logger

from clinical_note_generation.core.exceptions import (
    LLMError,
    LLMRateLimitError,
)


# =============================================================================
# STAGE 1: LLM CLIENT PROTOCOL
# =============================================================================
# Defines the contract that all LLM clients must follow.


@runtime_checkable
class LLMClientProtocol(Protocol):
    """
    Protocol defining the interface for LLM clients.

    What it does:
        Specifies the exact methods any LLM client must implement,
        enabling type-safe dependency injection and polymorphism.

    Why it exists:
        1. Type safety: IDE can verify correct usage
        2. Documentation: Clear contract for implementers
        3. Testing: Easy to create mock implementations

    Required Methods:
        generate(prompt) → Generate text from prompt

    Optional Properties:
        model_name → Name of the model being used
        provider_name → Name of the provider (gemini, openai)
    """

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The generation prompt

        Returns:
            Generated text string

        Raises:
            LLMError: If generation fails
        """
        ...

    @property
    def model_name(self) -> str:
        """Name of the model being used."""
        ...

    @property
    def provider_name(self) -> str:
        """Name of the LLM provider."""
        ...


# =============================================================================
# STAGE 2: BASE LLM CLIENT (ABSTRACT)
# =============================================================================
# Provides common functionality for all LLM clients.


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients with common functionality.

    What it does:
        Provides rate limiting, error handling, and logging that
        all LLM clients need, so concrete implementations only
        need to implement the API-specific logic.

    Why it exists:
        1. DRY: Common logic in one place
        2. Consistent behavior: All clients have same retry/rate limit
        3. Easier maintenance: Fix bugs in one place

    What subclasses must implement:
        - _call_api(prompt): Actual API call
        - model_name: Property returning model name
        - provider_name: Property returning provider name

    What base class provides:
        - Rate limiting between calls
        - Retry logic for transient errors
        - Logging and metrics
    """

    def __init__(
        self, api_key: str, model_name: str, rate_limit_delay: float = 0.5, max_retries: int = 3
    ):
        """
        Initialize base LLM client.

        Args:
            api_key: API key for the provider
            model_name: Name of model to use
            rate_limit_delay: Seconds to wait between API calls
            max_retries: Maximum retry attempts for failed calls
        """
        # =====================================================================
        # STAGE 2.1: STORE CONFIGURATION
        # =====================================================================
        self._api_key = api_key
        self._model_name = model_name
        self._rate_limit_delay = rate_limit_delay
        self._max_retries = max_retries

        # =====================================================================
        # STAGE 2.2: TRACKING STATE
        # =====================================================================
        self._last_call_time: Optional[float] = None
        self._total_calls = 0
        self._failed_calls = 0

    # =========================================================================
    # STAGE 3: PUBLIC API
    # =========================================================================

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt with rate limiting and retry.

        Algorithm:
            1. Apply rate limiting (wait if needed)
            2. Call API with retry logic
            3. Track metrics
            4. Return result

        Args:
            prompt: The generation prompt

        Returns:
            Generated text

        Raises:
            LLMError: If all retries fail
        """
        # Step 1: Rate limiting
        self._apply_rate_limit()

        # Step 2: Call with retry
        last_error = None

        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._call_api(prompt)
                self._total_calls += 1
                return result

            except LLMRateLimitError as e:
                last_error = e
                wait_time = e.retry_after or (2**attempt)
                logger.warning(
                    f"Rate limited by {self.provider_name}, "
                    f"waiting {wait_time}s (attempt {attempt})"
                )
                time.sleep(wait_time)

            except LLMError as e:
                last_error = e
                self._failed_calls += 1
                logger.warning(f"LLM call failed (attempt {attempt}/{self._max_retries}): {e}")
                time.sleep(1)  # Brief wait before retry

            except Exception as e:
                last_error = LLMError(str(e), provider=self.provider_name, original_error=e)
                self._failed_calls += 1
                logger.error(f"Unexpected error in LLM call: {e}")

        # All retries failed
        raise LLMError(
            f"Generation failed after {self._max_retries} attempts",
            provider=self.provider_name,
            original_error=last_error,
        )

    # =========================================================================
    # STAGE 4: ABSTRACT METHODS
    # =========================================================================

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """
        Make the actual API call. Must be implemented by subclasses.

        Args:
            prompt: The generation prompt

        Returns:
            Generated text from API

        Raises:
            LLMError: If API call fails
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'gemini', 'openai')."""
        ...

    # =========================================================================
    # STAGE 5: COMMON IMPLEMENTATION
    # =========================================================================

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        if self._last_call_time is not None:
            elapsed = time.time() - self._last_call_time
            if elapsed < self._rate_limit_delay:
                wait_time = self._rate_limit_delay - elapsed
                time.sleep(wait_time)

        self._last_call_time = time.time()

    # =========================================================================
    # STAGE 6: METRICS
    # =========================================================================

    @property
    def total_calls(self) -> int:
        """Total number of successful API calls."""
        return self._total_calls

    @property
    def failed_calls(self) -> int:
        """Number of failed API calls."""
        return self._failed_calls

    @property
    def success_rate(self) -> float:
        """Percentage of successful calls."""
        total = self._total_calls + self._failed_calls
        if total == 0:
            return 100.0
        return (self._total_calls / total) * 100
