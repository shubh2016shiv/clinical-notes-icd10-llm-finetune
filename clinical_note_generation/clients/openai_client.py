"""
OpenAI Client - OpenAI API Implementation

This module provides the concrete implementation of LLMClient for
OpenAI's API (GPT-4, GPT-3.5, etc.)

Why Separate File:
    1. Single Responsibility: one provider per file
    2. Easy to swap: just change import
    3. Provider-specific handling: token limits, etc.

Author: Shubham Singh
Date: December 2025
"""

from loguru import logger

from clinical_note_generation.clients.llm_client import BaseLLMClient
from clinical_note_generation.core.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMContentFilteredError,
)


# =============================================================================
# STAGE 1: OPENAI CLIENT IMPLEMENTATION
# =============================================================================


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client for text generation.

    What it does:
        Provides text generation using OpenAI's models via
        the openai library.

    Why it exists:
        1. Encapsulates OpenAI-specific API logic
        2. Handles OpenAI's rate limiting
        3. Translates OpenAI errors to domain exceptions

    When to use:
        - When using OpenAI as the LLM provider
        - Alternative to Gemini

    Supported Models:
        - gpt-4o-mini (cost-effective, fast)
        - gpt-4o (high quality)
        - gpt-4-turbo (balanced)
        - gpt-3.5-turbo (legacy, cheap)

    Example:
        >>> client = OpenAIClient(api_key="...", model_name="gpt-4o-mini")
        >>> text = client.generate("Write a clinical note...")
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI client.

        STAGE 1.1: Initialize base class
        STAGE 1.2: Configure OpenAI SDK

        Args:
            api_key: OpenAI API key
            model_name: Model to use (default: gpt-4o-mini)
            rate_limit_delay: Seconds between API calls
            max_retries: Max retry attempts
        """
        # =====================================================================
        # STAGE 1.1: INITIALIZE BASE CLASS
        # =====================================================================
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )

        # =====================================================================
        # STAGE 1.2: CONFIGURE OPENAI SDK
        # =====================================================================
        self._client = None
        self._initialize_client()

        logger.info(f"OpenAIClient initialized | Model: {model_name}")

    def _initialize_client(self) -> None:
        """
        Initialize the OpenAI client.

        Lazy import to avoid requiring openai at module load.
        """
        try:
            from openai import OpenAI

            # Create client with API key
            self._client = OpenAI(api_key=self._api_key)

        except ImportError:
            raise LLMError(
                "openai package not installed. Install with: pip install openai",
                provider="openai",
            )
        except Exception as e:
            raise LLMError(
                f"Failed to initialize OpenAI client: {e}", provider="openai", original_error=e
            )

    # =========================================================================
    # STAGE 2: API CALL IMPLEMENTATION
    # =========================================================================

    def _call_api(self, prompt: str) -> str:
        """
        Make the actual OpenAI API call.

        Args:
            prompt: Generation prompt

        Returns:
            Generated text

        Raises:
            LLMError: If API call fails
            LLMRateLimitError: If rate limited
            LLMContentFilteredError: If content was filtered
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
            )

            # Extract text from response
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message.content:
                    return message.content

            raise LLMError("OpenAI returned empty response", provider="openai")

        except LLMError:
            # Re-raise our own exceptions
            raise

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limiting
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                raise LLMRateLimitError(provider="openai", original_error=e)

            # Check for content filtering
            if "content_filter" in error_str or "policy" in error_str:
                raise LLMContentFilteredError(provider="openai", reason=str(e))

            # Generic error
            raise LLMError(f"OpenAI API error: {e}", provider="openai", original_error=e)

    # =========================================================================
    # STAGE 3: PROPERTIES
    # =========================================================================

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openai"
