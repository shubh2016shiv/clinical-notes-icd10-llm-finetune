"""
Gemini Client - Google Gemini API Implementation

This module provides the concrete implementation of LLMClient for
Google's Gemini API (gemini-1.5-flash, gemini-1.5-pro, etc.)

Why Separate File:
    1. Single Responsibility: one provider per file
    2. Easy to swap: just change import
    3. Provider-specific handling: safety settings, etc.

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
# STAGE 1: GEMINI CLIENT IMPLEMENTATION
# =============================================================================


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client for text generation.

    What it does:
        Provides text generation using Google's Gemini models via
        the google-generativeai library.

    Why it exists:
        1. Encapsulates Gemini-specific API logic
        2. Handles Gemini's safety settings
        3. Translates Gemini errors to domain exceptions

    When to use:
        - When using Gemini as the LLM provider
        - Default provider in the pipeline

    Supported Models:
        - gemini-1.5-flash (fast, cost-effective)
        - gemini-1.5-pro (higher quality)
        - gemini-1.0-pro (legacy)

    Example:
        >>> client = GeminiClient(api_key="...", model_name="gemini-1.5-flash")
        >>> text = client.generate("Write a clinical note...")
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ):
        """
        Initialize Gemini client.

        STAGE 1.1: Initialize base class
        STAGE 1.2: Configure Gemini SDK
        STAGE 1.3: Set safety settings

        Args:
            api_key: Google API key (Gemini)
            model_name: Model to use (default: gemini-1.5-flash)
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
        # STAGE 1.2: CONFIGURE GEMINI SDK
        # =====================================================================
        self._model = None
        self._initialize_client()

        logger.info(f"GeminiClient initialized | Model: {model_name}")

    def _initialize_client(self) -> None:
        """
        Initialize the Gemini client and model.

        Lazy import to avoid requiring google-generativeai at module load.
        """
        try:
            import google.generativeai as genai

            # Configure API key
            genai.configure(api_key=self._api_key)

            # Create model with safety settings for medical content
            # STAGE 1.3: Set safety settings (permissive for medical content)
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                safety_settings=safety_settings,
            )

        except ImportError:
            raise LLMError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai",
                provider="gemini",
            )
        except Exception as e:
            raise LLMError(
                f"Failed to initialize Gemini client: {e}", provider="gemini", original_error=e
            )

    # =========================================================================
    # STAGE 2: API CALL IMPLEMENTATION
    # =========================================================================

    def _call_api(self, prompt: str) -> str:
        """
        Make the actual Gemini API call.

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
            response = self._model.generate_content(prompt)

            # Check for blocked content
            if response.prompt_feedback.block_reason:
                raise LLMContentFilteredError(
                    provider="gemini", reason=str(response.prompt_feedback.block_reason)
                )

            # Extract text from response
            if response.text:
                return response.text

            # Handle empty response
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text

            raise LLMError("Gemini returned empty response", provider="gemini")

        except LLMError:
            # Re-raise our own exceptions
            raise

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limiting
            if "rate" in error_str or "quota" in error_str:
                raise LLMRateLimitError(provider="gemini", original_error=e)

            # Check for content filtering
            if "blocked" in error_str or "safety" in error_str:
                raise LLMContentFilteredError(provider="gemini", reason=str(e))

            # Generic error
            raise LLMError(f"Gemini API error: {e}", provider="gemini", original_error=e)

    # =========================================================================
    # STAGE 3: PROPERTIES
    # =========================================================================

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "gemini"
