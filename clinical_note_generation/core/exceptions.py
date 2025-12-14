"""
Domain Exceptions for Clinical Note Generation

This module defines all custom exceptions used throughout the clinical note
generation pipeline. Well-defined exceptions enable:
    1. Clear error categorization for debugging
    2. Specific catch blocks for different failure modes
    3. Rich error context for troubleshooting

Exception Hierarchy:
    ClinicalNoteGenerationError (base)
    ├── ConfigurationError      → Invalid configuration
    ├── ICD10CodeError          → Code-related errors
    │   ├── CodeNotFoundError
    │   ├── InvalidCodeFormatError
    │   └── InsufficientCodesError
    ├── GenerationError         → Note generation failures
    │   ├── PromptError
    │   └── LLMError
    └── ValidationError         → Validation failures

Usage:
    from clinical_note_generation.core.exceptions import CodeNotFoundError

    try:
        code = repository.get_code("INVALID")
    except CodeNotFoundError as e:
        logger.error(f"Code not found: {e.code}")

Author: Shubham Singh
Date: December 2025
"""

from typing import Optional


# =============================================================================
# STAGE 1: BASE EXCEPTION
# =============================================================================
# All domain exceptions inherit from this base class.


class ClinicalNoteGenerationError(Exception):
    """
    Base exception for all clinical note generation errors.

    What it does:
        Provides a common base class for all domain-specific exceptions,
        enabling catch-all handling while preserving specific error types.

    Why it exists:
        1. Enables `except ClinicalNoteGenerationError` to catch all domain errors
        2. Provides consistent error context structure
        3. Separates domain errors from system errors

    Attributes:
        message: Human-readable error description
        context: Dictionary of additional context for debugging
    """

    def __init__(self, message: str, context: Optional[dict] = None):
        """
        Initialize with message and optional context.

        Args:
            message: Human-readable error description
            context: Additional debugging context (code, stage, inputs)
        """
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format message with context for display."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


# =============================================================================
# STAGE 2: CONFIGURATION ERRORS
# =============================================================================
# Errors related to invalid or missing configuration.


class ConfigurationError(ClinicalNoteGenerationError):
    """
    Error in pipeline configuration.

    What it does:
        Indicates that the pipeline configuration is invalid, incomplete,
        or internally inconsistent.

    When raised:
        - Missing required API keys
        - Invalid file paths
        - Conflicting configuration options
        - Missing environment variables

    Example:
        >>> raise ConfigurationError(
        ...     "API key not configured",
        ...     context={"setting": "GEMINI_API_KEY", "source": "environment"}
        ... )
    """

    pass


# =============================================================================
# STAGE 3: ICD-10 CODE ERRORS
# =============================================================================
# Errors related to ICD-10 code operations.


class ICD10CodeError(ClinicalNoteGenerationError):
    """
    Base exception for ICD-10 code-related errors.

    What it does:
        Parent class for all errors related to ICD-10 code lookup,
        validation, and selection operations.
    """

    pass


class CodeNotFoundError(ICD10CodeError):
    """
    ICD-10 code does not exist in the repository.

    What it does:
        Indicates that a requested ICD-10 code could not be found
        in the code repository (file or API).

    When raised:
        - Looking up a code that doesn't exist
        - Typo in code format
        - Code removed in newer ICD-10 version

    Attributes:
        code: The code that was not found
    """

    def __init__(self, code: str, message: Optional[str] = None):
        self.code = code
        super().__init__(message or f"ICD-10 code not found: {code}", context={"code": code})


class InvalidCodeFormatError(ICD10CodeError):
    """
    ICD-10 code has invalid format.

    What it does:
        Indicates that a code string does not match valid ICD-10 format
        (e.g., missing characters, wrong structure).

    When raised:
        - Code doesn't match ICD-10 pattern
        - Missing required 7th character for injury codes
        - Invalid code length

    Attributes:
        code: The invalid code
        reason: Why the format is invalid
    """

    def __init__(self, code: str, reason: str):
        self.code = code
        self.reason = reason
        super().__init__(
            f"Invalid ICD-10 code format: {code} - {reason}",
            context={"code": code, "reason": reason},
        )


class InsufficientCodesError(ICD10CodeError):
    """
    Not enough codes available to fulfill request.

    What it does:
        Indicates that the requested number of codes could not be
        selected due to insufficient availability.

    When raised:
        - Requested 10 codes but only 5 match criteria
        - Category has too few billable codes
        - Filters are too restrictive

    Attributes:
        requested: Number of codes requested
        available: Number of codes available
        criteria: Selection criteria that were applied
    """

    def __init__(self, requested: int, available: int, criteria: Optional[str] = None):
        self.requested = requested
        self.available = available
        self.criteria = criteria
        super().__init__(
            f"Requested {requested} codes but only {available} available",
            context={
                "requested": requested,
                "available": available,
                "criteria": criteria or "none",
            },
        )


# =============================================================================
# STAGE 4: GENERATION ERRORS
# =============================================================================
# Errors related to clinical note generation.


class GenerationError(ClinicalNoteGenerationError):
    """
    Base exception for note generation errors.

    What it does:
        Parent class for all errors that occur during the clinical
        note generation process.
    """

    pass


class PromptError(GenerationError):
    """
    Error constructing or processing generation prompt.

    What it does:
        Indicates a problem with prompt construction or formatting
        before it's sent to the LLM.

    When raised:
        - Template rendering fails
        - Invalid prompt parameters
        - Prompt exceeds token limit
    """

    pass


class LLMError(GenerationError):
    """
    Error from LLM API call.

    What it does:
        Wraps errors from the underlying LLM API (Gemini, OpenAI, etc.)
        with additional context about the generation attempt.

    When raised:
        - API rate limit exceeded
        - API timeout
        - Invalid API response
        - Content filtered by safety settings

    Attributes:
        provider: The LLM provider (gemini, openai)
        original_error: The wrapped original exception
    """

    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(
            message,
            context={
                "provider": provider,
                "original_error": str(original_error) if original_error else None,
            },
        )


class LLMRateLimitError(LLMError):
    """
    LLM API rate limit exceeded.

    What it does:
        Specific error for rate limiting, enabling automatic retry
        with backoff.

    Attributes:
        retry_after: Seconds to wait before retrying (if known)
    """

    def __init__(
        self,
        provider: str,
        retry_after: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {provider}", provider=provider, original_error=original_error
        )
        self.context["retry_after"] = retry_after


class LLMContentFilteredError(LLMError):
    """
    LLM response was filtered due to safety settings.

    What it does:
        Indicates that the LLM refused to generate content due to
        safety filters, requiring prompt adjustment or regeneration.
    """

    def __init__(self, provider: str, reason: Optional[str] = None):
        super().__init__(
            f"Content filtered by {provider} safety settings: {reason or 'unknown reason'}",
            provider=provider,
        )
        self.reason = reason


# =============================================================================
# STAGE 5: VALIDATION ERRORS
# =============================================================================
# Errors related to clinical note validation.


class ValidationError(ClinicalNoteGenerationError):
    """
    Error during clinical note validation.

    What it does:
        Indicates a problem with the validation process itself,
        not that the note is invalid (that's a ValidationResult).

    When raised:
        - Validation API call fails
        - Unable to parse validation response
        - Validation timeout
    """

    pass


class ValidationTimeoutError(ValidationError):
    """
    Validation process timed out.

    What it does:
        Indicates that LLM-based validation took too long and was
        cancelled to prevent blocking.

    Attributes:
        timeout_seconds: The timeout that was exceeded
    """

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Validation timed out after {timeout_seconds}s",
            context={"timeout_seconds": timeout_seconds},
        )


# =============================================================================
# STAGE 6: REPOSITORY ERRORS
# =============================================================================
# Errors related to data repository operations.


class RepositoryError(ClinicalNoteGenerationError):
    """
    Error accessing data repository.

    What it does:
        Indicates a problem with the underlying data storage
        (file system, API, database).
    """

    pass


class DatasetLoadError(RepositoryError):
    """
    Error loading ICD-10 dataset.

    What it does:
        Indicates that the ICD-10 dataset file could not be loaded.

    When raised:
        - File not found
        - Invalid JSON format
        - Permission denied

    Attributes:
        file_path: Path to the dataset file
    """

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(
            f"Failed to load dataset from {file_path}: {reason}",
            context={"file_path": file_path, "reason": reason},
        )
