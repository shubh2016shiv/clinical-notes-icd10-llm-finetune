"""
Configuration for Clinical Note Generation Pipeline

This module defines the configuration dataclass used to initialize and
configure the clinical note generation pipeline. Configuration is:
    1. Loaded from environment variables (with .env support)
    2. Validated at startup to fail fast on misconfiguration
    3. Immutable after creation to prevent runtime changes

Configuration Hierarchy:
    PipelineConfiguration (main config)
    ├── LLM Settings (API keys, model names, rate limits)
    ├── Dataset Settings (file paths, caching)
    ├── Generation Settings (counts, profiles, retries)
    └── Validation Settings (thresholds, timeouts)

Usage:
    from clinical_note_generation.core.config import PipelineConfiguration

    # Load from environment
    config = PipelineConfiguration.from_environment()

    # Or configure programmatically
    config = PipelineConfiguration(
        gemini_api_key="your-key",
        icd10_dataset_path="path/to/dataset.json"
    )

Author: Shubham Singh
Date: December 2025
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from clinical_note_generation.core.enums import (
    SelectionStrategy,
    GenerationProfile,
)
from clinical_note_generation.core.exceptions import ConfigurationError


# =============================================================================
# STAGE 1: DEFAULT VALUES
# =============================================================================
# Centralized defaults make configuration transparent and overridable.


class ConfigDefaults:
    """Default configuration values."""

    # -------------------------------------------------------------------------
    # 1.1 LLM Provider Defaults
    # -------------------------------------------------------------------------
    DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
    DEFAULT_OPENAI_MODEL = "gpt-4"
    DEFAULT_RATE_LIMIT_DELAY = 0.5  # seconds between API calls
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 2.0  # seconds

    # -------------------------------------------------------------------------
    # 1.2 Generation Defaults
    # -------------------------------------------------------------------------
    DEFAULT_MIN_CODES = 2
    DEFAULT_MAX_CODES = 5
    DEFAULT_SELECTION_STRATEGY = SelectionStrategy.PROFILE_BASED
    DEFAULT_GENERATION_PROFILE = GenerationProfile.DEFAULT

    # -------------------------------------------------------------------------
    # 1.3 Validation Defaults
    # -------------------------------------------------------------------------
    DEFAULT_VALIDATION_TIMEOUT = 30.0  # seconds
    DEFAULT_CONFIDENCE_THRESHOLD = 70.0  # minimum acceptable confidence
    DEFAULT_MAX_VALIDATION_RETRIES = 2

    # -------------------------------------------------------------------------
    # 1.4 File Path Defaults
    # -------------------------------------------------------------------------
    DEFAULT_OUTPUT_DIR = "generated_clinical_notes"


# =============================================================================
# STAGE 2: CONFIGURATION DATACLASS
# =============================================================================
# The main configuration object for the pipeline.


@dataclass
class PipelineConfiguration:
    """
    Configuration for the clinical note generation pipeline.

    What it does:
        Encapsulates all configuration parameters needed to initialize
        and run the clinical note generation pipeline.

    Why it exists:
        1. Single source of truth for all configuration
        2. Validated at startup to fail fast on errors
        3. Supports both environment and programmatic configuration

    When to use:
        - At pipeline initialization
        - When creating test fixtures with custom config
        - When documenting available configuration options

    Attributes:
        See individual sections below for grouped attributes.

    Example:
        >>> config = PipelineConfiguration.from_environment()
        >>> config.validate()
        >>> print(config.gemini_model)
        'gemini-1.5-flash'
    """

    # -------------------------------------------------------------------------
    # 2.1 LLM Provider Configuration
    # -------------------------------------------------------------------------
    gemini_api_key: Optional[str] = None
    """Google Gemini API key. Required if using Gemini provider."""

    gemini_model: str = ConfigDefaults.DEFAULT_GEMINI_MODEL
    """Gemini model name (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')."""

    openai_api_key: Optional[str] = None
    """OpenAI API key. Required if using OpenAI provider."""

    openai_model: str = ConfigDefaults.DEFAULT_OPENAI_MODEL
    """OpenAI model name (e.g., 'gpt-4', 'gpt-4-turbo')."""

    llm_provider: str = "gemini"
    """Which LLM provider to use: 'gemini' or 'openai'."""

    rate_limit_delay: float = ConfigDefaults.DEFAULT_RATE_LIMIT_DELAY
    """Delay between API calls in seconds (rate limiting)."""

    # -------------------------------------------------------------------------
    # 2.2 Dataset Configuration
    # -------------------------------------------------------------------------
    icd10_dataset_path: Optional[str] = None
    """Path to ICD-10 dataset JSON file."""

    icd10_specificity_path: Optional[str] = None
    """Path to ICD-10 specificity dataset (for semantic selection)."""

    vector_index_path: Optional[str] = None
    """Path to vector index directory (for vector-based selection)."""

    use_api_for_codes: bool = False
    """If True, use NLM API for code lookup instead of file."""

    # -------------------------------------------------------------------------
    # 2.3 Generation Configuration
    # -------------------------------------------------------------------------
    min_codes_per_note: int = ConfigDefaults.DEFAULT_MIN_CODES
    """Minimum number of ICD-10 codes per generated note."""

    max_codes_per_note: int = ConfigDefaults.DEFAULT_MAX_CODES
    """Maximum number of ICD-10 codes per generated note."""

    selection_strategy: SelectionStrategy = ConfigDefaults.DEFAULT_SELECTION_STRATEGY
    """Strategy for selecting ICD-10 codes."""

    default_profile: GenerationProfile = ConfigDefaults.DEFAULT_GENERATION_PROFILE
    """Default patient profile for code selection."""

    max_retries: int = ConfigDefaults.DEFAULT_MAX_RETRIES
    """Maximum retry attempts for failed generations."""

    retry_delay: float = ConfigDefaults.DEFAULT_RETRY_DELAY
    """Delay between retry attempts in seconds."""

    # -------------------------------------------------------------------------
    # 2.4 Validation Configuration
    # -------------------------------------------------------------------------
    enable_validation: bool = True
    """Whether to validate generated notes."""

    validation_timeout: float = ConfigDefaults.DEFAULT_VALIDATION_TIMEOUT
    """Timeout for validation in seconds."""

    confidence_threshold: float = ConfigDefaults.DEFAULT_CONFIDENCE_THRESHOLD
    """Minimum confidence score to accept a note (0-100)."""

    skip_llm_validation_on_rule_failure: bool = True
    """Skip expensive LLM validation if rule-based validation fails."""

    # -------------------------------------------------------------------------
    # 2.5 Output Configuration
    # -------------------------------------------------------------------------
    output_directory: str = ConfigDefaults.DEFAULT_OUTPUT_DIR
    """Directory for saving generated notes."""

    # -------------------------------------------------------------------------
    # 2.6 Validation Methods
    # -------------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Checks:
            1. At least one LLM API key is configured
            2. Dataset path exists (if not using API)
            3. Numeric parameters are in valid ranges

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Check LLM API key
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ConfigurationError(
                "Gemini API key required when using Gemini provider",
                context={"setting": "GEMINI_API_KEY", "provider": "gemini"},
            )

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "OpenAI API key required when using OpenAI provider",
                context={"setting": "OPENAI_API_KEY", "provider": "openai"},
            )

        # Check dataset path (only if not using API)
        if not self.use_api_for_codes and self.icd10_dataset_path:
            dataset_path = Path(self.icd10_dataset_path)
            if not dataset_path.exists():
                raise ConfigurationError(
                    f"ICD-10 dataset file not found: {self.icd10_dataset_path}",
                    context={"setting": "ICD10_DATASET_PATH", "path": str(dataset_path)},
                )

        # Check numeric ranges
        if not (0 < self.min_codes_per_note <= self.max_codes_per_note):
            raise ConfigurationError(
                f"Invalid code range: min={self.min_codes_per_note}, max={self.max_codes_per_note}",
                context={"min": self.min_codes_per_note, "max": self.max_codes_per_note},
            )

        if not (0 <= self.confidence_threshold <= 100):
            raise ConfigurationError(
                f"Confidence threshold must be 0-100, got {self.confidence_threshold}",
                context={"threshold": self.confidence_threshold},
            )

    # -------------------------------------------------------------------------
    # 2.7 Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_environment(
        cls, env_file: Optional[str] = None, validate_on_load: bool = True
    ) -> "PipelineConfiguration":
        """
        Load configuration from environment variables.

        STAGE 1: Load .env file (if specified or found)
        STAGE 2: Read environment variables
        STAGE 3: Convert to typed configuration
        STAGE 4: Validate configuration (optional)

        Args:
            env_file: Path to .env file (optional, auto-detected if not provided)
            validate_on_load: Whether to validate after loading

        Returns:
            Configured PipelineConfiguration instance

        Raises:
            ConfigurationError: If required settings are missing or invalid

        Example:
            >>> config = PipelineConfiguration.from_environment()
            >>> print(config.llm_provider)
            'gemini'
        """
        # STAGE 1: Load .env file
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in common locations
            possible_locations = [
                Path.cwd() / ".env",
                Path.cwd() / "clinical_note_generation" / ".env",
                Path(__file__).parent.parent.parent / "data_generation" / ".env",
            ]
            for location in possible_locations:
                if location.exists():
                    load_dotenv(location)
                    break

        # STAGE 2: Read environment variables
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Determine LLM provider
        llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        if not gemini_key and openai_key:
            llm_provider = "openai"

        # Get dataset paths
        # Default to clinical_note_generation/data/ directory
        base_path = Path(__file__).parent.parent  # clinical_note_generation/
        default_dataset_path = base_path / "data" / "icd10_dataset.json"
        default_specificity_path = base_path / "data" / "icd10_specificity.json"

        # STAGE 3: Create configuration
        config = cls(
            # LLM settings
            gemini_api_key=gemini_key,
            gemini_model=os.getenv("GEMINI_MODEL", ConfigDefaults.DEFAULT_GEMINI_MODEL),
            openai_api_key=openai_key,
            openai_model=os.getenv("OPENAI_MODEL", ConfigDefaults.DEFAULT_OPENAI_MODEL),
            llm_provider=llm_provider,
            rate_limit_delay=float(
                os.getenv("RATE_LIMIT_DELAY", ConfigDefaults.DEFAULT_RATE_LIMIT_DELAY)
            ),
            # Dataset settings
            icd10_dataset_path=os.getenv("ICD10_DATASET_PATH", str(default_dataset_path)),
            icd10_specificity_path=os.getenv(
                "ICD10_SPECIFICITY_PATH", str(default_specificity_path)
            ),
            vector_index_path=os.getenv("VECTOR_INDEX_PATH"),
            use_api_for_codes=os.getenv("USE_API_FOR_CODES", "false").lower() == "true",
            # Generation settings
            min_codes_per_note=int(
                os.getenv("MIN_CODES_PER_NOTE", ConfigDefaults.DEFAULT_MIN_CODES)
            ),
            max_codes_per_note=int(
                os.getenv("MAX_CODES_PER_NOTE", ConfigDefaults.DEFAULT_MAX_CODES)
            ),
            max_retries=int(os.getenv("MAX_RETRIES", ConfigDefaults.DEFAULT_MAX_RETRIES)),
            retry_delay=float(os.getenv("RETRY_DELAY", ConfigDefaults.DEFAULT_RETRY_DELAY)),
            # Validation settings
            enable_validation=os.getenv("ENABLE_VALIDATION", "true").lower() == "true",
            validation_timeout=float(
                os.getenv("VALIDATION_TIMEOUT", ConfigDefaults.DEFAULT_VALIDATION_TIMEOUT)
            ),
            confidence_threshold=float(
                os.getenv("CONFIDENCE_THRESHOLD", ConfigDefaults.DEFAULT_CONFIDENCE_THRESHOLD)
            ),
            # Output settings
            output_directory=os.getenv("OUTPUT_DIRECTORY", ConfigDefaults.DEFAULT_OUTPUT_DIR),
        )

        # STAGE 4: Validate
        if validate_on_load:
            config.validate()

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary (for logging/debugging)."""
        return {
            "llm_provider": self.llm_provider,
            "gemini_model": self.gemini_model,
            "openai_model": self.openai_model,
            "gemini_api_key": "***" if self.gemini_api_key else None,
            "openai_api_key": "***" if self.openai_api_key else None,
            "icd10_dataset_path": self.icd10_dataset_path,
            "use_api_for_codes": self.use_api_for_codes,
            "min_codes_per_note": self.min_codes_per_note,
            "max_codes_per_note": self.max_codes_per_note,
            "selection_strategy": self.selection_strategy.value,
            "default_profile": self.default_profile.value,
            "enable_validation": self.enable_validation,
            "confidence_threshold": self.confidence_threshold,
            "output_directory": self.output_directory,
        }
