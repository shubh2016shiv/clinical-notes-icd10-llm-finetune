"""
Configuration Module for Clinical Notes Generation

This module handles configuration loading and management for the clinical notes
generation system, including environment variable loading and validation.

Author: Shubham Singh
Date: November 2025
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from .enums import APIProvider


@dataclass
class ClinicalNotesGeneratorConfiguration:
    """
    Configuration dataclass for clinical notes generation system.
    
    This class encapsulates all configuration parameters required for generating
    clinical notes including API keys, file paths, and generation settings. It
    handles environment variable loading and validation.
    
    Attributes:
        api_provider: Which API provider to use (GEMINI or OPENAI)
        gemini_api_key: Google Gemini API key for clinical note generation
        openai_api_key: OpenAI API key for clinical note generation
        icd10_dataset_file_path: Path to JSON file containing ICD-10 codes
        output_directory: Directory path where generated notes will be saved
        gemini_model_name: Name of the Gemini model to use
        openai_model_name: Name of the OpenAI model to use
        generation_temperature: Temperature setting for text generation (0.0-1.0)
        generation_top_p: Top-p sampling parameter for generation
        generation_top_k: Top-k sampling parameter for generation
        max_output_tokens: Maximum number of tokens in generated output
        enable_logging: Flag to enable or disable detailed logging
        log_file_path: Path to the log file for tracking generation progress
    
    Example:
        >>> config = ClinicalNotesGeneratorConfiguration.load_from_environment()
        >>> print(config.api_provider)
        APIProvider.GEMINI
    """
    
    # Step 1: API Provider Selection
    api_provider: APIProvider = APIProvider.GEMINI

    # Step 2: API Keys
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Step 3: File Path Configuration
    icd10_dataset_file_path: str = "data_generation/icd10_dataset.json"
    output_directory: str = "data_generation/generated_clinical_notes"

    # Step 4: Model Configuration
    gemini_model_name: str = "gemini-2.5-flash"
    openai_model_name: str = "gpt-4o"
    generation_temperature: float = 0.01
    generation_top_p: float = 0.95
    generation_top_k: int = 40
    max_output_tokens: int = 8192

    # Step 5: Validation Configuration
    validation_confidence_threshold: float = 95.0
    validation_max_retries: int = 2

    # Step 6: Logging Configuration
    enable_logging: bool = True
    log_file_path: str = "data_generation/clinical_notes_generation.log"
    
    @staticmethod
    def load_from_environment(
        dotenv_file_path: Optional[str] = None
    ) -> "ClinicalNotesGeneratorConfiguration":
        """
        Load configuration from environment variables with .env file support.
        
        Step 1: Determine .env file location
        Step 1.1: Use provided path or default to data_generation/.env
        Step 1.2: Log the .env file path being used
        
        Step 2: Load environment variables from .env file
        Step 2.1: Load .env file using python-dotenv
        Step 2.2: Log success or failure of .env loading
        
        Step 3: Extract API key from environment
        Step 3.1: Read GEMINI_API_KEY from environment variables
        Step 3.2: Validate that API key is present and not a placeholder
        
        Step 4: Create configuration object with loaded values
        Step 4.1: Initialize configuration with API key
        Step 4.2: Use default values for other settings (can be overridden)
        
        Step 5: Log configuration summary
        
        Args:
            dotenv_file_path: Optional path to .env file (defaults to data_generation/.env)
            
        Returns:
            ClinicalNotesGeneratorConfiguration: Loaded configuration object
            
        Raises:
            ValueError: If GEMINI_API_KEY is not found or is a placeholder
            FileNotFoundError: If specified .env file doesn't exist
            
        Example:
            >>> config = ClinicalNotesGeneratorConfiguration.load_from_environment()
            >>> print(config.gemini_api_key[:10])
            'AIzaSyB...'
        """
        # Step 1.1: Determine .env file location
        if dotenv_file_path is None:
            # Default to data_generation/.env
            dotenv_file_path = os.path.join("data_generation", ".env")
        
        dotenv_path = Path(dotenv_file_path)
        
        # Step 1.2: Log the .env file path
        logger.info(f"Loading environment variables from: {dotenv_path.absolute()}")
        
        # Step 2.1: Load environment variables from .env file
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=True)
            logger.info(f"Successfully loaded .env file from: {dotenv_path}")
        else:
            logger.warning(
                f".env file not found at: {dotenv_path.absolute()}. "
                f"Attempting to load from system environment variables."
            )
        
        # Step 3: Extract API provider preference
        api_provider_str = os.getenv("API_PROVIDER", "GEMINI").upper()
        try:
            api_provider = APIProvider[api_provider_str]
        except KeyError:
            available_providers = [p.name for p in APIProvider]
            error_message = (
                f"Invalid API_PROVIDER: '{api_provider_str}'. "
                f"Available providers: {', '.join(available_providers)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        # Step 4: Extract API keys based on provider
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Step 5: Validate required API key for selected provider
        if api_provider == APIProvider.GEMINI:
            if not gemini_api_key:
                error_message = (
                    "GEMINI_API_KEY not found in environment variables. "
                    "Please add to your .env file: GEMINI_API_KEY=your_gemini_key"
                )
                logger.error(error_message)
                raise ValueError(error_message)
            selected_api_key = gemini_api_key
        elif api_provider == APIProvider.OPENAI:
            if not openai_api_key:
                error_message = (
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please add to your .env file: OPENAI_API_KEY=your_openai_key"
                )
                logger.error(error_message)
                raise ValueError(error_message)
            selected_api_key = openai_api_key

        # Step 6: Validate API key is not a placeholder
        placeholder_values = [
            "your_api_key_here",
            "YOUR_GEMINI_API_KEY_HERE",
            "YOUR_OPENAI_API_KEY_HERE",
            "YOUR_API_KEY_HERE",
            "REPLACE_WITH_YOUR_API_KEY"
        ]

        if selected_api_key in placeholder_values:
            error_message = (
                f"API key appears to be a placeholder value: '{selected_api_key}'. "
                f"Please replace it with your actual API key."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        
        # Step 7: Extract validation configuration from environment
        validation_confidence_threshold = float(
            os.getenv("VALIDATION_CONFIDENCE_THRESHOLD", "95.0")
        )
        validation_max_retries = int(
            os.getenv("VALIDATION_MAX_RETRIES", "2")
        )
        
        # Step 8: Create configuration object
        configuration = ClinicalNotesGeneratorConfiguration(
            api_provider=api_provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            validation_confidence_threshold=validation_confidence_threshold,
            validation_max_retries=validation_max_retries
        )
        
        # Step 9: Log configuration summary (without exposing full API keys)
        masked_api_key = selected_api_key[:8] + "..." + selected_api_key[-4:] if len(selected_api_key) > 12 else "***"
        logger.info("=" * 80)
        logger.info("CLINICAL NOTES GENERATOR CONFIGURATION LOADED")
        logger.info("=" * 80)
        logger.info(f"API Provider: {api_provider.value}")
        logger.info(f"API Key: {masked_api_key}")

        if api_provider == APIProvider.GEMINI:
            logger.info(f"Model: {configuration.gemini_model_name}")
        elif api_provider == APIProvider.OPENAI:
            logger.info(f"Model: {configuration.openai_model_name}")

        logger.info(f"ICD-10 Dataset: {configuration.icd10_dataset_file_path}")
        logger.info(f"Output Directory: {configuration.output_directory}")
        logger.info(f"Validation Confidence Threshold: {configuration.validation_confidence_threshold}%")
        logger.info(f"Validation Max Retries: {configuration.validation_max_retries}")
        logger.info(f"Log File: {configuration.log_file_path}")
        logger.info("=" * 80)
        
        return configuration
    
    def validate(self) -> None:
        """
        Validate configuration parameters to ensure they are properly set.
        
        Step 1: Validate API key is present
        Step 2: Validate file paths are properly formatted
        Step 3: Validate generation parameters are in valid ranges
        Step 4: Log validation success
        
        Raises:
            ValueError: If any configuration parameter is invalid
            
        Example:
            >>> config = ClinicalNotesGeneratorConfiguration(gemini_api_key="test_key")
            >>> config.validate()
        """
        # Step 1: Validate API key based on provider
        if self.api_provider == APIProvider.GEMINI:
            if not self.gemini_api_key or len(self.gemini_api_key) < 10:
                raise ValueError("Invalid gemini_api_key: must be a valid API key string")
        elif self.api_provider == APIProvider.OPENAI:
            if not self.openai_api_key or len(self.openai_api_key) < 10:
                raise ValueError("Invalid openai_api_key: must be a valid API key string")
        
        # Step 2: Validate file paths
        if not self.icd10_dataset_file_path:
            raise ValueError("Invalid icd10_dataset_file_path: must not be empty")
        
        if not self.output_directory:
            raise ValueError("Invalid output_directory: must not be empty")
        
        # Step 3: Validate generation parameters
        if not 0.0 <= self.generation_temperature <= 1.0:
            raise ValueError("Invalid generation_temperature: must be between 0.0 and 1.0")
        
        if not 0.0 <= self.generation_top_p <= 1.0:
            raise ValueError("Invalid generation_top_p: must be between 0.0 and 1.0")
        
        if self.generation_top_k < 1:
            raise ValueError("Invalid generation_top_k: must be at least 1")
        
        if self.max_output_tokens < 100:
            raise ValueError("Invalid max_output_tokens: must be at least 100")
        
        # Step 4: Validate validation configuration parameters
        if not 0.0 <= self.validation_confidence_threshold <= 100.0:
            raise ValueError("Invalid validation_confidence_threshold: must be between 0.0 and 100.0")
        
        if self.validation_max_retries < 0:
            raise ValueError("Invalid validation_max_retries: must be non-negative")
        
        # Step 5: Log validation success
        logger.debug("Configuration validation successful")

