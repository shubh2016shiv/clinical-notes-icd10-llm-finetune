"""
Clinical Note Generation Pipeline - Main Orchestrator

This is the PUBLIC API entry point for the entire clinical note generation
system. It coordinates all layers (repository, selection, generation,
validation) into a simple, easy-to-use interface.

Architecture Diagram:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        ClinicalNotePipeline                         │
    │                         (This Orchestrator)                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐  │
    │   │Repository │ →  │ Selection │ →  │ Generation│ →  │ Validation│  │
    │   └───────────┘    └───────────┘    └───────────┘    └───────────┘  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Why Single Entry Point:
    1. Simple API: Users call one method to generate notes
    2. Encapsulation: Complex coordination hidden behind facade
    3. Configuration: Single place to configure entire pipeline
    4. Testability: Easy to mock for integration tests

Usage:
    from clinical_note_generation import ClinicalNotePipeline

    # Quick start - load configuration from environment
    pipeline = ClinicalNotePipeline.from_environment()

    # Generate notes
    notes = pipeline.generate_notes(
        count=10,
        profile="chronic_care"
    )

Author: Shubham Singh
Date: December 2025
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from clinical_note_generation.core.models import ICD10Code, ClinicalNote
from clinical_note_generation.core.enums import ClinicalNoteType
from clinical_note_generation.core.config import PipelineConfiguration
from clinical_note_generation.core.exceptions import ConfigurationError
from clinical_note_generation.repository import FileBasedICD10Repository
from clinical_note_generation.selection import CodeSelector
from clinical_note_generation.generation import NoteGenerator
from clinical_note_generation.validation import NoteValidator
from clinical_note_generation.clients import GeminiClient


# =============================================================================
# STAGE 1: PIPELINE CLASS
# =============================================================================


class ClinicalNotePipeline:
    """
    Main orchestrator for clinical note generation.

    What it does:
        Provides a single entry point for generating clinical notes
        with ICD-10 codes and validation. Coordinates repository,
        selection, generation, and validation layers.

    Why it exists:
        1. Simple API: One class to learn, one method to call
        2. Encapsulation: Hides complex coordination logic
        3. Configuration: Central place for all settings
        4. Extensibility: Easy to swap components

    When to use:
        - Always! This is the recommended way to use the module.

    How it works:
        STAGE 1: Initialize all components from configuration
        STAGE 2: On generate_notes():
            2.1 Select ICD-10 codes using CodeSelector
            2.2 Generate clinical notes using NoteGenerator
            2.3 Validate notes using NoteValidator
            2.4 Return validated notes

    Example:
        >>> pipeline = ClinicalNotePipeline.from_environment()
        >>> notes = pipeline.generate_notes(count=5, profile="chronic_care")
        >>> for note in notes:
        ...     print(f"{note.note_id}: {len(note.clinical_text)} chars")
    """

    def __init__(
        self,
        config: PipelineConfiguration,
        repository: Optional[FileBasedICD10Repository] = None,
        selector: Optional[CodeSelector] = None,
        generator: Optional[NoteGenerator] = None,
        validator: Optional[NoteValidator] = None,
    ):
        """
        Initialize pipeline with configuration and optional component overrides.

        Args:
            config: Pipeline configuration
            repository: Optional repository override (for testing)
            selector: Optional selector override (for testing)
            generator: Optional generator override (for testing)
            validator: Optional validator override (for testing)
        """
        # =====================================================================
        # STAGE 1.1: STORE CONFIGURATION
        # =====================================================================
        self._config = config

        # =====================================================================
        # STAGE 1.2: INITIALIZE REPOSITORY
        # =====================================================================
        if repository:
            self._repository = repository
        else:
            self._repository = self._create_repository(config)

        # =====================================================================
        # STAGE 1.3: INITIALIZE SELECTOR
        # =====================================================================
        if selector:
            self._selector = selector
        else:
            self._selector = CodeSelector(
                repository=self._repository, default_strategy=config.selection_strategy
            )

        # =====================================================================
        # STAGE 1.4: INITIALIZE LLM CLIENT
        # =====================================================================
        llm_client = self._create_llm_client(config)

        # =====================================================================
        # STAGE 1.5: INITIALIZE GENERATOR
        # =====================================================================
        if generator:
            self._generator = generator
        else:
            self._generator = NoteGenerator(
                llm_client=llm_client, max_retries=config.max_retries, sanitize_output=True
            )

        # =====================================================================
        # STAGE 1.6: INITIALIZE VALIDATOR
        # =====================================================================
        if validator:
            self._validator = validator
        else:
            self._validator = NoteValidator(
                llm_client=llm_client if config.enable_validation else None,
                use_llm_validation=False,  # Rule-based only by default
                confidence_threshold=config.confidence_threshold,
            )

        # =====================================================================
        # STAGE 1.7: TRACKING STATE
        # =====================================================================
        self._notes_generated = 0
        self._notes_valid = 0

        logger.info(
            f"ClinicalNotePipeline initialized | "
            f"Provider: {config.llm_provider} | "
            f"Model: {config.gemini_model if config.llm_provider == 'gemini' else config.openai_model}"
        )

    # =========================================================================
    # STAGE 2: MAIN GENERATION API
    # =========================================================================

    def generate_notes(
        self,
        count: int,
        profile: str = "default",
        note_type: ClinicalNoteType = ClinicalNoteType.PROGRESS_NOTE,
        codes_per_note: Optional[int] = None,
        validate: bool = True,
    ) -> List[ClinicalNote]:
        """
        Generate clinical notes with ICD-10 codes.

        This is the main entry point for note generation. It:
            1. Selects ICD-10 codes for each note
            2. Generates clinical note text
            3. Validates notes (if enabled)
            4. Returns list of ClinicalNote objects

        Args:
            count: Number of notes to generate
            profile: Generation profile (e.g., "chronic_care", "acute_injury")
            note_type: Type of clinical note to generate
            codes_per_note: Codes per note (uses config defaults if not specified)
            validate: Whether to validate generated notes

        Returns:
            List of generated ClinicalNote objects

        Example:
            >>> notes = pipeline.generate_notes(
            ...     count=10,
            ...     profile="chronic_care",
            ...     note_type=ClinicalNoteType.PROGRESS_NOTE
            ... )
        """
        # =====================================================================
        # STAGE 2.1: DETERMINE CODE COUNT
        # =====================================================================
        if codes_per_note is None:
            import random

            min_codes = self._config.min_codes_per_note
            max_codes = self._config.max_codes_per_note

        logger.info(f"Generating {count} {note_type.value} notes | " f"Profile: {profile}")

        generated_notes: List[ClinicalNote] = []

        # =====================================================================
        # STAGE 2.2: GENERATE EACH NOTE
        # =====================================================================
        for i in range(count):
            try:
                # Step 1: Determine code count for this note
                if codes_per_note:
                    num_codes = codes_per_note
                else:
                    import random

                    num_codes = random.randint(min_codes, max_codes)

                # Step 2: Select codes
                codes = self._selector.select(count=num_codes, profile=profile)

                # Step 3: Generate note
                note = self._generator.generate(codes=codes, note_type=note_type, profile=profile)

                # Step 4: Validate (if enabled)
                if validate:
                    note = self._validator.validate_and_update(note)
                    if note.is_valid:
                        self._notes_valid += 1

                generated_notes.append(note)
                self._notes_generated += 1

                logger.debug(
                    f"Generated note {i+1}/{count} | "
                    f"ID: {note.note_id} | "
                    f"Valid: {note.is_valid}"
                )

            except Exception as e:
                logger.error(f"Failed to generate note {i+1}: {e}")
                # Continue with remaining notes
                continue

        # =====================================================================
        # STAGE 2.3: SUMMARY
        # =====================================================================
        logger.info(
            f"Generation complete | "
            f"Generated: {len(generated_notes)}/{count} | "
            f"Valid: {sum(1 for n in generated_notes if n.is_valid)}"
        )

        return generated_notes

    def generate_single_note(
        self,
        codes: List[ICD10Code],
        note_type: ClinicalNoteType = ClinicalNoteType.PROGRESS_NOTE,
        profile: str = "default",
        validate: bool = True,
    ) -> ClinicalNote:
        """
        Generate a single note for specific ICD-10 codes.

        Args:
            codes: ICD-10 codes to use for this note
            note_type: Type of clinical note
            profile: Generation profile (for metadata)
            validate: Whether to validate the note

        Returns:
            Generated ClinicalNote
        """
        note = self._generator.generate(codes=codes, note_type=note_type, profile=profile)

        if validate:
            note = self._validator.validate_and_update(note)

        self._notes_generated += 1
        if note.is_valid:
            self._notes_valid += 1

        return note

    # =========================================================================
    # STAGE 3: SAVE NOTES
    # =========================================================================

    def save_notes(
        self,
        notes: List[ClinicalNote],
        output_dir: Optional[str] = None,
        filename_prefix: str = "clinical_notes",
    ) -> str:
        """
        Save generated notes to JSON file.

        Args:
            notes: Notes to save
            output_dir: Directory to save to (uses config default if not specified)
            filename_prefix: Prefix for output filename

        Returns:
            Path to saved file
        """
        # Determine output directory
        dir_path = Path(output_dir or self._config.output_directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = dir_path / filename

        # Prepare data
        output_data = [note.to_dict() for note in notes]

        # Save
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(notes)} notes to {filepath}")
        return str(filepath)

    # =========================================================================
    # STAGE 4: FACTORY METHODS
    # =========================================================================

    @classmethod
    def from_environment(cls, env_file: Optional[str] = None) -> "ClinicalNotePipeline":
        """
        Create pipeline from environment configuration.

        This is the recommended way to create a pipeline. It:
            1. Loads configuration from .env file
            2. Validates required settings
            3. Initializes all components

        Args:
            env_file: Path to .env file (optional)

        Returns:
            Configured ClinicalNotePipeline

        Raises:
            ConfigurationError: If required settings missing

        Example:
            >>> pipeline = ClinicalNotePipeline.from_environment()
        """
        config = PipelineConfiguration.from_environment(env_file=env_file, validate_on_load=True)
        return cls(config)

    # =========================================================================
    # STAGE 5: PRIVATE HELPERS
    # =========================================================================

    def _create_repository(self, config: PipelineConfiguration) -> FileBasedICD10Repository:
        """Create repository from configuration."""
        if not config.icd10_dataset_path:
            raise ConfigurationError(
                "ICD-10 dataset path not configured", context={"setting": "icd10_dataset_path"}
            )
        return FileBasedICD10Repository(config.icd10_dataset_path)

    def _create_llm_client(self, config: PipelineConfiguration):
        """Create LLM client from configuration."""
        if config.llm_provider == "gemini":
            if not config.gemini_api_key:
                raise ConfigurationError(
                    "Gemini API key required", context={"setting": "GEMINI_API_KEY"}
                )
            return GeminiClient(
                api_key=config.gemini_api_key,
                model_name=config.gemini_model,
                rate_limit_delay=config.rate_limit_delay,
                max_retries=config.max_retries,
            )
        else:
            raise ConfigurationError(
                f"Unsupported LLM provider: {config.llm_provider}",
                context={"supported": ["gemini"]},
            )

    # =========================================================================
    # STAGE 6: PROPERTIES AND METRICS
    # =========================================================================

    @property
    def notes_generated(self) -> int:
        """Total notes generated."""
        return self._notes_generated

    @property
    def notes_valid(self) -> int:
        """Number of valid notes generated."""
        return self._notes_valid

    @property
    def validation_rate(self) -> float:
        """Percentage of notes that passed validation."""
        if self._notes_generated == 0:
            return 0.0
        return (self._notes_valid / self._notes_generated) * 100

    @property
    def config(self) -> PipelineConfiguration:
        """Access to pipeline configuration."""
        return self._config

    @property
    def repository(self) -> FileBasedICD10Repository:
        """Access to ICD-10 repository."""
        return self._repository

    @property
    def selector(self) -> CodeSelector:
        """Access to code selector."""
        return self._selector


# =============================================================================
# STAGE 7: SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    # Configure logger for simple output
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("\n--- Clinical Note Generation Pipeline Smoke Test ---\n")

    try:
        # 1. Initialize Pipeline
        print("1. Initializing pipeline from environment...")
        pipeline = ClinicalNotePipeline.from_environment()
        print("   [OK] Pipeline initialized successfully")

        # 2. Inspect Configuration
        config = pipeline.config
        print("\n2. Configuration loaded:")
        print(f"   - LLM Provider: {config.llm_provider}")
        print(
            f"   - Model: {config.gemini_model if config.llm_provider == 'gemini' else config.openai_model}"
        )
        print(f"   - Dataset: {config.icd10_dataset_path}")
        print(f"   - Output Dir: {config.output_directory}")
        print(f"   - Validation Enabled: {config.enable_validation}")

        # 3. Inspect Repository
        print("\n3. Repository status:")
        print(f"   - Total codes: {pipeline.repository.total_codes:,}")
        print(f"   - Billable codes: {pipeline.repository.billable_code_count:,}")
        print(f"   - Chapters: {len(pipeline.repository.chapters)}")
        print("   [OK] Repository loaded successfully")

        # 4. Test Code Selection
        print("\n4. Testing code selection (Profile: chronic_care)...")
        codes = pipeline.selector.select(count=3, profile="chronic_care")
        print(f"   - Selected {len(codes)} codes:")
        for code in codes:
            print(f"     * {code.code}: {code.name}")
        print("   [OK] Selection working")

        print("\n[OK] SMOKE TEST PASSED: System is ready for generation.")

    except Exception as e:
        print(f"\n[FAIL] SMOKE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
