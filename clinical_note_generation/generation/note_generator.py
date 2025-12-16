"""
Note Generator - Clinical Note Generation with LLM

This module provides the core note generation functionality,
using LLM APIs to create clinical notes from ICD-10 codes.

Why Separate from Prompt Builder:
    1. Single Responsibility: this class handles LLM interaction
    2. Dependency injection: LLM client is injected
    3. Error handling: centralized retry logic
    4. Post-processing: sanitization after generation

Pipeline Position:
    Config → Repository → Selection → PromptBuilder → [NoteGenerator]
                                                       ^^^^^^^^^^^^^^
                                                       You are here

Author: Shubham Singh
Date: December 2025
"""

import re
import uuid
from datetime import datetime
from typing import List, Protocol, runtime_checkable

from loguru import logger

from clinical_note_generation.core.models import ICD10Code, ClinicalNote
from clinical_note_generation.core.enums import ClinicalNoteType
from clinical_note_generation.core.exceptions import GenerationError, LLMError
from clinical_note_generation.generation.prompt_builder import PromptBuilder


# =============================================================================
# STAGE 1: LLM CLIENT PROTOCOL
# =============================================================================
# Defines what an LLM client must provide. This allows dependency injection
# of different LLM providers (Gemini, OpenAI, etc.)


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol defining the interface for LLM clients.

    What it does:
        Specifies the contract that any LLM client must implement,
        enabling the generator to work with different providers.

    Why it exists:
        1. Decouples generation logic from specific LLM provider
        2. Enables mocking for tests
        3. Supports provider switching via configuration
    """

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The generation prompt

        Returns:
            Generated text

        Raises:
            LLMError: If generation fails
        """
        ...


# =============================================================================
# STAGE 2: NOTE GENERATOR CLASS
# =============================================================================
# Main generator that orchestrates note creation.


class NoteGenerator:
    """
    Generates clinical notes using LLM with ICD-10 code guidance.

    What it does:
        Takes ICD-10 codes and note type, produces realistic clinical
        documentation using LLM generation with post-processing.

    Why it exists:
        1. Encapsulates LLM interaction for note generation
        2. Handles post-processing (sanitization, formatting)
        3. Provides retry logic for transient failures
        4. Produces ClinicalNote objects ready for validation

    When to use:
        - After selecting ICD-10 codes with CodeSelector
        - When generating training data for model fine-tuning

    How it works:
        STAGE 2.1: Build prompt using PromptBuilder
        STAGE 2.2: Call LLM to generate raw text
        STAGE 2.3: Sanitize text (remove any leaked codes)
        STAGE 2.4: Create ClinicalNote object with metadata

    Example:
        >>> generator = NoteGenerator(llm_client)
        >>> note = generator.generate(
        ...     codes=[diabetes_code, htn_code],
        ...     note_type=ClinicalNoteType.PROGRESS_NOTE,
        ...     profile="chronic_care"
        ... )
        >>> print(note.clinical_text)
    """

    def __init__(self, llm_client: LLMClient, max_retries: int = 3, sanitize_output: bool = True):
        """
        Initialize the note generator.

        Args:
            llm_client: LLM client for text generation
            max_retries: Maximum retry attempts for failed generations
            sanitize_output: Whether to remove ICD-10 codes from output
        """
        # =====================================================================
        # STAGE 2.1: STORE DEPENDENCIES
        # =====================================================================
        self._llm_client = llm_client
        self._prompt_builder = PromptBuilder()
        self._max_retries = max_retries
        self._sanitize_output = sanitize_output

        # Track generation statistics
        self._generation_count = 0
        self._sanitization_count = 0

        logger.debug(
            f"NoteGenerator initialized | "
            f"Max retries: {max_retries} | "
            f"Sanitize: {sanitize_output}"
        )

    # =========================================================================
    # STAGE 3: MAIN GENERATION API
    # =========================================================================

    def generate(
        self,
        codes: List[ICD10Code],
        note_type: ClinicalNoteType = ClinicalNoteType.PROGRESS_NOTE,
        profile: str = "default",
        additional_context: str = "",
    ) -> ClinicalNote:
        """
        Generate a clinical note for given ICD-10 codes.

        Algorithm:
            1. Build generation prompt
            2. Call LLM with retry logic
            3. Sanitize output if enabled
            4. Create ClinicalNote object with metadata

        Args:
            codes: ICD-10 codes to incorporate in the note
            note_type: Type of clinical note to generate
            profile: Generation profile used (for metadata)
            additional_context: Optional extra guidance for generation

        Returns:
            ClinicalNote object with generated text and metadata

        Raises:
            GenerationError: If generation fails after all retries
        """
        # =====================================================================
        # STAGE 3.1: BUILD PROMPT
        # =====================================================================
        logger.info(
            f"Generating {note_type.value} | "
            f"Codes: {[c.code for c in codes]} | "
            f"Profile: {profile}"
        )

        prompt = self._prompt_builder.build_generation_prompt(
            note_type=note_type, codes=codes, additional_context=additional_context
        )

        # =====================================================================
        # STAGE 3.2: CALL LLM WITH RETRY
        # =====================================================================
        raw_text = self._call_llm_with_retry(prompt)

        # =====================================================================
        # STAGE 3.3: SANITIZE OUTPUT
        # =====================================================================
        if self._sanitize_output:
            clinical_text, codes_removed = self._sanitize_text(raw_text, codes)
            if codes_removed:
                self._sanitization_count += 1
                logger.warning(
                    f"Removed {codes_removed} code(s) from generated text " f"(auto-sanitization)"
                )
        else:
            clinical_text = raw_text

        # =====================================================================
        # STAGE 3.4: CREATE CLINICAL NOTE OBJECT
        # =====================================================================
        self._generation_count += 1

        note = ClinicalNote(
            note_id=self._generate_note_id(),
            clinical_text=clinical_text,
            note_type=note_type.value,
            assigned_codes=codes,
            validation_result=None,  # To be set by validator
            generation_profile=profile,
            generated_at=datetime.now(),
            generation_model=self._get_model_name(),
        )

        logger.info(
            f"Generated note {note.note_id} | "
            f"Length: {len(clinical_text)} chars | "
            f"Codes: {note.code_count}"
        )

        return note

    # =========================================================================
    # STAGE 4: LLM INTERACTION
    # =========================================================================

    def _call_llm_with_retry(self, prompt: str) -> str:
        """
        Call LLM with retry logic for transient failures.

        Args:
            prompt: Generation prompt

        Returns:
            Generated text

        Raises:
            GenerationError: After all retries exhausted
        """
        last_error = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(f"LLM call attempt {attempt}/{self._max_retries}")
                result = self._llm_client.generate(prompt)

                if not result or len(result.strip()) < 50:
                    raise GenerationError(
                        "LLM returned empty or too short response",
                        context={"response_length": len(result) if result else 0},
                    )

                return result

            except LLMError as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt}/{self._max_retries}): {e}")

            except Exception as e:
                last_error = GenerationError(
                    f"Unexpected error during generation: {e}", context={"original_error": str(e)}
                )
                logger.error(f"Unexpected generation error: {e}")

        # All retries exhausted
        raise GenerationError(
            f"Generation failed after {self._max_retries} attempts",
            context={"last_error": str(last_error)},
        )

    # =========================================================================
    # STAGE 5: POST-PROCESSING / SANITIZATION
    # =========================================================================

    def _sanitize_text(self, text: str, codes: List[ICD10Code]) -> tuple[str, int]:
        """
        Remove ICD-10 codes from generated text (data leakage prevention).

        Why this matters:
            Real clinical notes don't contain ICD-10 codes in the narrative.
            If codes appear, the model learns to "cheat" by looking for codes
            rather than understanding clinical content.

        Algorithm:
            1. Build regex pattern for each assigned code
            2. Also catch general ICD-10 code pattern
            3. Replace matches with empty string
            4. Clean up resulting whitespace

        Args:
            text: Raw generated text
            codes: Assigned codes that shouldn't appear

        Returns:
            Tuple of (sanitized_text, count_of_codes_removed)
        """
        sanitized = text
        codes_removed = 0

        # -----------------------------------------------------------------
        # 5.1: Remove specific assigned codes
        # -----------------------------------------------------------------
        for code in codes:
            # Match code with optional parentheses, brackets
            pattern = rf"[\(\[\s]*{re.escape(code.code)}[\)\]\s]*"
            matches = re.findall(pattern, sanitized, re.IGNORECASE)
            if matches:
                codes_removed += len(matches)
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # -----------------------------------------------------------------
        # 5.2: Remove general ICD-10 code pattern
        # -----------------------------------------------------------------
        # Pattern: Letter + 2 digits + optional dot + 0-4 digits + optional letter
        general_pattern = r"\b[A-Z]\d{2}\.?\d{0,4}[A-Z]?\b"
        general_matches = re.findall(general_pattern, sanitized)
        if general_matches:
            codes_removed += len(general_matches)
            sanitized = re.sub(general_pattern, "", sanitized)

        # -----------------------------------------------------------------
        # 5.3: Clean up whitespace
        # -----------------------------------------------------------------
        # Remove multiple spaces
        sanitized = re.sub(r"  +", " ", sanitized)
        # Remove empty parentheses
        sanitized = re.sub(r"\(\s*\)", "", sanitized)
        # Remove empty brackets
        sanitized = re.sub(r"\[\s*\]", "", sanitized)
        # Clean up line starts
        sanitized = re.sub(r"^\s*:\s*", "", sanitized, flags=re.MULTILINE)

        return sanitized.strip(), codes_removed

    # =========================================================================
    # STAGE 6: UTILITIES
    # =========================================================================

    def _generate_note_id(self) -> str:
        """Generate unique note identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:8]
        return f"note_{timestamp}_{unique_suffix}"

    def _get_model_name(self) -> str:
        """Get model name from LLM client if available."""
        if hasattr(self._llm_client, "model_name"):
            return self._llm_client.model_name
        return "unknown"

    # =========================================================================
    # STAGE 7: STATISTICS
    # =========================================================================

    @property
    def generation_count(self) -> int:
        """Number of notes generated."""
        return self._generation_count

    @property
    def sanitization_count(self) -> int:
        """Number of notes that required sanitization."""
        return self._sanitization_count

    @property
    def sanitization_rate(self) -> float:
        """Percentage of notes that required sanitization."""
        if self._generation_count == 0:
            return 0.0
        return (self._sanitization_count / self._generation_count) * 100
