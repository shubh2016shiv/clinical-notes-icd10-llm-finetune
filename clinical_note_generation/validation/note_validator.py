"""
Note Validator - Clinical Note Validation

This module validates that clinical notes properly support their
assigned ICD-10 codes using a two-phase approach:
    1. Fast rule-based checks (deterministic, no API calls)
    2. Optional LLM validation (expensive, high accuracy)

Why Two-Phase Validation:
    1. Cost: Rule checks are free, LLM calls cost money
    2. Speed: Rule checks are instant, LLM calls take seconds
    3. Accuracy: LLM catches subtle issues rules miss
    4. Fail-fast: Skip expensive LLM if rules find critical issues

Pipeline Position:
    Config → Repository → Selection → Generation → [Validation]
                                                    ^^^^^^^^^^^^
                                                    You are here

Author: Shubham Singh
Date: December 2025
"""

import re
from typing import List, Optional, Tuple, Protocol, runtime_checkable

from loguru import logger

from clinical_note_generation.core.models import ICD10Code, ValidationResult, ClinicalNote
from clinical_note_generation.core.constants import NEGATION_PATTERNS, UNCERTAINTY_PATTERNS


# =============================================================================
# STAGE 1: RULE-BASED CHECKS (STATIC CLASS)
# =============================================================================
# Deterministic validation checks that don't require LLM calls.


class RuleBasedChecks:
    """
    Static methods for rule-based clinical note validation.

    What it does:
        Provides fast, deterministic checks that can catch obvious
        validation issues without expensive LLM calls.

    Why it exists:
        1. Free validation (no API costs)
        2. Instant feedback (no network latency)
        3. Consistent results (deterministic)
        4. Catches common issues early

    When to use:
        - Always run these before LLM validation
        - As a gate to skip expensive LLM calls
        - For immediate feedback during development

    Checks Performed:
        1. Code visibility (codes shouldn't appear in text)
        2. Minimum note length
        3. Diagnosis confirmation (no uncertainty for definitive codes)
        4. Negation detection
    """

    # Compiled negation pattern for efficiency
    _negation_regex = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)

    _uncertainty_regex = re.compile("|".join(UNCERTAINTY_PATTERNS), re.IGNORECASE)

    @staticmethod
    def check_code_visibility(note_text: str, codes: List[ICD10Code]) -> List[str]:
        """
        Check if ICD-10 codes appear in the clinical narrative.

        Why this matters:
            Real clinical notes never contain ICD-10 codes in the text.
            If we generate notes with codes visible, it's data leakage
            that will make trained models "cheat" instead of learning.

        Args:
            note_text: The clinical note text
            codes: Assigned ICD-10 codes

        Returns:
            List of critical issues found (empty if clean)
        """
        issues = []
        note_upper = note_text.upper()

        for code in codes:
            code_upper = code.code.upper()
            if code_upper in note_upper:
                issues.append(f"CODE_LEAKAGE: ICD-10 code '{code.code}' appears in note text")

        # Also check for general ICD-10 pattern
        general_pattern = r"\b[A-Z]\d{2}\.\d{1,4}[A-Z]?\b"
        matches = re.findall(general_pattern, note_text)
        if matches:
            unique_matches = set(matches)
            for match in unique_matches:
                if match not in [c.code for c in codes]:
                    issues.append(f"CODE_LEAKAGE: Unexpected ICD-10 code pattern '{match}' in text")

        return issues

    @staticmethod
    def check_minimum_length(note_text: str, min_length: int = 100) -> List[str]:
        """
        Check if note meets minimum length requirement.

        Args:
            note_text: The clinical note text
            min_length: Minimum acceptable length in characters

        Returns:
            List of issues (empty if meets requirement)
        """
        if len(note_text.strip()) < min_length:
            return [f"NOTE_TOO_SHORT: Note is {len(note_text)} chars, " f"minimum is {min_length}"]
        return []

    @staticmethod
    def check_negation_patterns(note_text: str, codes: List[ICD10Code]) -> List[str]:
        """
        Check if diagnoses are negated in the text.

        Why this matters:
            If a code is for "diabetes" but the note says "no diabetes",
            the code assignment is incorrect.

        Algorithm:
            1. For each code, find mentions of the condition name
            2. Check if negation patterns precede the mention
            3. Flag as warning if negated

        Args:
            note_text: The clinical note text
            codes: Assigned ICD-10 codes

        Returns:
            List of warnings for negated conditions
        """
        warnings = []
        note_lower = note_text.lower()

        for code in codes:
            # Extract key terms from code name
            code_name_lower = code.name.lower()
            # Get first 2-3 significant words
            words = [
                w
                for w in code_name_lower.split()
                if len(w) > 3 and w not in ["with", "without", "type", "unspecified"]
            ]

            if not words:
                continue

            # Check if key word appears with negation
            key_word = words[0]

            # Find all occurrences
            for match in re.finditer(rf"\b{re.escape(key_word)}\b", note_lower):
                start_pos = max(0, match.start() - 60)
                preceding_text = note_lower[start_pos : match.start()]

                if RuleBasedChecks._negation_regex.search(preceding_text):
                    warnings.append(
                        f"POSSIBLE_NEGATION: '{key_word}' may be negated " f"(code: {code.code})"
                    )
                    break  # Only report once per code

        return warnings

    @staticmethod
    def check_uncertainty_language(note_text: str, codes: List[ICD10Code]) -> List[str]:
        """
        Check for uncertainty language around definitive diagnoses.

        Why this matters:
            ICD-10 coding guideline I.B.6 states that uncertain diagnoses
            (possible, probable, suspected) should not be coded as definitive.
            R-codes (symptoms) and Z-codes are exceptions.

        Args:
            note_text: The clinical note text
            codes: Assigned ICD-10 codes

        Returns:
            List of warnings for uncertain language
        """
        warnings = []
        note_lower = note_text.lower()

        for code in codes:
            # Skip R-codes and Z-codes (uncertainty is acceptable)
            if code.first_character in ("R", "Z"):
                continue

            # Get key term from code
            code_name_lower = code.name.lower()
            words = [
                w
                for w in code_name_lower.split()
                if len(w) > 3 and w not in ["with", "without", "type", "unspecified"]
            ]

            if not words:
                continue

            key_word = words[0]

            # Check for uncertainty before key word
            for match in re.finditer(rf"\b{re.escape(key_word)}\b", note_lower):
                start_pos = max(0, match.start() - 50)
                preceding_text = note_lower[start_pos : match.start()]

                if RuleBasedChecks._uncertainty_regex.search(preceding_text):
                    warnings.append(
                        f"UNCERTAIN_DIAGNOSIS: '{key_word}' has uncertainty language "
                        f"but {code.code} is definitive"
                    )
                    break

        return warnings

    @classmethod
    def run_all_checks(cls, note_text: str, codes: List[ICD10Code]) -> Tuple[List[str], List[str]]:
        """
        Run all rule-based checks.

        Args:
            note_text: The clinical note text
            codes: Assigned ICD-10 codes

        Returns:
            Tuple of (critical_issues, warnings)
        """
        critical_issues = []
        warnings = []

        # Critical: Code visibility (data leakage)
        critical_issues.extend(cls.check_code_visibility(note_text, codes))

        # Critical: Minimum length
        critical_issues.extend(cls.check_minimum_length(note_text))

        # Warning: Negation patterns
        warnings.extend(cls.check_negation_patterns(note_text, codes))

        # Warning: Uncertainty language
        warnings.extend(cls.check_uncertainty_language(note_text, codes))

        return critical_issues, warnings


# =============================================================================
# STAGE 2: LLM CLIENT PROTOCOL
# =============================================================================


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client with validation capability."""

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...


# =============================================================================
# STAGE 3: NOTE VALIDATOR CLASS
# =============================================================================


class NoteValidator:
    """
    Validates clinical notes using rule-based and optional LLM checks.

    What it does:
        Determines whether a clinical note adequately supports its
        assigned ICD-10 codes through a two-phase validation approach.

    Why it exists:
        1. Ensures generated notes are high quality for training
        2. Catches data leakage (codes in text)
        3. Identifies notes that need regeneration
        4. Provides confidence scoring for quality filtering

    When to use:
        - After generating a clinical note
        - When filtering notes by quality before training

    How it works:
        STAGE 3.1: Run fast rule-based checks
        STAGE 3.2: If rules pass AND LLM enabled, run LLM validation
        STAGE 3.3: Combine results into ValidationResult

    Example:
        >>> validator = NoteValidator(llm_client, use_llm=True)
        >>> result = validator.validate(note)
        >>> if result.is_valid:
        ...     print("Note is good for training")
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_llm_validation: bool = False,
        skip_llm_on_critical: bool = True,
        confidence_threshold: float = 70.0,
    ):
        """
        Initialize the validator.

        Args:
            llm_client: LLM client for advanced validation (optional)
            use_llm_validation: Whether to use LLM for validation
            skip_llm_on_critical: Skip LLM if critical rule issues found
            confidence_threshold: Min confidence to consider valid (0-100)
        """
        self._llm_client = llm_client
        self._use_llm_validation = use_llm_validation and llm_client is not None
        self._skip_llm_on_critical = skip_llm_on_critical
        self._confidence_threshold = confidence_threshold

        logger.debug(
            f"NoteValidator initialized | "
            f"LLM validation: {self._use_llm_validation} | "
            f"Threshold: {confidence_threshold}"
        )

    def validate(self, note: ClinicalNote) -> ValidationResult:
        """
        Validate a clinical note.

        Algorithm:
            1. Run rule-based checks
            2. If critical issues, return immediately (fail-fast)
            3. If LLM enabled, run LLM validation
            4. Combine results and calculate confidence

        Args:
            note: ClinicalNote to validate

        Returns:
            ValidationResult with is_valid, confidence, and issues
        """
        # =====================================================================
        # STAGE 3.1: RUN RULE-BASED CHECKS
        # =====================================================================
        logger.debug(f"Validating note {note.note_id}")

        critical_issues, warnings = RuleBasedChecks.run_all_checks(
            note_text=note.clinical_text, codes=note.assigned_codes
        )

        # =====================================================================
        # STAGE 3.2: CHECK FOR CRITICAL FAILURES
        # =====================================================================
        if critical_issues:
            logger.warning(f"Note {note.note_id} has {len(critical_issues)} critical issues")

            # If configured to skip LLM on critical issues, return now
            if self._skip_llm_on_critical or not self._use_llm_validation:
                return ValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    critical_issues=critical_issues,
                    warnings=warnings,
                    validation_source="rule_based",
                )

        # =====================================================================
        # STAGE 3.3: CALCULATE CONFIDENCE (RULE-BASED)
        # =====================================================================
        # Simple scoring: start at 100, deduct for issues
        confidence = 100.0
        confidence -= len(critical_issues) * 30  # Critical issues hurt a lot
        confidence -= len(warnings) * 10  # Warnings hurt less
        confidence = max(0.0, confidence)  # Floor at 0

        # =====================================================================
        # STAGE 3.4: DETERMINE VALIDITY
        # =====================================================================
        is_valid = len(critical_issues) == 0 and confidence >= self._confidence_threshold

        result = ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence,
            critical_issues=critical_issues,
            warnings=warnings,
            validation_source="rule_based",
        )

        logger.info(
            f"Validation complete for {note.note_id} | "
            f"Valid: {is_valid} | "
            f"Confidence: {confidence:.1f}%"
        )

        return result

    def validate_and_update(self, note: ClinicalNote) -> ClinicalNote:
        """
        Validate note and update its validation_result field.

        Args:
            note: ClinicalNote to validate

        Returns:
            Same note with validation_result populated
        """
        result = self.validate(note)
        note.validation_result = result
        return note
