"""
Enumerations for Clinical Note Generation

This module defines all enumeration types used throughout the clinical note
generation pipeline. Enums provide:
    1. Type safety for categorical values
    2. IDE autocomplete support
    3. Clear domain semantics

Enumeration Categories:
    ClinicalNoteType    → Types of clinical documentation
    ValidationStatus    → Validation outcome categories
    SelectionStrategy   → ICD-10 code selection strategies
    GenerationProfile   → Patient profile types for realistic generation

Author: Shubham Singh
Date: December 2025
"""

from enum import Enum


# =============================================================================
# STAGE 1: CLINICAL NOTE TYPE ENUMERATION
# =============================================================================
# Represents the different types of clinical documentation that can be generated.
# Each type has distinct structure, content expectations, and use cases.


class ClinicalNoteType(str, Enum):
    """
    Types of clinical notes that can be generated.

    What it does:
        Categorizes clinical documentation by purpose and structure,
        enabling specialized prompt templates and validation rules.

    Why it exists:
        1. Different note types have different content expectations
        2. Enables type-specific generation prompts
        3. Maps to real-world clinical documentation standards

    When to use:
        - When specifying what kind of note to generate
        - When selecting appropriate prompt templates
        - When applying type-specific validation rules

    Note Type Hierarchy (by complexity):
        Simple:   PROGRESS_NOTE, CONSULTATION_NOTE
        Medium:   H_AND_P, EMERGENCY_DEPARTMENT_VISIT
        Complex:  DISCHARGE_SUMMARY, OPERATIVE_NOTE
    """

    # -------------------------------------------------------------------------
    # 1.1 Routine Clinical Notes
    # -------------------------------------------------------------------------
    PROGRESS_NOTE = "PROGRESS_NOTE"
    """
    Daily update on patient status during hospitalization or clinic visit.
    Structure: SOAP format (Subjective, Objective, Assessment, Plan)
    Typical length: 200-500 words
    """

    CONSULTATION_NOTE = "CONSULTATION_NOTE"
    """
    Specialist evaluation requested by another provider.
    Structure: Reason for consult, findings, recommendations
    Typical length: 300-600 words
    """

    # -------------------------------------------------------------------------
    # 1.2 Comprehensive Encounter Notes
    # -------------------------------------------------------------------------
    H_AND_P = "H_AND_P"
    """
    History and Physical - comprehensive initial evaluation.
    Structure: Chief complaint, HPI, PMH, medications, exam, assessment, plan
    Typical length: 500-1000 words
    """

    EMERGENCY_DEPARTMENT_VISIT = "EMERGENCY_DEPARTMENT_VISIT"
    """
    Emergency department encounter documentation.
    Structure: Chief complaint, triage, exam, diagnostics, disposition
    Typical length: 400-800 words
    """

    # -------------------------------------------------------------------------
    # 1.3 Transition of Care Notes
    # -------------------------------------------------------------------------
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    """
    Summary of hospital stay upon discharge.
    Structure: Admission diagnosis, hospital course, discharge plan
    Typical length: 600-1200 words
    """

    OPERATIVE_NOTE = "OPERATIVE_NOTE"
    """
    Documentation of surgical procedure.
    Structure: Pre-op diagnosis, procedure, findings, post-op diagnosis
    Typical length: 300-600 words
    """

    @classmethod
    def get_all_types(cls) -> list:
        """Return all note type values as a list."""
        return [note_type.value for note_type in cls]

    @classmethod
    def from_string(cls, value: str) -> "ClinicalNoteType":
        """
        Convert string to ClinicalNoteType with case-insensitive matching.

        Args:
            value: String representation of note type

        Returns:
            Matching ClinicalNoteType enum member

        Raises:
            ValueError: If string doesn't match any note type
        """
        normalized = value.upper().replace(" ", "_").replace("-", "_")
        for note_type in cls:
            if note_type.value == normalized or note_type.name == normalized:
                return note_type
        raise ValueError(f"Unknown note type: '{value}'. " f"Valid types: {cls.get_all_types()}")


# =============================================================================
# STAGE 2: VALIDATION STATUS ENUMERATION
# =============================================================================
# Represents the outcome of clinical note validation.


class ValidationStatus(str, Enum):
    """
    Status of clinical note validation.

    What it does:
        Categorizes the outcome of validating whether a clinical note
        properly supports its assigned ICD-10 codes.

    Why it exists:
        1. Clear categorization of validation outcomes
        2. Enables filtering notes by validation status
        3. Provides semantic meaning beyond boolean is_valid

    When to use:
        - After validation to categorize the result
        - When filtering notes for training data
        - When generating quality reports
    """

    # -------------------------------------------------------------------------
    # 2.1 Positive Outcomes
    # -------------------------------------------------------------------------
    VALID = "VALID"
    """Note fully supports all assigned codes with high confidence."""

    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    """Note supports codes but has minor issues (e.g., could be more specific)."""

    # -------------------------------------------------------------------------
    # 2.2 Negative Outcomes
    # -------------------------------------------------------------------------
    INVALID = "INVALID"
    """Note does not adequately support one or more assigned codes."""

    INVALID_CRITICAL = "INVALID_CRITICAL"
    """Note has critical issues (e.g., contradictions, code leakage)."""

    # -------------------------------------------------------------------------
    # 2.3 Processing States
    # -------------------------------------------------------------------------
    PENDING = "PENDING"
    """Validation has not yet been performed."""

    SKIPPED = "SKIPPED"
    """Validation was intentionally skipped (e.g., known regeneration)."""

    ERROR = "ERROR"
    """Validation failed due to technical error."""


# =============================================================================
# STAGE 3: SELECTION STRATEGY ENUMERATION
# =============================================================================
# Represents different strategies for selecting ICD-10 codes.


class SelectionStrategy(str, Enum):
    """
    Strategies for selecting ICD-10 codes for clinical note generation.

    What it does:
        Defines different approaches to selecting medically coherent
        ICD-10 codes, from simple random selection to sophisticated
        semantic matching.

    Why it exists:
        1. Different use cases require different selection approaches
        2. Enables strategy pattern for code selection
        3. Supports progressive enhancement of selection quality

    When to use:
        - When configuring the code selection component
        - When trading off between speed and medical accuracy
        - When testing with different selection approaches

    Strategy Comparison:
        RANDOM:     Fast, low coherence, good for stress testing
        PROFILE:    Medium speed, profile-based, good for themed datasets
        SEMANTIC:   Slow, high coherence, best for production
    """

    # -------------------------------------------------------------------------
    # 3.1 Basic Strategy
    # -------------------------------------------------------------------------
    RANDOM = "random"
    """
    Random selection from billable codes.
    Speed: Fast | Coherence: Low | Use: Testing/debugging
    """

    # -------------------------------------------------------------------------
    # 3.2 Profile-Based Strategy
    # -------------------------------------------------------------------------
    PROFILE_BASED = "profile_based"
    """
    Selection guided by patient profile (chronic_care, acute_injury, etc.)
    Speed: Medium | Coherence: Medium | Use: Themed datasets
    """

    # -------------------------------------------------------------------------
    # 3.3 Semantic Strategies
    # -------------------------------------------------------------------------
    SEMANTIC = "semantic"
    """
    Selection using co-occurrence patterns from real medical data.
    Speed: Slow | Coherence: High | Use: Production datasets
    """

    VECTOR = "vector"
    """
    Selection using vector similarity search.
    Speed: Slow | Coherence: High | Use: When vector index available
    """


# =============================================================================
# STAGE 4: GENERATION PROFILE ENUMERATION
# =============================================================================
# Represents patient profile types for realistic note generation.


class GenerationProfile(str, Enum):
    """
    Patient profile types for generating realistic clinical scenarios.

    What it does:
        Defines archetypal patient profiles that guide code selection
        and clinical narrative generation for realistic, diverse training data.

    Why it exists:
        1. Ensures diversity in generated training data
        2. Produces medically coherent code combinations
        3. Enables targeted generation for specific use cases

    When to use:
        - When generating batches of clinical notes
        - When testing specific clinical domains
        - When balancing dataset diversity
    """

    # -------------------------------------------------------------------------
    # 4.1 Common Clinical Profiles
    # -------------------------------------------------------------------------
    CHRONIC_CARE = "chronic_care"
    """Focus: Diabetes, hypertension, COPD, heart disease, arthritis."""

    ACUTE_INJURY = "acute_injury"
    """Focus: Fractures, wounds, trauma, burns, dislocations."""

    MENTAL_HEALTH = "mental_health"
    """Focus: Depression, anxiety, bipolar, PTSD, substance abuse."""

    INFECTIOUS_DISEASE = "infectious_disease"
    """Focus: Pneumonia, sepsis, UTI, COVID, tuberculosis."""

    # -------------------------------------------------------------------------
    # 4.2 Specialty Profiles
    # -------------------------------------------------------------------------
    CARDIOLOGY = "cardiology"
    """Focus: Heart failure, coronary disease, arrhythmia, MI."""

    ONCOLOGY = "oncology"
    """Focus: Cancer diagnoses, treatment complications, metastasis."""

    PEDIATRIC = "pediatric"
    """Focus: Childhood diseases, developmental disorders, infections."""

    GERIATRIC = "geriatric"
    """Focus: Dementia, falls, osteoporosis, polypharmacy."""

    # -------------------------------------------------------------------------
    # 4.3 Default Profile
    # -------------------------------------------------------------------------
    DEFAULT = "default"
    """Mixed profile with varied code selection across all categories."""

    @classmethod
    def get_all_profiles(cls) -> list:
        """Return all profile values as a list."""
        return [profile.value for profile in cls]
