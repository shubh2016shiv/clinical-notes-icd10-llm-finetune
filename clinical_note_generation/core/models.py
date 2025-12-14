"""
Domain Models for Clinical Note Generation

This module defines the core data structures used throughout the clinical note
generation pipeline. All models are immutable dataclasses designed for:
    1. Type safety and IDE support
    2. Serialization to/from JSON
    3. Clear domain semantics

Model Hierarchy:
    ICD10Code        → Represents a single ICD-10 diagnosis code
    ClinicalNote     → A generated clinical note with metadata
    ValidationResult → Validation outcome with confidence/issues

Usage:
    from clinical_note_generation.core.models import ICD10Code, ClinicalNote

    code = ICD10Code(
        code="E11.9",
        name="Type 2 diabetes mellitus without complications",
        is_billable=True
    )

Author: Shubham Singh
Date: December 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


# =============================================================================
# STAGE 1: ICD-10 CODE MODEL
# =============================================================================
# The fundamental unit of diagnosis coding in the US healthcare system.
# This model captures all attributes needed for clinical note generation.


@dataclass(frozen=True)
class ICD10Code:
    """
    Represents a single ICD-10-CM diagnosis code.

    What it does:
        Encapsulates all metadata about an ICD-10 code needed for clinical
        note generation, including billing status and category information.

    Why it exists:
        1. Type-safe representation of ICD-10 codes throughout the pipeline
        2. Immutable to prevent accidental modification during processing
        3. Contains domain logic for code validation and formatting

    When to use:
        - When selecting codes for clinical note generation
        - When validating code assignments against notes
        - When serializing/deserializing code data

    Attributes:
        code: The ICD-10-CM code string (e.g., "E11.9", "S52.501A")
        name: Human-readable description of the diagnosis
        is_billable: Whether this code can be used for billing (leaf node)
        category_code: Parent category code (e.g., "E11" for diabetes)
        category_name: Human-readable category description
        diagnosis_category: Classification tier (e.g., "Keep - CCIR 1")
        hcc_raf: Hierarchical Condition Category RAF score (for risk adjustment)

    Example:
        >>> code = ICD10Code(
        ...     code="E11.65",
        ...     name="Type 2 diabetes mellitus with hyperglycemia",
        ...     is_billable=True,
        ...     category_code="E11",
        ...     category_name="Type 2 diabetes mellitus"
        ... )
        >>> code.first_character
        'E'
        >>> code.is_injury_code
        False
    """

    # -------------------------------------------------------------------------
    # 1.1 Required Fields
    # -------------------------------------------------------------------------
    code: str
    name: str
    is_billable: bool = True

    # -------------------------------------------------------------------------
    # 1.2 Optional Category Information
    # -------------------------------------------------------------------------
    category_code: Optional[str] = None
    category_name: Optional[str] = None
    diagnosis_category: Optional[str] = None

    # -------------------------------------------------------------------------
    # 1.3 Optional Risk Adjustment Score
    # -------------------------------------------------------------------------
    hcc_raf: Optional[float] = None

    # -------------------------------------------------------------------------
    # 1.4 Computed Properties
    # -------------------------------------------------------------------------

    @property
    def first_character(self) -> str:
        """
        Returns the first character of the code, indicating ICD-10 chapter.

        ICD-10 Chapter Mapping:
            A-B: Infectious diseases
            C-D: Neoplasms
            E: Endocrine/metabolic
            F: Mental disorders
            G: Nervous system
            I: Circulatory system
            J: Respiratory system
            M: Musculoskeletal
            R: Symptoms/signs (not definitive)
            S-T: Injuries
            V-Y: External causes
            Z: Factors influencing health
        """
        return self.code[0].upper() if self.code else ""

    @property
    def is_injury_code(self) -> bool:
        """Check if this is an injury/trauma code (S/T chapters)."""
        return self.first_character in ("S", "T")

    @property
    def is_external_cause_code(self) -> bool:
        """Check if this is an external cause code (V-Y chapters)."""
        return self.first_character in ("V", "W", "X", "Y")

    @property
    def is_symptom_code(self) -> bool:
        """Check if this is a symptom/sign code (R chapter)."""
        return self.first_character == "R"

    @property
    def is_z_code(self) -> bool:
        """Check if this is a Z-code (factors influencing health status)."""
        return self.first_character == "Z"

    @property
    def requires_7th_character(self) -> bool:
        """
        Check if this code requires a 7th character extension.
        Injury codes (S/T) and some others require 7th character
        to indicate encounter type (A=initial, D=subsequent, S=sequela).
        """
        return self.is_injury_code and len(self.code) >= 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "name": self.name,
            "is_billable": self.is_billable,
            "category_code": self.category_code,
            "category_name": self.category_name,
            "diagnosis_category": self.diagnosis_category,
            "hcc_raf": self.hcc_raf,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ICD10Code":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            code=data.get("code", data.get("icd10_code", "")),
            name=data.get("name", data.get("icd10_name", "")),
            is_billable=data.get("is_billable", True),
            category_code=data.get("category_code"),
            category_name=data.get("category_name", data.get("category_description")),
            diagnosis_category=data.get("diagnosis_category"),
            hcc_raf=data.get("hcc_raf"),
        )


# =============================================================================
# STAGE 2: VALIDATION RESULT MODEL
# =============================================================================
# Captures the outcome of clinical note validation with detailed diagnostics.


@dataclass
class ValidationResult:
    """
    Result of validating a clinical note against its assigned ICD-10 codes.

    What it does:
        Captures whether a clinical note properly supports its assigned
        diagnosis codes, with confidence scoring and detailed issue tracking.

    Why it exists:
        1. Structured representation of validation outcome
        2. Enables programmatic decision-making on note quality
        3. Provides debugging information for failed validations

    When to use:
        - After running rule-based or LLM validation
        - When deciding whether to regenerate a note
        - When collecting quality metrics across batches

    Attributes:
        is_valid: Whether the note passed validation
        confidence_score: 0-100 confidence in the validation
        critical_issues: Errors that must be fixed (e.g., code not supported)
        warnings: Non-critical issues (e.g., could be more specific)
        suggestions: Recommendations for improvement
        quality_metrics: Detailed scoring breakdown

    Example:
        >>> result = ValidationResult(
        ...     is_valid=True,
        ...     confidence_score=85.0,
        ...     critical_issues=[],
        ...     warnings=["Consider adding laterality"]
        ... )
    """

    # -------------------------------------------------------------------------
    # 2.1 Core Validation Outcome
    # -------------------------------------------------------------------------
    is_valid: bool
    confidence_score: float = 0.0

    # -------------------------------------------------------------------------
    # 2.2 Issue Categorization
    # -------------------------------------------------------------------------
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # 2.3 Code-Level Details
    # -------------------------------------------------------------------------
    invalid_codes: List[str] = field(default_factory=list)
    unsupported_codes: List[str] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # 2.4 Quality Metrics (Optional Detailed Breakdown)
    # -------------------------------------------------------------------------
    quality_metrics: Optional[Dict[str, float]] = None

    # -------------------------------------------------------------------------
    # 2.5 Validation Metadata
    # -------------------------------------------------------------------------
    validation_source: str = "unknown"  # "rule_based", "llm", "hybrid"

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return len(self.critical_issues) > 0

    @property
    def issue_count(self) -> int:
        """Total number of issues (critical + warnings)."""
        return len(self.critical_issues) + len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "confidence_score": self.confidence_score,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "invalid_codes": self.invalid_codes,
            "unsupported_codes": self.unsupported_codes,
            "quality_metrics": self.quality_metrics,
            "validation_source": self.validation_source,
        }


# =============================================================================
# STAGE 3: CLINICAL NOTE MODEL
# =============================================================================
# The complete representation of a generated clinical note with all metadata.


@dataclass
class ClinicalNote:
    """
    A complete clinical note with assigned ICD-10 codes and metadata.

    What it does:
        Represents a fully generated clinical note ready for use in
        model fine-tuning, including the note text, assigned codes,
        validation status, and generation metadata.

    Why it exists:
        1. Single source of truth for a generated training example
        2. Captures full provenance (when/how note was generated)
        3. Includes validation status for quality filtering

    When to use:
        - As the output of the generation pipeline
        - When serializing notes to training datasets
        - When filtering notes by quality metrics

    Attributes:
        note_id: Unique identifier for this note
        note_type: Type of clinical note (e.g., PROGRESS_NOTE)
        clinical_text: The generated note content
        assigned_codes: List of ICD-10 codes assigned to this note
        validation_result: Outcome of validation (if performed)
        generation_profile: Profile used for generation (e.g., "chronic_care")
        generated_at: Timestamp of generation

    Example:
        >>> note = ClinicalNote(
        ...     note_id="note_001",
        ...     note_type=ClinicalNoteType.PROGRESS_NOTE,
        ...     clinical_text="Patient presents with...",
        ...     assigned_codes=[diabetes_code, hypertension_code]
        ... )
    """

    # -------------------------------------------------------------------------
    # 3.1 Identity
    # -------------------------------------------------------------------------
    note_id: str

    # -------------------------------------------------------------------------
    # 3.2 Content
    # -------------------------------------------------------------------------
    clinical_text: str
    note_type: str = "PROGRESS_NOTE"  # Will use ClinicalNoteType enum value

    # -------------------------------------------------------------------------
    # 3.3 Assigned Diagnosis Codes
    # -------------------------------------------------------------------------
    assigned_codes: List[ICD10Code] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # 3.4 Validation Status
    # -------------------------------------------------------------------------
    validation_result: Optional[ValidationResult] = None

    # -------------------------------------------------------------------------
    # 3.5 Generation Metadata
    # -------------------------------------------------------------------------
    generation_profile: str = "default"
    generated_at: datetime = field(default_factory=datetime.now)
    generation_model: str = "unknown"

    # -------------------------------------------------------------------------
    # 3.6 Computed Properties
    # -------------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """Check if note passed validation (or not yet validated)."""
        if self.validation_result is None:
            return True  # Not validated yet, assume valid
        return self.validation_result.is_valid

    @property
    def confidence_score(self) -> float:
        """Get validation confidence score (0 if not validated)."""
        if self.validation_result is None:
            return 0.0
        return self.validation_result.confidence_score

    @property
    def code_count(self) -> int:
        """Number of assigned ICD-10 codes."""
        return len(self.assigned_codes)

    @property
    def assigned_code_strings(self) -> List[str]:
        """List of assigned ICD-10 code strings."""
        return [code.code for code in self.assigned_codes]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "note_id": self.note_id,
            "note_type": self.note_type,
            "clinical_text": self.clinical_text,
            "assigned_icd10_codes": self.assigned_code_strings,
            "assigned_codes_details": [c.to_dict() for c in self.assigned_codes],
            "validation_status": "VALID" if self.is_valid else "INVALID",
            "confidence_score": self.confidence_score,
            "validation_details": (
                self.validation_result.to_dict() if self.validation_result else None
            ),
            "generation_profile": self.generation_profile,
            "generated_at": self.generated_at.isoformat(),
            "generation_model": self.generation_model,
        }

    def to_training_format(self) -> Dict[str, Any]:
        """
        Convert to format suitable for model fine-tuning.

        Returns:
            Dictionary with 'input' (clinical text) and 'output' (codes)
        """
        return {
            "input": self.clinical_text,
            "output": self.assigned_code_strings,
            "metadata": {
                "note_id": self.note_id,
                "note_type": self.note_type,
                "profile": self.generation_profile,
                "confidence": self.confidence_score,
            },
        }
