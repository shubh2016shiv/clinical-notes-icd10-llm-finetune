"""
Data Models for Clinical Notes Generation

This module contains all data model classes (dataclasses) used throughout
the clinical notes generation system, including ICD-10 codes, clinical notes,
and validation results.

Author: RhythmX AI Team
Date: November 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ICD10Code:
    """
    Represents a single ICD-10 diagnosis code with all relevant metadata.
    
    This dataclass encapsulates all information about an ICD-10 code including
    its identifier, description, billability status, and hierarchical category
    information from the ICD-10 classification system.
    
    Attributes:
        icd10_code: The ICD-10 code identifier (e.g., "E11.9")
        icd10_name: Full descriptive name of the diagnosis
        is_billable: Whether this code can be used for billing purposes
        diagnosis_category: High-level diagnostic category
        category_code: Category identifier code
        category_description: Description of the category
        effective_start_date: Date when code became effective (optional)
        effective_end_date: Date when code was deprecated (optional)
        hcc_raf: Hierarchical Condition Category Risk Adjustment Factor (optional)
    
    Example:
        >>> icd10 = ICD10Code(
        ...     icd10_code="E11.9",
        ...     icd10_name="Type 2 diabetes mellitus without complications",
        ...     is_billable=True,
        ...     diagnosis_category="Endocrine, nutritional and metabolic diseases",
        ...     category_code="E08-E13",
        ...     category_description="Diabetes mellitus"
        ... )
    """
    
    icd10_code: str
    icd10_name: str
    is_billable: bool
    diagnosis_category: str
    category_code: str
    category_description: str
    effective_start_date: Optional[str] = None
    effective_end_date: Optional[str] = None
    hcc_raf: Optional[float] = None


@dataclass
class ClinicalNote:
    """
    Represents a generated clinical note with associated ICD-10 codes and validation.
    
    This dataclass encapsulates a complete clinical note including its content,
    assigned diagnosis codes, and validation status. It serves as the primary
    data structure for storing and exporting generated clinical documentation.
    
    Attributes:
        note_id: Unique identifier for this clinical note
        note_type: Type of clinical note (e.g., "Progress Note")
        clinical_content: Full text content of the clinical note
        assigned_icd10_codes: List of ICD-10 code strings assigned to this note
        icd10_code_details: Detailed information for each assigned ICD-10 code
        generation_timestamp: ISO format timestamp of when note was generated
        validation_status: Overall validation status (VALID, ACCEPTABLE, REVIEW_REQUIRED)
        validation_details: Detailed validation results including confidence scores
    
    Example:
        >>> note = ClinicalNote(
        ...     note_id="NOTE_20250101_120000_1234",
        ...     note_type="Progress Note",
        ...     clinical_content="Patient presents with...",
        ...     assigned_icd10_codes=["E11.9", "I10"],
        ...     icd10_code_details=[...],
        ...     generation_timestamp="2025-01-01T12:00:00",
        ...     validation_status="VALID",
        ...     validation_details={"confidence_score": 95}
        ... )
    """
    
    note_id: str
    note_type: str
    clinical_content: str
    assigned_icd10_codes: List[str]
    icd10_code_details: List[Dict]
    generation_timestamp: str
    validation_status: str
    validation_details: Dict


@dataclass
class ValidationResult:
    """
    Represents the validation result for ICD-10 code assignments to a clinical note.
    
    This dataclass contains the complete validation assessment including which codes
    are valid, invalid, or missing, along with a quantitative validation score and
    explanatory notes from the validation process.
    
    Attributes:
        is_valid: Overall boolean indicating if validation passed
        valid_codes: List of ICD-10 codes that are properly supported by the note
        invalid_codes: List of ICD-10 codes not supported by documentation
        missing_codes: List of diagnoses present in note but not coded
        validation_score: Numeric score (0-100) indicating coding accuracy
        validation_notes: Human-readable explanation of validation findings
    
    Example:
        >>> result = ValidationResult(
        ...     is_valid=True,
        ...     valid_codes=["E11.9", "I10"],
        ...     invalid_codes=[],
        ...     missing_codes=[],
        ...     validation_score=95.0,
        ...     validation_notes="All codes properly supported by documentation"
        ... )
    """
    
    is_valid: bool
    valid_codes: List[str]
    invalid_codes: List[str]
    missing_codes: List[str]
    validation_score: float
    validation_notes: str

