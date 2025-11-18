"""
ICD-10 Code Validation Prompt

This module contains the centralized prompt template for validating ICD-10 code
assignments against clinical documentation with comprehensive auditing methodology.

Author: RhythmX AI Team
Date: November 2025
"""

from typing import List
from ..core.data_models import ICD10Code


def get_icd10_validation_prompt(
    clinical_note: str,
    icd10_reference_data: List[ICD10Code]
) -> str:
    """
    Generate the comprehensive prompt for ICD-10 code validation.
    
    This function creates a detailed prompt that instructs the AI model to perform
    a thorough audit of ICD-10 code assignments against clinical documentation,
    checking encounter types, laterality, severity, patient applicability, and
    identifying missing codes.
    
    Args:
        clinical_note: The clinical note text to validate against
        icd10_reference_data: List of ICD10Code objects assigned to the note
        
    Returns:
        Formatted prompt string ready for AI model consumption
        
    Example:
        >>> prompt = get_icd10_validation_prompt(
        ...     clinical_note="Patient presents with...",
        ...     icd10_reference_data=[code1, code2]
        ... )
    """
    # Build codes description for validation prompt
    codes_with_descriptions = "\n".join([
        f"- {code.icd10_code}: {code.icd10_name}"
        for code in icd10_reference_data
    ])
    
    prompt = f"""You are a certified medical coding auditor (CCS, CPC) with expertise in ICD-10-CM coding accuracy and clinical documentation improvement.

Your task is to perform a detailed audit of ICD-10 code assignments against clinical documentation.

CLINICAL NOTE TO AUDIT:
{clinical_note}

ASSIGNED ICD-10 CODES FOR VALIDATION:
{codes_with_descriptions}

VALIDATION METHODOLOGY - Follow these steps systematically:

STEP 1 - ANALYZE ENCOUNTER TYPE ACCURACY:
For EACH assigned code, verify:
- Does the code character 7 match the documented encounter?
  * 'A' (initial) = First time diagnosis, new presentation, initial visit
  * 'D' (subsequent) = Follow-up visit, ongoing care, return for same condition
  * 'S' (sequela) = Late effect, complication of healed condition
- Quote the EXACT phrase from the note that indicates encounter type
- Flag if encounter type is INCONSISTENT

STEP 2 - VERIFY LATERALITY SPECIFICITY:
For codes with laterality (left/right/bilateral/unspecified):
- Search the note for explicit laterality documentation
- If code specifies "left" but note says "unspecified" → INVALID
- If code says "unspecified" but note specifies "left" → SUBOPTIMAL (should use more specific code)
- Quote the exact documentation of laterality

STEP 3 - VALIDATE SEVERITY/DEGREE/TYPE:
For codes with severity qualifiers:
- Verify the documented severity matches the code
- Example: Code says "third degree burn" but note describes "superficial redness" → INVALID
- Check degree, displacement, type specifications

STEP 4 - CONFIRM PATIENT APPLICABILITY:
- Verify the code applies to the correct patient
- Example: Z38.xx codes are ONLY for newborns, NOT mothers
- Check age, gender appropriateness

STEP 5 - ASSESS CLINICAL SUPPORT:
- Is there sufficient documentation of symptoms, findings, or diagnostic results?
- Are there signs/symptoms described but NOT coded?

STEP 6 - IDENTIFY MISSING CODES:
- Look for documented diagnoses, symptoms, or conditions NOT reflected in codes
- Consider external cause codes if injury/poisoning is documented

EXAMPLE VALIDATION REASONING:

Code: S62.126D (Nondisp fx of lunate, unsp wrist, subsequent encounter)
Note states: "Patient presents for INITIAL evaluation of LEFT wrist fracture of lunate bone"
Analysis:
- Encounter type: Code 'D' (subsequent) but note says "INITIAL" → INVALID ❌
- Laterality: Code says "unspecified" but note specifies "LEFT" → SUBOPTIMAL ❌
- Correct code should be: S62.116A (Nondisp fx of lunate, LEFT wrist, INITIAL)

OUTPUT REQUIREMENT:
Provide your analysis in this JSON format:

{{
    "valid_codes": [
        "List codes that are COMPLETELY accurate"
    ],
    "invalid_codes": [
        {{
            "code": "ICD10 code",
            "reason": "SPECIFIC reason: encounter type mismatch/laterality error/severity mismatch/patient applicability/insufficient documentation",
            "quote_from_note": "Exact phrase from note showing the issue",
            "suggested_correct_code": "The code that SHOULD have been used"
        }}
    ],
    "missing_codes": [
        {{
            "suggested_code": "ICD10 code",
            "reason": "Diagnosis/condition documented but not coded",
            "supporting_documentation": "Quote from note"
        }}
    ],
    "confidence_score": <0-100>,
    "validation_notes": "Summary of key findings, patterns of errors, documentation quality assessment"
}}

Be thorough and specific. Your audit determines coding compliance and reimbursement accuracy.
"""
    
    return prompt
