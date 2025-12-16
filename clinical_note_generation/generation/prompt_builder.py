"""
Prompt Builder - Clinical Note Generation Prompts

This module constructs prompts for LLM-based clinical note generation.
Prompts are carefully designed to:
    1. Generate realistic clinical narratives
    2. NOT include ICD-10 codes in the text (data leakage prevention)
    3. Match the structure of real clinical documentation

Why Separate Prompt Builder:
    1. Single Responsibility: prompt construction separate from generation
    2. Testability: prompts can be tested without LLM calls
    3. Maintainability: centralized prompt templates
    4. Versioning: easy to A/B test different prompt strategies

Pipeline Position:
    Config → Repository → Selection → [PromptBuilder] → NoteGenerator
                                       ^^^^^^^^^^^^^^
                                       You are here

Author: Shubham Singh
Date: December 2025
"""

from typing import List

from clinical_note_generation.core.models import ICD10Code
from clinical_note_generation.core.enums import ClinicalNoteType


# =============================================================================
# STAGE 1: PROMPT TEMPLATES
# =============================================================================
# Base templates for different note types. These are carefully crafted to
# produce realistic clinical notes without ICD-10 code leakage.

CLINICAL_NOTE_TEMPLATE = """You are a senior physician creating detailed, realistic clinical documentation for training AI systems.

**CRITICAL RULES:**
1. DO NOT write ICD-10 codes (E11.9, S52.501A, etc.) ANYWHERE in your response
2. Write conditions in natural clinical language only
3. The note must appear authentic - as if from a real EHR system
4. Include realistic patient demographics (de-identified: use placeholder names)
5. Follow the exact structure for {note_type}

**TARGET DIAGNOSES TO SUPPORT (write about these conditions naturally):**
{diagnoses_section}

**NOTE TYPE:** {note_type}

**REQUIRED STRUCTURE:**
{structure_section}

**ADDITIONAL CONTEXT:**
{additional_context}

Generate a complete {note_type} that naturally describes and supports ALL the listed diagnoses through clinical observations, history, and findings. The diagnoses should be evident from the clinical content without ever stating the ICD-10 codes.
"""


# =============================================================================
# STAGE 2: NOTE TYPE STRUCTURES
# =============================================================================
# Defines the expected structure for each clinical note type.

NOTE_TYPE_STRUCTURES = {
    ClinicalNoteType.PROGRESS_NOTE: """
**SUBJECTIVE:**
- Chief complaint and history of present illness
- Review of systems (relevant positives and negatives)
- Patient-reported symptoms and concerns

**OBJECTIVE:**
- Vital signs (realistic values)
- Physical examination findings
- Laboratory/imaging results (if applicable)

**ASSESSMENT:**
- Clinical impression for EACH diagnosis
- DO NOT include ICD-10 codes - only clinical descriptions

**PLAN:**
- Treatment plan for each condition
- Medications, follow-up, patient education
""",
    ClinicalNoteType.H_AND_P: """
**CHIEF COMPLAINT:**
- Primary reason for visit in patient's words

**HISTORY OF PRESENT ILLNESS:**
- Detailed narrative of current symptoms
- Timeline, severity, associated symptoms

**PAST MEDICAL HISTORY:**
- Previous diagnoses, surgeries, hospitalizations

**MEDICATIONS:**
- Current medications with doses

**ALLERGIES:**
- Drug allergies and reactions

**SOCIAL HISTORY:**
- Relevant lifestyle factors

**FAMILY HISTORY:**
- Relevant family medical history

**REVIEW OF SYSTEMS:**
- Systematic review of organ systems

**PHYSICAL EXAMINATION:**
- Complete examination findings by system

**ASSESSMENT & PLAN:**
- Clinical impressions and treatment plan
- DO NOT include ICD-10 codes
""",
    ClinicalNoteType.EMERGENCY_DEPARTMENT_VISIT: """
**CHIEF COMPLAINT:**
- Reason for ED visit

**TRIAGE:**
- Acuity level and initial vitals

**HISTORY OF PRESENT ILLNESS:**
- Detailed symptom description

**PHYSICAL EXAMINATION:**
- Focused examination findings

**DIAGNOSTIC WORKUP:**
- Labs, imaging, other tests ordered

**CLINICAL COURSE:**
- Treatment provided in ED

**DISPOSITION:**
- Discharge, admission, or transfer plan
- Follow-up instructions
""",
    ClinicalNoteType.DISCHARGE_SUMMARY: """
**ADMISSION DATE / DISCHARGE DATE:**
- Dates and length of stay

**ADMITTING DIAGNOSIS:**
- Initial clinical impression

**HOSPITAL COURSE:**
- Detailed narrative of hospitalization
- Procedures, consultations, complications

**DISCHARGE DIAGNOSES:**
- Final clinical diagnoses (NO ICD-10 codes)

**DISCHARGE MEDICATIONS:**
- Medications with instructions

**FOLLOW-UP:**
- Outpatient appointments and instructions

**DISCHARGE CONDITION:**
- Patient status at discharge
""",
    ClinicalNoteType.CONSULTATION_NOTE: """
**REASON FOR CONSULTATION:**
- Why consultation was requested

**HISTORY OF PRESENT ILLNESS:**
- Relevant clinical background

**REVIEW OF RECORDS:**
- Pertinent prior documentation

**PHYSICAL EXAMINATION:**
- Specialty-focused examination

**CURRENT FINDINGS:**
- Laboratory, imaging, other data

**ASSESSMENT:**
- Consultant's clinical impression

**RECOMMENDATIONS:**
- Detailed management recommendations
""",
    ClinicalNoteType.OPERATIVE_NOTE: """
**PREOPERATIVE DIAGNOSIS:**
- Clinical indication for surgery

**POSTOPERATIVE DIAGNOSIS:**
- Diagnosis based on operative findings

**PROCEDURE:**
- Name of procedure performed

**SURGEON / ASSISTANTS:**
- Operating team

**ANESTHESIA:**
- Type of anesthesia

**FINDINGS:**
- Intraoperative findings

**PROCEDURE DETAILS:**
- Step-by-step description

**SPECIMENS:**
- Samples sent for pathology

**ESTIMATED BLOOD LOSS:**
- EBL in milliliters

**COMPLICATIONS:**
- Any intraoperative complications
""",
}


# =============================================================================
# STAGE 3: PROMPT BUILDER CLASS
# =============================================================================


class PromptBuilder:
    """
    Constructs prompts for clinical note generation.

    What it does:
        Takes ICD-10 codes and note type, produces a complete LLM prompt
        that will generate realistic clinical documentation.

    Why it exists:
        1. Centralizes prompt logic for maintainability
        2. Ensures consistent anti-leakage rules across all prompts
        3. Enables testing prompts without making LLM calls
        4. Supports multiple note types with appropriate structures

    When to use:
        - Inside NoteGenerator before calling LLM
        - When testing/debugging prompt quality

    Example:
        >>> builder = PromptBuilder()
        >>> prompt = builder.build_generation_prompt(
        ...     note_type=ClinicalNoteType.PROGRESS_NOTE,
        ...     codes=[diabetes_code, htn_code]
        ... )
        >>> print(prompt)  # Ready for LLM
    """

    def build_generation_prompt(
        self, note_type: ClinicalNoteType, codes: List[ICD10Code], additional_context: str = ""
    ) -> str:
        """
        Build complete generation prompt.

        STAGE 3.1: Format diagnoses section (natural language, NO codes)
        STAGE 3.2: Get structure for note type
        STAGE 3.3: Assemble final prompt

        Args:
            note_type: Type of clinical note to generate
            codes: ICD-10 codes to incorporate (as clinical descriptions)
            additional_context: Optional extra guidance

        Returns:
            Complete prompt string ready for LLM
        """
        # =====================================================================
        # STAGE 3.1: FORMAT DIAGNOSES SECTION
        # =====================================================================
        # Present codes as clinical conditions, NOT as codes
        diagnoses_lines = []
        for i, code in enumerate(codes, 1):
            # Use natural language description, never the code itself
            diagnosis_text = f"{i}. {code.name}"
            if code.category_name:
                diagnosis_text += f" (Category: {code.category_name})"
            diagnoses_lines.append(diagnosis_text)

        diagnoses_section = "\n".join(diagnoses_lines)

        # =====================================================================
        # STAGE 3.2: GET STRUCTURE FOR NOTE TYPE
        # =====================================================================
        structure_section = NOTE_TYPE_STRUCTURES.get(
            note_type, NOTE_TYPE_STRUCTURES[ClinicalNoteType.PROGRESS_NOTE]
        )

        # =====================================================================
        # STAGE 3.3: ASSEMBLE FINAL PROMPT
        # =====================================================================
        prompt = CLINICAL_NOTE_TEMPLATE.format(
            note_type=note_type.value.replace("_", " ").title(),
            diagnoses_section=diagnoses_section,
            structure_section=structure_section,
            additional_context=additional_context or "None",
        )

        return prompt

    def build_validation_prompt(self, clinical_note: str, codes: List[ICD10Code]) -> str:
        """
        Build prompt for validating if a clinical note supports its codes.

        Args:
            clinical_note: The generated clinical note text
            codes: ICD-10 codes that should be supported

        Returns:
            Validation prompt for LLM
        """
        codes_list = "\n".join([f"- {code.code}: {code.name}" for code in codes])

        return f"""You are a medical coding expert validating ICD-10 code assignments.

**TASK:** Evaluate if the following clinical note adequately supports the assigned ICD-10 codes.

**CLINICAL NOTE:**
{clinical_note}

**ASSIGNED ICD-10 CODES:**
{codes_list}

**EVALUATE EACH CODE:**
For each code, determine if the clinical documentation provides sufficient evidence to support the diagnosis.

**RESPOND IN JSON FORMAT:**
{{
    "is_valid": true/false,
    "confidence_score": 0-100,
    "quality_metrics": {{
        "precision": 0.0-1.0,
        "evidence": 0.0-1.0,
        "confirmation": 0.0-1.0
    }},
    "critical_issues": ["list of critical problems"],
    "warnings": ["list of minor issues"],
    "code_analysis": [
        {{"code": "X00.0", "supported": true/false, "evidence": "brief quote or note"}}
    ]
}}
"""
