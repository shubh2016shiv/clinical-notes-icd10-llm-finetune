"""
Clinical Note Generation Prompt

This module contains the centralized prompt template for generating realistic,
de-identified clinical notes that accurately support specific ICD-10 codes.

Author: RhythmX AI Team
Date: November 2025
"""

from typing import List
from ..core.data_models import ICD10Code
from ..core.enums import ClinicalNoteType


def get_clinical_note_generation_prompt(
    note_type: ClinicalNoteType,
    target_icd10_codes: List[ICD10Code],
    additional_context: str = ""
) -> str:
    """
    Generate the comprehensive prompt for clinical note generation.
    
    This function creates a detailed prompt that instructs the AI model to generate
    accurate clinical notes that precisely support the provided ICD-10 codes with
    correct encounter types, laterality, severity, and anatomical precision.
    
    Args:
        note_type: The type of clinical note to generate
        target_icd10_codes: List of ICD-10 codes to incorporate in the note
        additional_context: Optional additional instructions or context
        
    Returns:
        Formatted prompt string ready for AI model consumption
        
    Example:
        >>> prompt = get_clinical_note_generation_prompt(
        ...     note_type=ClinicalNoteType.PROGRESS_NOTE,
        ...     target_icd10_codes=[code1, code2],
        ...     additional_context="Focus on post-surgical recovery"
        ... )
    """
    # Build ICD-10 codes description
    icd10_codes_description = "\n".join([
        f"- {code.icd10_code}: {code.icd10_name} (Category: {code.diagnosis_category})"
        for code in target_icd10_codes
    ])
    
    prompt = f"""You are an expert medical documentation specialist with 15+ years of experience in clinical documentation and ICD-10 coding.

Your task is to generate a realistic, de-identified clinical note that ACCURATELY supports specific ICD-10 codes.

STEP 1 - ANALYZE THE ICD-10 CODES:
Review these ICD-10 codes carefully:

{icd10_codes_description}

For EACH code, identify:
- The exact encounter type (initial encounter 'A', subsequent encounter 'D', sequela 'S')
- The specific laterality (left, right, bilateral, unspecified)
- The severity/degree specified (first degree, second degree, third degree, unspecified degree, etc.)
- The displacement status (displaced vs nondisplaced for fractures)
- The fracture type (open vs closed)
- Who the code applies to (patient, newborn, fetus, etc.)
- Any episode of care indicator (active treatment, routine healing, delayed healing)
- The clinical context that would justify this EXACT code

STEP 2 - VALIDATE CODE COMPATIBILITY:
Before creating the scenario, check:
- Can ALL these codes reasonably occur in the SAME patient at the SAME encounter?
- Do the encounter types match (don't mix initial 'A' and subsequent 'D' unless clinically justified)?
- Is the patient demographics compatible (age, gender, pregnancy status)?
- Do the anatomical locations make sense together?

STEP 3 - CREATE A COHERENT CLINICAL SCENARIO:
Design a realistic clinical scenario where:
- ALL codes logically co-exist in the SAME patient encounter
- The encounter type matches the clinical timeline (initial vs subsequent)
- The laterality is explicitly documented in examination findings
- The severity/degree is clearly described in the note
- Age, gender, and circumstances align with all diagnoses
- The mechanism of injury/illness is plausible for ALL diagnoses
- Timing of events is consistent with encounter types

STEP 4 - WRITE THE CLINICAL NOTE:
Note Type: {note_type.value}

Include these MANDATORY elements:

1. **Patient Demographics**: 
   - Age and gender that fits ALL diagnoses
   - Use generic identifiers: "Patient ID: XXXXX", "MRN: XXXXX"

2. **Date/Time Information**:
   - Use fictional dates (format: YYYY-MM-DD or Month DD, YYYY)
   - Ensure dates align with encounter type (initial vs subsequent)

3. **Chief Complaint**: 
   - Directly related to the primary diagnosis
   - Must mention laterality if applicable

4. **History of Present Illness**: 
   - Timeline that matches encounter types:
     * Initial (A): "presents today for the first time", "new onset", "initially developed"
     * Subsequent (D): "returns for follow-up", "seen previously on [date]", "continued care"
     * Sequela (S): "presenting with late effects of", "long-term complication"
   - Explicit mention of affected body part with LATERALITY (e.g., "left wrist", not "wrist")
   - For injuries: describe mechanism that explains ALL injuries
   - For subsequent visits: reference initial visit date and treatment

5. **Physical Examination**:
   - Document EXACT laterality for EVERY finding (left/right/bilateral)
   - Describe severity/degree using EXACT terminology from ICD-10 code:
     * For burns: specify "first degree", "second degree", "third degree", or "unspecified degree"
     * For fractures: specify "displaced" vs "nondisplaced", "open" vs "closed"
   - Include specific anatomical details from the code descriptions
   - Use precise anatomical terms (e.g., "fifth metacarpal", "navicular bone", "sigmoid colon")

6. **Diagnostic Studies** (if applicable):
   - Imaging reports that confirm diagnoses
   - Lab results that support clinical findings
   - Use findings that match code specificity

7. **Assessment/Diagnosis**: 
   - List diagnoses using the EXACT terminology from ICD-10 code names
   - Specify encounter context explicitly:
     * "initial encounter for [diagnosis]"
     * "subsequent encounter for [diagnosis]"
   - Include laterality in every diagnosis statement

8. **Plan**: 
   - Treatments appropriate for the encounter type
   - For initial (A): New treatment plan, prescriptions, consultations
   - For subsequent (D): Ongoing care, progress assessment, adjustment of treatment

CRITICAL CODING RULES - MUST FOLLOW:

**ENCOUNTER TYPE RULES:**
- 'A' (Initial) = Document: "initial presentation", "first visit for this condition", "newly diagnosed", "presenting today with new onset"
- 'D' (Subsequent) = Document: "follow-up visit", "return for continued care", "seen previously on [date]", "ongoing treatment"
- 'S' (Sequela) = Document: "late effect of", "residual condition from", "complication following healed"
- NEVER use initial language ('first time', 'new') with 'D' codes
- NEVER use follow-up language ('return visit', 'previously seen') with 'A' codes

**LATERALITY RULES:**
- If code specifies LEFT: Use "left [body part]" in EVERY mention (HPI, PE, Assessment)
- If code specifies RIGHT: Use "right [body part]" in EVERY mention
- If code specifies BILATERAL: Document both sides affected
- If code specifies UNSPECIFIED: You may use "left", "right", or leave unspecified, but be consistent
- NEVER write generic "wrist" or "ankle" when code has specific laterality

**SEVERITY/DEGREE RULES:**
- If code states "unspecified degree": Document symptoms without specifying exact degree
- If code states specific degree: Document exact degree in physical exam
- For burns: Match exactly (first/second/third degree vs unspecified)
- For fractures: Match displacement (displaced vs nondisplaced) and type (open vs closed)

**FRACTURE-SPECIFIC RULES:**
- Open fracture: Document "open wound", "bone protrusion", "skin break"
- Closed fracture: Document no skin break
- Displaced: Document "displacement", "malalignment"
- Nondisplaced: Document "good alignment", "no displacement"

**PATIENT APPLICABILITY RULES:**
- Z38.xx codes (liveborn infant): Note must be for NEWBORN, not mother
- O codes (pregnancy/delivery): Note must be for PREGNANT PATIENT/MOTHER
- Pediatric conditions: Ensure age-appropriate
- Gender-specific codes: Match patient gender

**ANATOMICAL PRECISION RULES:**
- Use exact bone names: "fifth metacarpal", "lunate", "navicular/scaphoid"
- Use exact organ names: "sigmoid colon", not just "colon"
- Include anatomical descriptors: "neck of fifth metacarpal", "middle third of navicular"

**TEMPORAL CONSISTENCY RULES:**
- All dates must be fictional but internally consistent
- For subsequent encounters: Reference initial encounter date
- Timeline must support healing stage (initial vs subsequent)

**DOCUMENTATION QUALITY RULES:**
- Use complete sentences with proper medical terminology
- Avoid abbreviations unless standard (e.g., CT, MRI, BP)
- Include vital signs for admission/ED notes
- Include medications for discharge/admission notes
- Sign and date the note with fictional provider name: "Dr. [LAST NAME], MD"

EXAMPLES OF CORRECT DOCUMENTATION:

**Example 1 - Laterality & Encounter Type:**
Code: S62367B (Nondisplaced fracture of neck of fifth metacarpal bone, LEFT hand, INITIAL encounter for OPEN fracture)
✓ CORRECT: "Patient presents today for initial evaluation of an open fracture involving the neck of the fifth metacarpal bone of the LEFT hand. Physical examination reveals an open wound over the left fifth metacarpal with visible bone, no displacement noted on radiograph."
✗ WRONG: "Patient with fracture of hand, follow-up visit" (missing laterality, wrong encounter type, missing open/nondisplaced details)

**Example 2 - Multiple Encounter Types:**
If you have mixed 'A' and 'D' codes:
Code 1: S62367B (initial 'B' which includes 'A')
Code 2: S36523D (subsequent 'D')
✓ CORRECT: "Patient presents for INITIAL evaluation of new left hand fracture sustained today in motor vehicle accident. Patient is also here for FOLLOW-UP care of sigmoid colon contusion diagnosed one week ago during previous hospitalization."

**Example 3 - Newborn vs Mother:**
Code: Z3800 (Single liveborn infant, delivered vaginally)
✓ CORRECT: "PATIENT: Newborn infant, delivered vaginally at 0800 today. Birthweight: 3200g. APGAR scores 8 and 9."
✗ WRONG: "Patient is a 28-year-old female who delivered a healthy infant today." (This is the MOTHER's note, not newborn's)

**Example 4 - Burn Degree:**
Code: T25011D (Burn of UNSPECIFIED degree of right ankle, subsequent)
✓ CORRECT: "Patient returns for follow-up of right ankle burn sustained two weeks ago. Physical exam shows erythema and healing tissue on right ankle, no blistering currently present."
✗ WRONG: "Patient with third-degree burn of right ankle, initial visit" (specified degree when code says unspecified, wrong encounter type)

COMMON ERRORS TO AVOID:
❌ Mixing initial and subsequent language in same diagnosis
❌ Using "wrist" when code specifies "left wrist"
❌ Describing newborn delivery in mother's chart with Z38 code
❌ Using "chemical irritation" when code says "corrosion"
❌ Documenting "follow-up" with 'A' (initial) codes
❌ Omitting "open" designation when code has 'B' (open fracture)
❌ Missing displacement status in fracture documentation

{additional_context}

Generate ONLY the complete clinical note. Do not include any preamble, explanations, or commentary.
Ensure EVERY detail aligns PRECISELY with the specific ICD-10 codes provided above.
"""
    
    return prompt
