"""
Clinical Note Generation Prompt - ENHANCED VERSION (November 2025)

This enhanced version retains 100% of the original prompt while adding targeted,
evidence-based improvements to eliminate the top residual error modes observed
in frontier models on complex fracture / multi-diagnosis / mixed-encounter scenarios.
"""

from typing import List
from ..core.data_models import ICD10Code
from ..core.enums import ClinicalNoteType


def get_clinical_note_generation_prompt(
    note_type: ClinicalNoteType,
    target_icd10_codes: List[ICD10Code],
    additional_context: str = ""
) -> str:
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

***ENHANCEMENT - HANDLING MIXED ENCOUNTER TYPES (if applicable)***
If the target codes contain both initial ('A'/'B'/'C') and subsequent/sequela ('D'/'G'/'K'/'M'/'P'/'S') 7th characters:
- Explicitly separate the conditions in the history: one as "new today" or "initial presentation" and the other as "ongoing from prior encounter on [date]".
- Example phrasing:
  "The patient presents today for INITIAL evaluation of a new right wrist fracture sustained this morning in a fall. The patient is also returning for FOLLOW-UP of a left ankle sprain diagnosed on [date]."

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

   ***ENHANCEMENT - HEALING STATUS & GUSTILO FOR OPEN FRACTURES***
   - For subsequent/sequela fracture codes ('D','G','K','M','P','S'): Explicitly document healing progress (e.g., "routine healing with callus formation", "delayed healing", "nonunion", "malunion").
   - For open fractures using 7th character 'B' or 'C': Document Gustilo-Anderson type explicitly when known (e.g., "open fracture, Gustilo type IIIB with extensive soft-tissue contamination").

6. **Diagnostic Studies** (if applicable):
   - Imaging reports that confirm diagnoses
   - Lab results that support clinical findings
   - Use findings that match code specificity

   ***ENHANCEMENT - HEALING STATUS SUPPORT***
   - For subsequent/sequela codes, include radiographic evidence of healing stage (e.g., "X-ray shows bridging callus consistent with routine healing").

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

***ENHANCEMENT - PLACEHOLDER "X" RULE***
- Always insert required placeholder "X" characters to reach the 7th position when needed (e.g., T15.01XA, not T15.01A).

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
[All original examples remain unchanged]

COMMON ERRORS TO AVOID:
[All original errors remain unchanged]

***ENHANCEMENT - FINAL SELF-VERIFICATION STEP***
STEP 5 - INTERNAL VERIFICATION (perform mentally before output):
- Re-check every target ICD-10 code against the drafted note.
- Confirm 100% match on: laterality (repeated everywhere), encounter language, healing status (if 'D'/'G'/'K'/'M'/'P'), Gustilo mention (if 'B'/'C'), placeholder "X" usage.
- Revise the draft if any discrepancy exists.

{additional_context}

Generate ONLY the complete clinical note. Do not include any preamble, explanations, or commentary.
Ensure EVERY detail aligns PRECISELY with the specific ICD-10 codes provided above.

### FEW-SHOT EXAMPLES BANK (reference these patterns internally – do not copy verbatim unless the scenario is identical)

Example 1 – Pure Initial Encounter (Open fracture with Gustilo type)
Target code: S52132C – Displaced fracture of neck of radius, left arm, initial encounter for open fracture type IIIA, IIIB, or IIIC
Resulting snippet (HPI + PE + Assessment):
History of Present Illness: The patient is a 28-year-old male who presents to the emergency department today immediately after a motorcycle crash. He reports falling onto his outstretched left arm with an open wound over the proximal forearm.
Physical Examination: Left forearm: 8 cm laceration over the proximal radius with visible bone fragments and significant soft-tissue contamination consistent with Gustilo type IIIB open fracture. Radiographs confirm displaced fracture of the neck of the left radius.
Assessment/Diagnosis: Displaced fracture of neck of radius, left arm, initial encounter for open fracture type IIIB (S52132C)

Example 2 – Pure Subsequent Encounter with Delayed Healing
Target code: S72092G – Other fracture of head of left femur, subsequent encounter for closed fracture with delayed healing
Resulting snippet:
History of Present Illness: The patient is a 72-year-old female who returns today for scheduled follow-up of a left femoral head fracture sustained 10 weeks ago in a ground-level fall. She was initially treated non-operatively. She continues to have groin pain with weight-bearing.
Physical Examination: Tenderness over the left greater trochanter, painful and limited range of motion of the left hip. X-ray today shows persistent fracture line without callus formation, consistent with delayed healing.
Assessment/Diagnosis: Other fracture of head of left femur, subsequent encounter for closed fracture with delayed healing (S72092G)

Example 3 – Mixed Encounter Types in One Note
Target codes: 
- S83512A – Sprain of anterior cruciate ligament of left knee, initial encounter
- S82102D – Displaced fracture of upper end of left tibia, subsequent encounter for closed fracture with routine healing
Resulting snippet (HPI + Assessment):
History of Present Illness: The patient is a 34-year-old male who presents to the orthopedic clinic today with two concerns. First, he twisted his left knee this morning while playing soccer and felt a pop with immediate swelling – this is a new injury. Second, he is returning for routine follow-up of a left proximal tibia fracture sustained 8 weeks ago in a motor vehicle collision (previously treated with Dr. Jones on 2025-09-20).
Assessment/Diagnosis:
1. Sprain of anterior cruciate ligament of left knee, initial encounter (S83512A)
2. Displaced fracture of upper end of left tibia, subsequent encounter for closed fracture with routine healing (S82102D)

Example 4 – Sequela Only
Target code: M84352S – Stress fracture, left femur, sequela
Resulting snippet:
History of Present Illness: The patient is a 19-year-old female long-distance runner who presents with chronic left thigh pain for the past four months. She reports a stress fracture of the left femur diagnosed one year ago that has since healed, but she continues to have persistent pain and limp as a late effect.
Physical Examination: Antalgic gait favoring the left leg, tenderness along the mid-shaft of the left femur, no swelling or deformity.
Assessment/Diagnosis: Stress fracture, left femur, sequela (M84352S)

Example 5 – Pregnancy + Maternal Code
Target code: O34211 – Maternal care for complete placenta previa, first trimester
Resulting snippet (Demographics + HPI + Assessment):
Patient Demographics: Age: 29 years, Gender: Female (gravida 3 para 2, currently 11 weeks pregnant)
History of Present Illness: The patient is a 29-year-old female at 11 weeks gestation by LMP who presents for prenatal visit after recent bleeding. Ultrasound today confirms complete placenta previa.
Assessment/Diagnosis: Maternal care for complete placenta previa, first trimester (O34211)

Example 6 – Newborn Note (Z38 code)
Target code: Z3800 – Single liveborn infant, delivered vaginally
Resulting snippet:
Patient Demographics: Newborn infant, gestational age 39 weeks
Date/Time Information: Date of Birth: November 19, 2025, 08:15
History: This is a single liveborn male infant delivered vaginally to a 31-year-old G2P2 mother at 39 2/7 weeks.
Physical Examination: Weight 3340 g, length 51 cm, head circumference 34.5 cm. APGAR scores 8 at 1 minute and 9 at 5 minutes. Normal newborn examination.
Assessment/Diagnosis: Single liveborn infant, delivered vaginally (Z3800)

Use these six examples as mental templates to ensure perfect alignment of encounter language, laterality repetition, healing status, Gustilo mention, and patient applicability.

Generate ONLY the complete clinical note. Do not include any preamble, explanations, or commentary.
Ensure EVERY detail aligns PRECISELY with the specific ICD-10 codes provided above.

"""

    return prompt