"""
Prompt specs for final ICD-10-CM ground-truth adjudication.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.constraints import ClinicalBundleSemanticConstraints
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote
from clinical_note_generation_v3.prompt_specs.contracts import (
    DIAGNOSIS_EXTRACTION_RESPONSE_SCHEMA,
    JSON_ONLY_OUTPUT_CONTRACT,
    PromptSpec,
)
from clinical_note_generation_v3.prompt_specs.rendering import render_list_block

DIAGNOSIS_EXTRACTION_PROMPT_ID = "final_diagnosis_extraction"
DIAGNOSIS_EXTRACTION_PROMPT_VERSION = "v2_z_code_surveillance"


def build_diagnosis_extraction_prompt_spec(
    *,
    generated_clinical_note: GeneratedClinicalNote,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
) -> PromptSpec:
    seeded_bundle = bundle_semantic_constraints.seeded_bundle
    seeded_condition_lines = [
        f"- {entry.condition_name} | seeded_code={entry.icd_code}"
        for entry in seeded_bundle.resolved_conditions
    ]
    system_prompt = (
        "You are an expert outpatient ICD-10-CM diagnosis extraction reviewer. "
        "Extract every diagnosis concept that could affect coding and set should_code "
        "based solely on whether the diagnosis is billable for this specific encounter — "
        "status alone does not determine should_code.\n\n"
        "CRITICAL RULE — Personal history / Z-category codes in surveillance encounters: "
        "When the encounter is a surveillance, follow-up, or monitoring visit for a "
        "previously treated condition (e.g., 'follow-up after cancer resection', "
        "'surveillance colonoscopy for prior malignancy'), the personal history of that "
        "condition IS the reason for the encounter and MUST be coded (should_code=true). "
        "Set status='historical' (the disease is not currently active) AND should_code=true "
        "(it drives the visit). These are NOT mutually exclusive. "
        "Example: personal history of sigmoid colon adenocarcinoma in a surveillance visit "
        "→ status='historical', should_code=true.\n\n"
        "Negated, ruled-out, family-history, and purely incidental background concepts "
        "must have should_code=false. " + JSON_ONLY_OUTPUT_CONTRACT
    )
    user_prompt = f"""
SEEDED CASE PROVENANCE
- Template ID: {seeded_bundle.template_id}
- Archetype: {seeded_bundle.archetype}
- Encounter context: {seeded_bundle.encounter_context}

SEEDED CONDITIONS AND CODES
{render_list_block(seeded_condition_lines, default_line="- None.")}

GENERATED CLINICAL NOTE
{generated_clinical_note.note_text}

TASK
Return every diagnosis-like concept that could affect coding.
For each diagnosis:
- status: one of active, historical, negated, ruled_out, family_history, incidental.
- should_code: true when the diagnosis is billable/reportable for THIS encounter,
  regardless of status. A historical condition that is the reason for a surveillance
  or follow-up visit is should_code=true even though status='historical'.
- Include a short evidence quote or paraphrase from the note.
- In coding_rationale, explain why the diagnosis is or is not billable for this encounter.

Return JSON only:
{{
  "diagnoses": [
    {{
      "diagnosis_name": "...",
      "status": "active|historical|negated|ruled_out|family_history|incidental",
      "evidence": "...",
      "should_code": true,
      "coding_rationale": "..."
    }}
  ],
  "extraction_rationale": "..."
}}
""".strip()
    return PromptSpec(
        prompt_id=DIAGNOSIS_EXTRACTION_PROMPT_ID,
        prompt_version=DIAGNOSIS_EXTRACTION_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=DIAGNOSIS_EXTRACTION_RESPONSE_SCHEMA,
    )
