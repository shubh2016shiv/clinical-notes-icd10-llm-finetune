"""
Centralized prompt spec for ICD single-condition resolution.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.icd_codes import CandidateCode
from clinical_note_generation_v3.core.retrieval.icd_candidate_formatter import (
    assign_candidate_ids,
    format_candidates_as_markdown_kv,
)
from clinical_note_generation_v3.prompt_specs.contracts import (
    ICD_SELECTION_RESPONSE_SCHEMA,
    JSON_ONLY_OUTPUT_CONTRACT,
    PromptSpec,
)

ICD_RESOLUTION_PROMPT_ID = "icd_single_condition_resolution"
ICD_RESOLUTION_PROMPT_VERSION = "v1_centralized"

_INJURY_ENCOUNTER_SUFFIX_GUIDANCE_BY_ARCHETYPE: dict[str, str] = {
    "acute_injury_initial_encounter": (
        "This is an INITIAL encounter for an acute injury or fracture. Prefer 7th-character 'A' codes when relevant."
    ),
    "acute_injury_subsequent_encounter": (
        "This is a SUBSEQUENT encounter for an injury or fracture. Prefer follow-up/healing 7th-character codes when relevant."
    ),
    "acute_injury_sequela_followup": (
        "This is a SEQUELA encounter. Prefer 7th-character 'S' codes when relevant."
    ),
}

_NON_INJURY_ARCHETYPES_THAT_DO_NOT_USE_ENCOUNTER_SUFFIX = {
    "chronic_care_followup",
    "behavioral_health_outpatient_followup",
    "pediatric_acute_illness",
    "pediatric_chronic_respiratory_management",
    "oncology_active_treatment",
    "oncology_surveillance_and_remission",
}


def build_icd_resolution_prompt_spec(
    *,
    condition_name: str,
    bundle_archetype: str,
    encounter_context: str,
    retrieved_candidates: list[CandidateCode],
) -> PromptSpec:
    candidates_with_stable_ids = assign_candidate_ids(retrieved_candidates)
    formatted_candidate_block = format_candidates_as_markdown_kv(candidates_with_stable_ids)
    candidate_count = len(candidates_with_stable_ids)
    encounter_suffix_paragraph = _build_encounter_suffix_guidance_paragraph(bundle_archetype)

    system_prompt = (
        "You are an expert ICD-10-CM medical coder selecting a single billable code "
        "for one fixed condition from a retrieved candidate list. " + JSON_ONLY_OUTPUT_CONTRACT
    )
    user_prompt = f"""
CONDITION TO CODE
Condition name: {condition_name}

CLINICAL ENCOUNTER CONTEXT
Encounter archetype: {bundle_archetype}
Encounter description: {encounter_context}
Encounter suffix guidance: {encounter_suffix_paragraph or 'None'}

RETRIEVED ICD-10-CM CANDIDATES ({candidate_count} candidates)
You must select from this list only.

{formatted_candidate_block}

CODE SELECTION RULES
1. Select exactly one code from the candidate list above.
2. Choose the most specific billable code that accurately represents the stated condition.
3. Honor laterality, severity, complication status, encounter context, and explicit unspecified wording.
4. If no candidate accurately represents the condition, set resolution_succeeded to false and selected_icd_code to null.

REQUIRED JSON OUTPUT
{{
  "selected_icd_code": "<code>" or null,
  "selected_icd_description": "<official description>" or null,
  "selection_rationale": "<one sentence>",
  "resolution_succeeded": true or false
}}
""".strip()
    return PromptSpec(
        prompt_id=ICD_RESOLUTION_PROMPT_ID,
        prompt_version=ICD_RESOLUTION_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=ICD_SELECTION_RESPONSE_SCHEMA,
    )


def _build_encounter_suffix_guidance_paragraph(archetype: str) -> str:
    if archetype in _INJURY_ENCOUNTER_SUFFIX_GUIDANCE_BY_ARCHETYPE:
        return _INJURY_ENCOUNTER_SUFFIX_GUIDANCE_BY_ARCHETYPE[archetype]
    if archetype in _NON_INJURY_ARCHETYPES_THAT_DO_NOT_USE_ENCOUNTER_SUFFIX:
        return ""
    return (
        "If the code family uses encounter suffixes, honor the encounter description "
        "and prefer the suffix consistent with the archetype."
    )
