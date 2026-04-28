"""
Backward-compatible wrapper for centralized ICD single-condition resolution prompts.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.icd_codes import CandidateCode
from clinical_note_generation_v3.prompt_specs.registry import (
    build_icd_resolution_prompt_spec,
)


def build_icd_code_selection_prompt_for_single_condition(
    *,
    condition_name: str,
    bundle_archetype: str,
    encounter_context: str,
    retrieved_candidates: list[CandidateCode],
) -> str:
    prompt_spec = build_icd_resolution_prompt_spec(
        condition_name=condition_name,
        bundle_archetype=bundle_archetype,
        encounter_context=encounter_context,
        retrieved_candidates=retrieved_candidates,
    )
    return prompt_spec.user_prompt
