"""
Backward-compatible wrappers for centralized seeded clinical note prompt specs.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.bundle import SeededClinicalBundle
from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
)
from clinical_note_generation_v3.prompt_specs.registry import (
    build_clinical_note_generation_prompt_spec,
    build_clinical_note_revision_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.rendering import compose_chat_prompt


def build_seeded_clinical_note_generation_response_schema() -> dict:
    prompt_spec = build_clinical_note_generation_prompt_spec(
        bundle_semantic_constraints=_build_empty_constraints_placeholder(),
        fake_patient_name="Placeholder Patient",
        fake_patient_medical_record_number="MRN-00000000",
        fake_patient_date_of_birth="1970-01-01",
        generation_attempt_number=1,
    )
    return prompt_spec.response_schema or {}


def build_seeded_clinical_note_generation_prompt(
    *,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    fake_patient_name: str,
    fake_patient_medical_record_number: str,
    fake_patient_date_of_birth: str,
    generation_attempt_number: int,
) -> str:
    prompt_spec = build_clinical_note_generation_prompt_spec(
        bundle_semantic_constraints=bundle_semantic_constraints,
        fake_patient_name=fake_patient_name,
        fake_patient_medical_record_number=fake_patient_medical_record_number,
        fake_patient_date_of_birth=fake_patient_date_of_birth,
        generation_attempt_number=generation_attempt_number,
    )
    return compose_chat_prompt(
        system_prompt=prompt_spec.system_prompt,
        user_prompt=prompt_spec.user_prompt,
    )


def build_seeded_clinical_note_revision_prompt(
    *,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    previous_clinical_note_text: str,
    revision_targets: list[str],
    metadata_fix_instructions: list[str],
    fake_patient_name: str,
    fake_patient_medical_record_number: str,
    fake_patient_date_of_birth: str,
    generation_attempt_number: int,
) -> str:
    prompt_spec = build_clinical_note_revision_prompt_spec(
        bundle_semantic_constraints=bundle_semantic_constraints,
        previous_clinical_note_text=previous_clinical_note_text,
        revision_targets=revision_targets,
        metadata_fix_instructions=metadata_fix_instructions,
        fake_patient_name=fake_patient_name,
        fake_patient_medical_record_number=fake_patient_medical_record_number,
        fake_patient_date_of_birth=fake_patient_date_of_birth,
        generation_attempt_number=generation_attempt_number,
    )
    return compose_chat_prompt(
        system_prompt=prompt_spec.system_prompt,
        user_prompt=prompt_spec.user_prompt,
    )


def _build_empty_constraints_placeholder() -> ClinicalBundleSemanticConstraints:
    return ClinicalBundleSemanticConstraints(
        seeded_bundle=SeededClinicalBundle(
            template_id="placeholder",
            archetype="placeholder",
            encounter_context="placeholder",
            active_condition_names=[],
            resolved_conditions=[],
        ),
        per_code_note_writing_constraints=[],
    )
