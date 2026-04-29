"""
Centralized registry for all v3 prompt builders.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.constraints import ClinicalBundleSemanticConstraints
from clinical_note_generation_v3.core.models.evaluation import (
    ConditionSupportVerificationOutcome,
)
from clinical_note_generation_v3.core.models.icd_codes import CandidateCode
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote
from clinical_note_generation_v3.prompt_specs.constraint_extraction import (
    build_constraint_extraction_prompt_spec as _build_constraint_extraction_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.evaluation import (
    build_condition_support_prompt_spec as _build_condition_support_prompt_spec,
    build_rubric_judge_prompt_spec as _build_rubric_judge_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.icd_resolution import (
    build_icd_resolution_prompt_spec as _build_icd_resolution_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.icd_adjudication import (
    build_diagnosis_extraction_prompt_spec as _build_diagnosis_extraction_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.note_generation import (
    build_generation_prompt_spec as _build_generation_prompt_spec,
    build_revision_prompt_spec as _build_revision_prompt_spec,
)


def build_clinical_note_generation_prompt_spec(**kwargs):
    return _build_generation_prompt_spec(**kwargs)


def build_clinical_note_revision_prompt_spec(**kwargs):
    return _build_revision_prompt_spec(**kwargs)


def build_condition_support_verifier_prompt_spec(
    *,
    generated_clinical_note: GeneratedClinicalNote,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
):
    return _build_condition_support_prompt_spec(
        generated_clinical_note=generated_clinical_note,
        bundle_semantic_constraints=bundle_semantic_constraints,
    )


def build_rubric_judge_prompt_spec(
    *,
    generated_clinical_note: GeneratedClinicalNote,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    condition_support_verification_outcome: ConditionSupportVerificationOutcome,
):
    return _build_rubric_judge_prompt_spec(
        generated_clinical_note=generated_clinical_note,
        bundle_semantic_constraints=bundle_semantic_constraints,
        condition_support_verification_outcome=condition_support_verification_outcome,
    )


def build_icd_resolution_prompt_spec(
    *,
    condition_name: str,
    bundle_archetype: str,
    encounter_context: str,
    retrieved_candidates: list[CandidateCode],
):
    return _build_icd_resolution_prompt_spec(
        condition_name=condition_name,
        bundle_archetype=bundle_archetype,
        encounter_context=encounter_context,
        retrieved_candidates=retrieved_candidates,
    )


def build_constraint_extraction_prompt_spec(**kwargs):
    return _build_constraint_extraction_prompt_spec(**kwargs)


def build_diagnosis_extraction_prompt_spec(**kwargs):
    return _build_diagnosis_extraction_prompt_spec(**kwargs)
