"""
Centralized prompt specs for verification and judging.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.constraints import ClinicalBundleSemanticConstraints
from clinical_note_generation_v3.core.models.evaluation import ConditionSupportVerificationOutcome
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote
from clinical_note_generation_v3.prompt_specs.contracts import (
    CONDITION_SUPPORT_RESPONSE_SCHEMA,
    JSON_ONLY_OUTPUT_CONTRACT,
    PromptSpec,
    RUBRIC_JUDGE_RESPONSE_SCHEMA,
)
from clinical_note_generation_v3.prompt_specs.rendering import render_list_block

VERIFIER_PROMPT_ID = "condition_support_verifier"
VERIFIER_PROMPT_VERSION = "v1_centralized"
RUBRIC_JUDGE_PROMPT_ID = "rubric_judge"
RUBRIC_JUDGE_PROMPT_VERSION = "v1_centralized"


def build_condition_support_prompt_spec(
    *,
    generated_clinical_note: GeneratedClinicalNote,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
) -> PromptSpec:
    seeded_bundle = bundle_semantic_constraints.seeded_bundle
    system_prompt = (
        "You are verifying whether a generated clinical note adequately supports "
        "a fixed seeded condition bundle. " + JSON_ONLY_OUTPUT_CONTRACT
    )
    active_condition_lines = [
        f"{resolved_condition_entry.condition_name}"
        for resolved_condition_entry in seeded_bundle.resolved_conditions
    ]
    prohibited_implications = render_list_block(
        bundle_semantic_constraints.all_must_not_imply_items(),
        default_line="- None explicitly listed.",
    )
    user_prompt = f"""
SEEDED ACTIVE CONDITIONS
{render_list_block(active_condition_lines, default_line='- None.')}

THINGS THE NOTE MUST NOT IMPLY
{prohibited_implications}

GENERATED CLINICAL NOTE
{generated_clinical_note.note_text}

TASK
Decide whether the note adequately expresses the intended active conditions.
Return only JSON with:
- outcome: "pass" or "fail"
- under_supported_conditions
- unsupported_implied_conditions
- history_or_negation_drift_detected
- verifier_notes
""".strip()

    return PromptSpec(
        prompt_id=VERIFIER_PROMPT_ID,
        prompt_version=VERIFIER_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=CONDITION_SUPPORT_RESPONSE_SCHEMA,
    )


def build_rubric_judge_prompt_spec(
    *,
    generated_clinical_note: GeneratedClinicalNote,
    bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    condition_support_verification_outcome: ConditionSupportVerificationOutcome,
) -> PromptSpec:
    seeded_bundle = bundle_semantic_constraints.seeded_bundle
    condition_constraint_lines: list[str] = []
    for resolved_condition_entry, code_semantic_constraints in zip(
        seeded_bundle.resolved_conditions,
        bundle_semantic_constraints.per_code_note_writing_constraints,
        strict=False,
    ):
        condition_constraint_lines.append(
            f"- {resolved_condition_entry.condition_name} | "
            f"must_include={code_semantic_constraints.must_include_in_note} | "
            f"must_not_imply={code_semantic_constraints.must_not_imply_in_note}"
        )

    system_prompt = (
        "You are judging the quality of a seeded synthetic clinical note for "
        "general note quality and metadata alignment. " + JSON_ONLY_OUTPUT_CONTRACT
    )
    user_prompt = f"""
SEEDED CASE
- Template ID: {seeded_bundle.template_id}
- Archetype: {seeded_bundle.archetype}
- Encounter context: {seeded_bundle.encounter_context}

CONDITION CONSTRAINTS
{chr(10).join(condition_constraint_lines)}

SUPPORT VERIFIER RESULT
- Outcome: {condition_support_verification_outcome.outcome}
- Under-supported conditions: {condition_support_verification_outcome.under_supported_conditions}
- Unsupported implied conditions: {condition_support_verification_outcome.unsupported_implied_conditions}
- History/negation drift: {condition_support_verification_outcome.history_or_negation_drift_detected}
- Notes: {condition_support_verification_outcome.verifier_notes}

GENERATED CLINICAL NOTE
{generated_clinical_note.note_text}

TASK
Return ONLY a JSON object with exactly this structure. Every key shown is required.
Score meanings: 0 = fails criterion, 1 = partially meets, 2 = fully meets.
Use null for icd_constraint_alignment_scores criteria not applicable to this case.
icd_constraint_violations must be [] if there are no violations.

{{
  "general_quality_rubric_scores": {{
    "condition_support_coverage":      {{"score": 0, "rationale": "..."}},
    "internal_consistency":            {{"score": 0, "rationale": "..."}},
    "clinical_realism":                {{"score": 0, "rationale": "..."}},
    "encounter_structure_quality":     {{"score": 0, "rationale": "..."}},
    "evidence_specificity":            {{"score": 0, "rationale": "..."}},
    "distractor_handling":             {{"score": 0, "rationale": "..."}},
    "assessment_to_plan_linkage":      {{"score": 0, "rationale": "..."}},
    "language_naturalness":            {{"score": 0, "rationale": "..."}},
    "diversity_contribution":          {{"score": 0, "rationale": "..."}},
    "training_utility":                {{"score": 0, "rationale": "..."}}
  }},
  "icd_constraint_alignment_scores": {{
    "specificity_alignment":                   {{"score": 0, "rationale": "..."}} or null,
    "laterality_alignment":                    {{"score": 0, "rationale": "..."}} or null,
    "encounter_stage_alignment":               {{"score": 0, "rationale": "..."}} or null,
    "temporal_state_alignment":                {{"score": 0, "rationale": "..."}} or null,
    "with_without_complication_alignment":     {{"score": 0, "rationale": "..."}} or null,
    "chapter_style_alignment":                 {{"score": 0, "rationale": "..."}} or null,
    "must_not_imply_compliance":               {{"score": 0, "rationale": "..."}} or null
  }},
  "icd_constraint_violations": [
    {{
      "violated_constraint_type": "...",
      "violation_severity": "advisory|major|critical",
      "what_was_expected": "...",
      "what_was_observed_in_note": "...",
      "fix_instruction_for_revision_prompt": "...",
      "source_icd_code": "..."
    }}
  ]
}}
""".strip()

    return PromptSpec(
        prompt_id=RUBRIC_JUDGE_PROMPT_ID,
        prompt_version=RUBRIC_JUDGE_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=RUBRIC_JUDGE_RESPONSE_SCHEMA,
    )
