"""
Shared prompt contracts and prompt-spec models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PromptSpec(BaseModel):
    """
    A fully rendered prompt ready for deterministic concatenation and submission.
    """

    prompt_id: str
    prompt_version: str
    system_prompt: str
    user_prompt: str
    response_schema: dict | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


JSON_ONLY_OUTPUT_CONTRACT = (
    "Return only a valid JSON object. Do not include markdown fences, preamble, "
    "commentary, or extra keys."
)


CLINICAL_NOTE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "clinical_note_text": {
            "type": "string",
            "description": "The full synthetic clinical note text.",
        }
    },
    "required": ["clinical_note_text"],
}


CONDITION_SUPPORT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string"},
        "under_supported_conditions": {"type": "array", "items": {"type": "string"}},
        "unsupported_implied_conditions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "history_or_negation_drift_detected": {"type": "boolean"},
        "verifier_notes": {"type": "string"},
    },
    "required": [
        "outcome",
        "under_supported_conditions",
        "unsupported_implied_conditions",
        "history_or_negation_drift_detected",
        "verifier_notes",
    ],
}


_RUBRIC_CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
}

_NULLABLE_RUBRIC_CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    # intentionally no "required" — Gemini can omit these for non-applicable criteria
}

RUBRIC_JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "general_quality_rubric_scores": {
            "type": "object",
            "properties": {
                "condition_support_coverage": _RUBRIC_CRITERION_SCHEMA,
                "internal_consistency": _RUBRIC_CRITERION_SCHEMA,
                "clinical_realism": _RUBRIC_CRITERION_SCHEMA,
                "encounter_structure_quality": _RUBRIC_CRITERION_SCHEMA,
                "evidence_specificity": _RUBRIC_CRITERION_SCHEMA,
                "distractor_handling": _RUBRIC_CRITERION_SCHEMA,
                "assessment_to_plan_linkage": _RUBRIC_CRITERION_SCHEMA,
                "language_naturalness": _RUBRIC_CRITERION_SCHEMA,
                "diversity_contribution": _RUBRIC_CRITERION_SCHEMA,
                "training_utility": _RUBRIC_CRITERION_SCHEMA,
            },
            "required": [
                "condition_support_coverage",
                "internal_consistency",
                "clinical_realism",
                "encounter_structure_quality",
                "evidence_specificity",
                "distractor_handling",
                "assessment_to_plan_linkage",
                "language_naturalness",
                "diversity_contribution",
                "training_utility",
            ],
        },
        "icd_constraint_alignment_scores": {
            "type": "object",
            "properties": {
                "specificity_alignment": _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "laterality_alignment": _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "encounter_stage_alignment": _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "temporal_state_alignment": _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "with_without_complication_alignment": _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "chapter_style_alignment": _NULLABLE_RUBRIC_CRITERION_SCHEMA,
                "must_not_imply_compliance": _NULLABLE_RUBRIC_CRITERION_SCHEMA,
            },
        },
        "icd_constraint_violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "violated_constraint_type": {"type": "string"},
                    "violation_severity": {"type": "string"},
                    "what_was_expected": {"type": "string"},
                    "what_was_observed_in_note": {"type": "string"},
                    "fix_instruction_for_revision_prompt": {"type": "string"},
                    "source_icd_code": {"type": "string"},
                },
                "required": [
                    "violated_constraint_type",
                    "violation_severity",
                    "what_was_expected",
                    "what_was_observed_in_note",
                    "fix_instruction_for_revision_prompt",
                    "source_icd_code",
                ],
            },
        },
    },
    "required": [
        "general_quality_rubric_scores",
        "icd_constraint_alignment_scores",
        "icd_constraint_violations",
    ],
}


ICD_SELECTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_icd_code": {"type": "string"},
        "selected_icd_description": {"type": "string"},
        "selection_rationale": {"type": "string"},
        "resolution_succeeded": {"type": "boolean"},
    },
    "required": [
        "selected_icd_code",
        "selected_icd_description",
        "selection_rationale",
        "resolution_succeeded",
    ],
}
# Note: Gemini structured output does not support JSON Schema union types.
# selected_icd_code / selected_icd_description are typed as "string" here;
# treat an empty string as a null/absent value in downstream parsing.


CONSTRAINT_EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "must_include_in_note": {"type": "array", "items": {"type": "string"}},
        "must_not_imply_in_note": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["must_include_in_note", "must_not_imply_in_note"],
}
