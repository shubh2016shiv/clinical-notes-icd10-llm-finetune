"""
Centralized prompt specifications for the v3 pipeline.
"""

from .registry import (
    build_clinical_note_generation_prompt_spec,
    build_clinical_note_revision_prompt_spec,
    build_condition_support_verifier_prompt_spec,
    build_constraint_extraction_prompt_spec,
    build_icd_resolution_prompt_spec,
    build_rubric_judge_prompt_spec,
)

__all__ = [
    "build_clinical_note_generation_prompt_spec",
    "build_clinical_note_revision_prompt_spec",
    "build_condition_support_verifier_prompt_spec",
    "build_constraint_extraction_prompt_spec",
    "build_icd_resolution_prompt_spec",
    "build_rubric_judge_prompt_spec",
]
