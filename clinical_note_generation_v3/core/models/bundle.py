"""
Bundle data models.

These are the earliest types in the pipeline:
  ClinicalBundleTemplate  ->  SeededClinicalBundle

The bundle is defined before any note is written and must remain frozen
through note generation, constraint extraction, and evaluation.
"""

from pydantic import BaseModel, Field


class ClinicalBundleTemplate(BaseModel):
    """
    A reusable clinical scenario template that declares what conditions
    belong in a case before ICD codes are resolved.

    This is the input to the ICD resolution layer — it is NOT a note
    and it does NOT contain ICD codes yet.

    Fields
    ------
    template_id
        Unique, stable identifier used to track which template produced a note.
    archetype
        High-level encounter style, e.g. 'chronic_care_followup',
        'acute_injury_initial', 'acute_injury_subsequent',
        'behavioral_health_outpatient', 'pediatric_respiratory_acute'.
    complexity_tier
        1 = single straightforward condition,
        2 = moderate comorbidity bundle (2–3 conditions),
        3 = complex multi-system bundle (4+ conditions or tricky interactions).
    active_conditions
        Human-readable condition names that will be resolved to ICD codes.
        These become the seeded active conditions in the generated note.
    encounter_context
        A short description of the encounter setting, e.g.
        'outpatient follow-up', 'emergency department visit',
        'inpatient admission day 1'.
    allowed_distractors
        Conditions that SHOULD appear in the note as negated, historical,
        or ruled-out findings — not as active diagnoses.
    trap_patterns
        Explicit prohibitions for the note generator, e.g.
        'do not imply diabetic nephropathy unless it is in active_conditions'.
    """

    template_id: str
    archetype: str
    complexity_tier: int = Field(ge=1, le=3)
    active_conditions: list[str]
    encounter_context: str
    allowed_distractors: list[str] = Field(default_factory=list)
    trap_patterns: list[str] = Field(default_factory=list)


class ResolvedConditionCode(BaseModel):
    """
    One active condition from the bundle after ICD resolution.

    Pairs the human-readable condition name with its official ICD-10-CM
    code and full description so downstream components (constraint extractor,
    note generator, evaluator) all work from the same resolved record.

    Fields
    ------
    condition_name
        The original human-readable name from the bundle template,
        e.g. 'type 2 diabetes mellitus'.
    icd_code
        The resolved official ICD-10-CM code, e.g. 'E11.9'.
    icd_short_description
        Short description from the official order file,
        e.g. 'Type 2 diabetes mellitus without complications'.
    icd_long_description
        Full description from the official order file when available.
    """

    condition_name: str
    icd_code: str
    icd_short_description: str
    icd_long_description: str
    resolver_prompt_id: str | None = None
    resolver_prompt_version: str | None = None
    source_condition_names: list[str] = Field(default_factory=list)


class SeededClinicalBundle(BaseModel):
    """
    A ClinicalBundleTemplate after ICD codes have been resolved for every
    active condition.

    This is the fixed, read-only input to all downstream pipeline steps:
    constraint extraction, note generation, support verification, and
    rubric evaluation.

    IMPORTANT: once this object is created, nothing downstream may modify
    active_conditions, resolved_conditions, encounter_context, or any
    other field. The bundle defines the case; the note generator only
    renders it.

    Fields
    ------
    template_id
        Preserved from the source template for traceability.
    archetype
        Preserved from the source template.
    encounter_context
        Preserved from the source template.
    active_condition_names
        Flat list of human-readable condition names for quick reference
        without unpacking resolved_conditions.
    resolved_conditions
        The authoritative list of conditions with their official ICD codes.
        Every note generator and evaluator must treat this as the ground truth.
    allowed_distractors
        Preserved from the source template.
    trap_patterns
        Preserved from the source template.
    """

    template_id: str
    archetype: str
    encounter_context: str
    active_condition_names: list[str]
    resolved_conditions: list[ResolvedConditionCode]
    allowed_distractors: list[str] = Field(default_factory=list)
    trap_patterns: list[str] = Field(default_factory=list)

    @property
    def icd_codes(self) -> list[str]:
        """All resolved ICD codes in this bundle, in order."""
        return [entry.icd_code for entry in self.resolved_conditions]


# Backward-compatible aliases for earlier naming used during the transition.
ResolvedConditionWithIcdCode = ResolvedConditionCode
IcdSeededClinicalBundle = SeededClinicalBundle
