"""
Clinical note data models.

Covers everything produced by the seeded note generator:
  GenerationModelInfo            - which LLM provider/model was used
  GeneratedClinicalNote          - the note text plus generation metadata
  ClinicalNoteGenerationAttempt  - bundles the note with the seeded bundle
                                   and constraints that produced it,
                                   plus which attempt number this is

Dependency chain (no circular imports):
  bundle.py -> constraints.py -> note.py
"""

from pydantic import BaseModel, Field

from .bundle import SeededClinicalBundle
from .constraints import ClinicalBundleSemanticConstraints


# ---------------------------------------------------------------------------
# LLM provenance — tracks which model produced a piece of generated text
# ---------------------------------------------------------------------------


class GenerationModelInfo(BaseModel):
    """
    Records which LLM provider and model name produced a generated output.

    Stored on every GeneratedClinicalNote so batch metrics can break down
    quality scores by model and prompt version over time.

    Fields
    ------
    provider_name
        The LLM provider, e.g. 'gemini', 'deepseek', 'openai'.
    model_name
        The specific model identifier used, e.g. 'gemini-2.0-flash',
        'deepseek-chat'.
    """

    provider_name: str
    model_name: str


# ---------------------------------------------------------------------------
# The generated clinical note itself
# ---------------------------------------------------------------------------


class GeneratedClinicalNote(BaseModel):
    """
    A synthetic clinical note produced by the seeded note generator.

    Contains the raw note text plus enough metadata to reproduce the
    generation, track which model was used, and identify the fake PHI
    that was injected.

    Fields
    ------
    note_text
        The full text of the generated clinical note.  This is what the
        evaluation stack reads and what ends up in the training artifact.
    generation_prompt_version
        A short version tag for the prompt template that produced this note,
        e.g. 'seeded_v1', 'seeded_v2_injury_style'.
        Used to compute mean quality score by prompt version in batch metrics.
    generation_model_info
        Which provider and model generated this note.
    fake_patient_name
        The fictitious patient name injected as PHI, e.g. 'Jane Okafor'.
    fake_patient_mrn
        The fictitious MRN injected as PHI, e.g. 'MRN-00847231'.
    fake_patient_date_of_birth
        The fictitious DOB injected as PHI, e.g. '1962-03-15'.
    """

    note_text: str
    generation_prompt_id: str
    generation_prompt_version: str
    generation_model_info: GenerationModelInfo
    fake_patient_name: str
    fake_patient_mrn: str
    fake_patient_date_of_birth: str


# ---------------------------------------------------------------------------
# One generation attempt — note + the bundle and constraints that produced it
# ---------------------------------------------------------------------------


class ClinicalNoteGenerationAttempt(BaseModel):
    """
    The complete record of a single note generation call.

    Bundles the note that was produced with the seeded bundle and
    constraint set that drove generation, plus which attempt number
    this is within the revision loop.

    This object is the input to the evaluation stack (deterministic
    pre-checks, support verifier, rubric judge).  The evaluation stack
    reads the note_text from generated_note, and reads the expected
    constraints from bundle_note_writing_constraints.

    Fields
    ------
    seeded_bundle
        The IcdSeededClinicalBundle that defined the case.
        Preserved here for traceability — evaluation components should
        read constraints from bundle_note_writing_constraints, not
        re-derive them from the bundle.
    bundle_note_writing_constraints
        The full constraint set derived from the seeded bundle.
        This is the single source of truth for what the note must express.
    generated_note
        The note produced in this attempt.
    generation_attempt_number
        1 = initial generation.
        2 = first revision attempt.
        3 = second (and final) revision attempt.
    """

    seeded_bundle: SeededClinicalBundle
    bundle_note_writing_constraints: ClinicalBundleSemanticConstraints
    generated_note: GeneratedClinicalNote
    generation_attempt_number: int = Field(ge=1, le=3)

    def is_initial_generation(self) -> bool:
        return self.generation_attempt_number == 1

    def is_revision_attempt(self) -> bool:
        return self.generation_attempt_number > 1
