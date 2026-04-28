"""
ICD code semantic constraint models.

The constraint extractor converts each resolved ICD code + its official
metadata description into structured obligations and prohibitions for the
note generator.  The same constraint objects are then passed unchanged
into the support verifier and rubric judge so evaluation stays anchored
to the same semantic requirements as generation.

Dependency chain (no circular imports):
  bundle.py  ->  constraints.py
"""

from enum import Enum

from pydantic import BaseModel, Field

from .bundle import SeededClinicalBundle


# ---------------------------------------------------------------------------
# Enums used across constraint types
# ---------------------------------------------------------------------------


class ConstraintViolationSeverity(str, Enum):
    """
    How seriously a constraint violation should affect the evaluation decision.

    CRITICAL  - Forces a hard fail regardless of rubric scores.
                Examples: laterality mismatch in an injury note,
                encounter-stage mismatch for encounter-sensitive codes,
                'with/without' contradiction.
    IMPORTANT - Usually triggers revision. The note is recoverable but
                not acceptable in current form.
                Examples: temporal-state weakness (chronic note reads acute),
                chapter-style weakness (injury note missing mechanism).
    ADVISORY  - Informs scoring and revision targets but does not block
                acceptance on its own.
                Examples: stylistic awkwardness, minor specificity ambiguity.
    """

    CRITICAL = "critical"
    IMPORTANT = "important"
    ADVISORY = "advisory"


class ConstraintExtractionSource(str, Enum):
    """
    Which tier of the constraint extractor produced a given signal.

    DETERMINISTIC_REGEX   - Matched by a hard-coded regex pattern on the
                            ICD description text.  Highest confidence.
    FAMILY_SPECIFIC_RULE  - Matched by a chapter/family rule pack that knows
                            domain-specific conventions (injury, endocrine, …).
    LLM_ESCALATION        - Resolved by an LLM call for cases where
                            deterministic patterns conflict or are absent.
                            Carries lower default confidence and must be
                            logged for audit.
    """

    DETERMINISTIC_REGEX = "deterministic_regex"
    FAMILY_SPECIFIC_RULE = "family_specific_rule"
    LLM_ESCALATION = "llm_escalation"


# ---------------------------------------------------------------------------
# Individual signal extracted from one ICD description
# ---------------------------------------------------------------------------


class ExtractedSemanticSignal(BaseModel):
    """
    One semantic signal pulled from an ICD code's official description.

    Multiple signals combine to form an IcdCodeNoteWritingConstraints object.
    The extraction provenance (source, confidence, matched text) is preserved
    so the constraint extractor can be audited and debugged.

    Fields
    ------
    signal_type
        What kind of semantic signal this is:
        'laterality', 'encounter_type', 'temporal_state',
        'with_complication', 'without_complication', 'is_unspecified'.
    extracted_value
        The resolved value of the signal, e.g. 'left', 'initial encounter',
        'chronic', 'diabetic nephropathy', 'unspecified'.
    confidence
        0.0 to 1.0.  Deterministic regex hits are typically 0.95–1.0;
        LLM-escalated signals are typically 0.70–0.85.
    extraction_source
        Which extractor tier produced this signal.
    matched_text_span
        The exact substring from the ICD description that triggered this
        signal, for traceability.  None when produced by LLM escalation
        without a single clear anchor span.
    """

    signal_type: str
    extracted_value: str
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_source: ConstraintExtractionSource
    matched_text_span: str | None = None


# ---------------------------------------------------------------------------
# Per-code constraint set (one per resolved ICD code in a bundle)
# ---------------------------------------------------------------------------


class IcdCodeSemanticConstraints(BaseModel):
    """
    All note-writing obligations and prohibitions derived from a single
    resolved ICD code and its official metadata.

    Tells the note generator exactly what the note MUST express and MUST
    avoid for this specific code.  The same object is passed to the
    support verifier and rubric judge so evaluation uses the same
    semantic requirements as generation.

    Fields
    ------
    icd_code
        The resolved code this constraint set was derived from.
    icd_short_description
        Short description from the official file, preserved for context.
    laterality
        'right', 'left', 'bilateral', or None.
        If set, the note MUST state laterality clearly and correctly.
    encounter_type
        'initial', 'subsequent', 'sequela', or None.
        Drives the narrative stage of the note (fresh presentation vs.
        follow-up vs. residual effects).
    temporal_states
        One or more of: 'acute', 'chronic', 'recurrent', 'in_remission',
        'history'.  The note must match the implied time-course.
    severity_qualifiers
        Severity descriptors from the description, e.g. 'moderate', 'severe'.
        The note should reflect this severity level.
    with_complication_flags
        Complications the note MUST include clinical evidence for
        (from 'with X' in the description).
    without_complication_flags
        Complications the note MUST NOT imply (from 'without X').
    is_unspecified_code
        True when the description contains 'unspecified'.  The note must
        avoid accidentally providing enough specificity to justify a more
        precise subtype code.
    chapter_family
        The clinical chapter family that determines note-structure style:
        'injury', 'endocrine_metabolic', 'behavioral_health',
        'respiratory_infectious', 'neoplasm', 'other'.
    must_include_in_note
        Concrete clinical elements the note must contain, derived from the
        combination of the signals above, e.g.
        'left-sided symptoms or exam findings',
        'fracture confirmation or imaging evidence'.
    must_not_imply_in_note
        Things the note must not state or imply, e.g.
        'right-sided injury', 'follow-up healing context',
        'psychotic features unless explicitly seeded'.
    all_extracted_signals
        Full audit trail of every signal the extractor found, in the order
        they were extracted.
    """

    icd_code: str
    icd_short_description: str
    laterality: str | None = None
    encounter_type: str | None = None
    temporal_states: list[str] = Field(default_factory=list)
    severity_qualifiers: list[str] = Field(default_factory=list)
    with_complication_flags: list[str] = Field(default_factory=list)
    without_complication_flags: list[str] = Field(default_factory=list)
    is_unspecified_code: bool = False
    chapter_family: str = "other"
    must_include_in_note: list[str] = Field(default_factory=list)
    must_not_imply_in_note: list[str] = Field(default_factory=list)
    all_extracted_signals: list[ExtractedSemanticSignal] = Field(default_factory=list)
    constraint_extractor_prompt_id: str | None = None
    constraint_extractor_prompt_version: str | None = None

    def has_laterality_constraint(self) -> bool:
        return self.laterality is not None

    def has_encounter_type_constraint(self) -> bool:
        return self.encounter_type is not None

    def has_temporal_constraint(self) -> bool:
        return len(self.temporal_states) > 0

    def has_with_without_constraint(self) -> bool:
        return bool(self.with_complication_flags or self.without_complication_flags)


# ---------------------------------------------------------------------------
# Bundle-level constraint set (one per full IcdSeededClinicalBundle)
# ---------------------------------------------------------------------------


class ClinicalBundleSemanticConstraints(BaseModel):
    """
    The complete set of note-writing constraints for an entire clinical bundle.

    This is the single object passed unchanged through note generation,
    support verification, and rubric evaluation.  It aggregates the
    per-code constraints and exposes convenience accessors so each
    downstream component does not have to iterate per_code_constraints
    manually.

    Fields
    ------
    seeded_bundle
        The IcdSeededClinicalBundle this constraint set was derived from.
        Preserved here so the evaluator always has the full bundle context.
    per_code_note_writing_constraints
        One IcdCodeNoteWritingConstraints per resolved condition in the bundle.
        Ordered to match seeded_bundle.resolved_conditions.
    """

    seeded_bundle: SeededClinicalBundle
    per_code_note_writing_constraints: list[IcdCodeSemanticConstraints]

    def all_must_include_items(self) -> list[str]:
        """Flat list of every must-include item across all codes."""
        return [
            item
            for code_constraints in self.per_code_note_writing_constraints
            for item in code_constraints.must_include_in_note
        ]

    def all_must_not_imply_items(self) -> list[str]:
        """Flat list of every must-not-imply item across all codes."""
        return [
            item
            for code_constraints in self.per_code_note_writing_constraints
            for item in code_constraints.must_not_imply_in_note
        ]

    def codes_with_laterality_constraints(self) -> list[IcdCodeSemanticConstraints]:
        """Returns only the per-code constraints that carry laterality requirements."""
        return [c for c in self.per_code_note_writing_constraints if c.has_laterality_constraint()]

    def codes_with_encounter_type_constraints(self) -> list[IcdCodeSemanticConstraints]:
        """Returns only the per-code constraints that carry encounter-type requirements."""
        return [
            c for c in self.per_code_note_writing_constraints if c.has_encounter_type_constraint()
        ]

    def codes_with_with_without_constraints(self) -> list[IcdCodeSemanticConstraints]:
        """Returns only the per-code constraints that carry with/without complication requirements."""
        return [
            c for c in self.per_code_note_writing_constraints if c.has_with_without_constraint()
        ]

    def dominant_chapter_family(self) -> str:
        """
        Returns the chapter family that appears most often across codes in the bundle.
        Used to pick note-structure style guidance when the bundle spans multiple families.
        Falls back to 'other' when the bundle is mixed with no clear majority.
        """
        from collections import Counter

        counts = Counter(
            c.chapter_family
            for c in self.per_code_note_writing_constraints
            if c.chapter_family != "other"
        )
        if not counts:
            return "other"
        return counts.most_common(1)[0][0]


# Backward-compatible aliases for earlier naming used during the transition.
IcdCodeNoteWritingConstraints = IcdCodeSemanticConstraints
ClinicalBundleNoteWritingConstraints = ClinicalBundleSemanticConstraints
