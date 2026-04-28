"""
Evaluation data models.

Covers every type produced or consumed by the evaluation stack:

  Layer 1 — Deterministic pre-checks
    DeterministicPreCheckOutcome

  Layer 2 — Support verification (does the note express the bundle?)
    ConditionSupportVerificationOutcome

  Layer 3 — Rubric-based LLM judge
    RubricCriterionEvaluation
    NoteGeneralQualityRubricScores          (10 general criteria)
    IcdConstraintAlignmentRubricScores      (up to 7 metadata criteria)
    IcdConstraintViolationDetail

  Combined critique result
    NoteEvaluationCritiqueResult            (all three layers merged)

  Revision loop types
    RevisionAttemptRecord
    AcceptedClinicalNoteResult
    RejectedClinicalNoteResult

  Pipeline batch metrics
    PipelineBatchRunMetrics

Dependency chain (no circular imports):
  bundle.py -> constraints.py -> note.py -> evaluation.py
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .bundle import SeededClinicalBundle
from .constraints import ClinicalBundleSemanticConstraints, ConstraintViolationSeverity
from .note import GeneratedClinicalNote


# ===========================================================================
# Layer 1 — Deterministic pre-checks
# ===========================================================================


class DeterministicPreCheckOutcome(BaseModel):
    """
    Result of the cheap, fast pre-checks that run before any LLM evaluation.

    These checks catch obvious failures without spending LLM tokens.
    A hard_fail here skips the support verifier and rubric judge entirely.

    Checks performed (all implemented in core/services/deterministic_checks.py):
      - note is non-empty and above a minimum character length
      - note does not contain literal ICD code strings (e.g. 'E11.9')
      - note contains the required structural sections (HPI, Assessment, Plan)
      - note is not a near-duplicate of a recently accepted note (cosine > 0.92)
      - note does not overuse repeated boilerplate phrases

    Fields
    ------
    outcome
        'pass' — note proceeds to support verification.
        'hard_fail' — note is rejected immediately without further evaluation.
    failure_reasons
        Human-readable descriptions of every check that failed.
        Empty list when outcome is 'pass'.
    """

    outcome: Literal["pass", "hard_fail"]
    failure_reasons: list[str] = Field(default_factory=list)

    def passed(self) -> bool:
        return self.outcome == "pass"


# ===========================================================================
# Layer 2 — Support verification
# ===========================================================================


class ConditionSupportVerificationOutcome(BaseModel):
    """
    Result of checking whether the generated note adequately expresses
    every intended active condition from the seeded bundle.

    This is a required gate — not a soft score.  A fail here does not
    prevent revision, but a note that fails twice is rejected.

    Fields
    ------
    outcome
        'pass' — every intended condition is adequately expressed.
        'fail' — one or more intended conditions are absent, only historical,
                 negated, or so weakly expressed that they would not support
                 the intended ICD code in a real coding workflow.
    under_supported_conditions
        Conditions that are present in the note but lack enough clinical
        evidence (mentioned only in passing, no exam finding, no symptom).
    unsupported_implied_conditions
        Extra active conditions the note implies that are NOT in the seeded
        bundle.  These are unintended positive signals.
    history_or_negation_drift_detected
        True when one or more intended active conditions appear in the note
        but read as historical ('history of …') or negated ('denies …')
        rather than active.
    verifier_notes
        Free-text explanation from the verifier for use in the revision prompt.
    """

    outcome: Literal["pass", "fail"]
    under_supported_conditions: list[str] = Field(default_factory=list)
    unsupported_implied_conditions: list[str] = Field(default_factory=list)
    history_or_negation_drift_detected: bool = False
    verifier_notes: str = ""
    verifier_prompt_id: str | None = None
    verifier_prompt_version: str | None = None

    def passed(self) -> bool:
        return self.outcome == "pass"


# ===========================================================================
# Layer 3 — Rubric-based LLM judge
# ===========================================================================


class RubricCriterionEvaluation(BaseModel):
    """
    Score and rationale for one rubric criterion.

    Scale:
      0 = fail — the note clearly does not meet this criterion
      1 = partial — the note partially meets the criterion but has gaps
      2 = strong — the note clearly and convincingly meets this criterion

    Fields
    ------
    score
        0, 1, or 2.
    rationale
        A short explanation of why this score was assigned.
        Used to populate revision_targets in the critique result.
    """

    score: int = Field(ge=0, le=2)
    rationale: str
    is_missing: bool = False


# ---------------------------------------------------------------------------
# General quality rubric (10 criteria — always scored)
# ---------------------------------------------------------------------------


class NoteGeneralQualityRubricScores(BaseModel):
    """
    Scores for the 10 general clinical note quality criteria.

    These judge whether the note is a good clinical note independent of
    ICD-code-specific nuance.  Every note is scored on all 10 criteria.

    Criteria
    --------
    condition_support_coverage
        Does the note provide enough textual evidence for each intended
        active condition?
    internal_consistency
        Is the note internally coherent — no contradictions between
        history, exam findings, assessment, and plan?
    clinical_realism
        Does the note resemble a plausible real-world clinical note for
        the encounter type?
    encounter_structure_quality
        Does the note contain a believable clinical workflow structure
        (CC, HPI, ROS, Exam, Assessment, Plan)?
    evidence_specificity
        Are diagnoses supported by note-specific evidence (findings, labs,
        imaging) rather than generic label restatements?
    distractor_handling
        Are negated, historical, or uncertain conditions included in a
        controlled and believable way — clearly separated from active ones?
    assessment_to_plan_linkage
        Do the plan elements logically follow from the assessment?
    language_naturalness
        Does the prose sound like a clinician note rather than a
        paraphrased ontology entry or code description?
    diversity_contribution
        Does this note add meaningful variety to the dataset in phrasing,
        structure, or encounter style relative to recently accepted notes?
    training_utility
        Would this note be useful as supervision text for a model learning
        note-to-code mapping?
    """

    condition_support_coverage: RubricCriterionEvaluation
    internal_consistency: RubricCriterionEvaluation
    clinical_realism: RubricCriterionEvaluation
    encounter_structure_quality: RubricCriterionEvaluation
    evidence_specificity: RubricCriterionEvaluation
    distractor_handling: RubricCriterionEvaluation
    assessment_to_plan_linkage: RubricCriterionEvaluation
    language_naturalness: RubricCriterionEvaluation
    diversity_contribution: RubricCriterionEvaluation
    training_utility: RubricCriterionEvaluation

    def raw_total(self) -> int:
        """Sum of all 10 scores (max 20)."""
        return sum(
            [
                self.condition_support_coverage.score,
                self.internal_consistency.score,
                self.clinical_realism.score,
                self.encounter_structure_quality.score,
                self.evidence_specificity.score,
                self.distractor_handling.score,
                self.assessment_to_plan_linkage.score,
                self.language_naturalness.score,
                self.diversity_contribution.score,
                self.training_utility.score,
            ]
        )

    def normalized_score(self) -> float:
        """raw_total / 20.0, giving a value in [0.0, 1.0]."""
        return self.raw_total() / 20.0

    def has_any_hard_fail_criterion(self) -> bool:
        """
        True when any hard-fail criterion has a genuine score of 0.

        Hard-fail criteria per the architecture (section 11):
          condition_support_coverage, internal_consistency, clinical_realism,
          training_utility.

        Criteria with is_missing=True are skipped — a missing evaluation is
        not a genuine quality failure and is handled separately upstream.
        """
        hard_fail_criteria = [
            self.condition_support_coverage,
            self.internal_consistency,
            self.clinical_realism,
            self.training_utility,
        ]
        return any(c.score == 0 and not c.is_missing for c in hard_fail_criteria)

    def criteria_scoring_zero(self) -> list[str]:
        """Returns field names of every criterion with a genuine score of 0."""
        return [
            name
            for name, evaluation in self.__dict__.items()
            if isinstance(evaluation, RubricCriterionEvaluation)
            and evaluation.score == 0
            and not evaluation.is_missing
        ]


# ---------------------------------------------------------------------------
# ICD constraint alignment rubric (up to 7 criteria — only applicable ones scored)
# ---------------------------------------------------------------------------


class IcdConstraintAlignmentRubricScores(BaseModel):
    """
    Scores for the metadata alignment criteria — whether the note respects
    the semantic obligations derived from the seeded ICD codes.

    Only applicable subcriteria are scored; non-applicable ones are None.
    This prevents unfair scoring across very different note types (e.g.
    a chronic metabolic note should not lose points for missing laterality
    because it has no laterality-sensitive code).

    Criteria
    --------
    specificity_alignment
        Does the note support the seeded specificity level without
        accidentally implying a different subtype?
        None when no unspecified/specific distinction applies.
    laterality_alignment
        If the seeded code implies right, left, or bilateral involvement,
        does the note match it clearly?
        None when no laterality-sensitive code is in the bundle.
    encounter_stage_alignment
        If the code implies initial, subsequent, or sequela encounter,
        does the note narrative match that stage?
        None when no encounter-stage-sensitive code is in the bundle.
    temporal_state_alignment
        If the code implies acute, chronic, recurrent, in-remission, or
        history state, does the note reflect that time-course correctly?
        None when no temporal-state-sensitive code is in the bundle.
    with_without_complication_alignment
        If the code includes 'with' or 'without' semantics, does the note
        include or exclude the corresponding complication appropriately?
        None when no with/without constraints exist.
    chapter_style_alignment
        Does the note structure and evidence style match the encounter
        expectations implied by the code family or chapter?
        None when chapter family is 'other' or ambiguous.
    must_not_imply_compliance
        Does the note avoid implying things the semantic constraint
        extractor marked as forbidden drift?
        None when no must-not-imply constraints exist.
    """

    specificity_alignment: RubricCriterionEvaluation | None = None
    laterality_alignment: RubricCriterionEvaluation | None = None
    encounter_stage_alignment: RubricCriterionEvaluation | None = None
    temporal_state_alignment: RubricCriterionEvaluation | None = None
    with_without_complication_alignment: RubricCriterionEvaluation | None = None
    chapter_style_alignment: RubricCriterionEvaluation | None = None
    must_not_imply_compliance: RubricCriterionEvaluation | None = None

    def applicable_criteria(self) -> list[RubricCriterionEvaluation]:
        """Returns all criteria that were scored (not None)."""
        return [
            v
            for v in [
                self.specificity_alignment,
                self.laterality_alignment,
                self.encounter_stage_alignment,
                self.temporal_state_alignment,
                self.with_without_complication_alignment,
                self.chapter_style_alignment,
                self.must_not_imply_compliance,
            ]
            if v is not None
        ]

    def normalized_score(self) -> float | None:
        """
        Score as a fraction of max possible for applicable criteria only.
        Returns None when no criteria apply (e.g. a simple unspecified chronic note
        with no laterality, no encounter stage, no with/without).
        """
        applicable = self.applicable_criteria()
        if not applicable:
            return None
        return sum(c.score for c in applicable) / (len(applicable) * 2.0)

    def has_any_critical_criterion_scoring_zero(
        self,
        critical_criterion_names: list[str],
    ) -> bool:
        """
        True when any criterion in critical_criterion_names scored 0.

        Used by the score combiner to enforce hard-fail on critical metadata
        violations before computing the combined score.
        """
        field_map = {
            "specificity_alignment": self.specificity_alignment,
            "laterality_alignment": self.laterality_alignment,
            "encounter_stage_alignment": self.encounter_stage_alignment,
            "temporal_state_alignment": self.temporal_state_alignment,
            "with_without_complication_alignment": self.with_without_complication_alignment,
            "chapter_style_alignment": self.chapter_style_alignment,
            "must_not_imply_compliance": self.must_not_imply_compliance,
        }
        for name in critical_criterion_names:
            criterion = field_map.get(name)
            if criterion is not None and criterion.score == 0:
                return True
        return False


# ---------------------------------------------------------------------------
# Individual constraint violation (machine-readable, for revision loop)
# ---------------------------------------------------------------------------


class IcdConstraintViolationDetail(BaseModel):
    """
    A specific violation of a code-derived semantic constraint detected by
    the rubric judge or the support verifier.

    Carries a machine-readable fix instruction so the revision loop can
    pass concrete, targeted corrections to the note generator rather than
    sending vague feedback.

    Fields
    ------
    violated_constraint_type
        The type of constraint that was violated:
        'laterality', 'encounter_stage', 'temporal_state',
        'with_complication', 'without_complication', 'specificity',
        'must_not_imply'.
    violation_severity
        CRITICAL / IMPORTANT / ADVISORY — determines whether this violation
        alone forces a hard fail or triggers revision.
    what_was_expected
        What the constraint required, e.g. 'left-sided involvement'.
    what_was_observed_in_note
        What the note actually said, e.g. 'right wrist pain'.
    fix_instruction_for_revision_prompt
        A concrete, actionable instruction to include in the revision prompt,
        e.g. 'Change all findings and assessment references to left-sided
        involvement only; remove any mention of right-sided injury.'
    source_icd_code
        The ICD code whose metadata produced this constraint, for traceability.
    """

    violated_constraint_type: str
    violation_severity: ConstraintViolationSeverity
    what_was_expected: str
    what_was_observed_in_note: str
    fix_instruction_for_revision_prompt: str
    source_icd_code: str


# ===========================================================================
# Combined critique result — all three evaluation layers merged
# ===========================================================================


class NoteEvaluationCritiqueResult(BaseModel):
    """
    The complete structured output of the evaluation stack for one generated note.

    Merges all three evaluation layers (deterministic pre-checks, support
    verifier, rubric judge) into a single object that drives the revision
    loop and populates the audit artifact.

    Fields
    ------
    deterministic_precheck_outcome
        Layer 1 result.  If hard_fail, all rubric fields below are None.
    condition_support_verification_outcome
        Layer 2 result.  If fail, general and alignment scores are still
        computed so the rubric can inform the revision targets.
    general_quality_rubric_scores
        Layer 3 general quality scores.  None only when deterministic
        pre-checks hard-failed (no point scoring an obviously bad note).
    icd_constraint_alignment_scores
        Layer 3 metadata alignment scores.  Same None condition.
    icd_constraint_violations
        Machine-readable list of every constraint violation the rubric
        judge detected.  Used by the revision loop to build targeted prompts.
    hard_fail_reasons
        Plain-text reasons for any hard fail, drawn from deterministic
        pre-checks, support verifier, or critical constraint violations.
    revision_targets
        Concrete, targeted items to send back to the note generator if the
        decision is 'revise'.  Drawn from rubric rationale fields and
        constraint violations.
    combined_score
        The weighted final score: 0.6 * normalized_general + 0.4 * normalized_metadata.
        None when hard-failed (score is meaningless in that case).
    final_decision
        'accept'  — note passes all gates and scores above the threshold.
        'revise'  — note is close to valid; send revision_targets back.
        'reject'  — note is fundamentally wrong or too artificial to fix.
    """

    deterministic_precheck_outcome: DeterministicPreCheckOutcome
    condition_support_verification_outcome: ConditionSupportVerificationOutcome | None = None
    general_quality_rubric_scores: NoteGeneralQualityRubricScores | None = None
    icd_constraint_alignment_scores: IcdConstraintAlignmentRubricScores | None = None
    icd_constraint_violations: list[IcdConstraintViolationDetail] = Field(default_factory=list)
    hard_fail_reasons: list[str] = Field(default_factory=list)
    revision_targets: list[str] = Field(default_factory=list)
    combined_score: float | None = None
    final_decision: Literal["accept", "revise", "reject"]
    rubric_judge_prompt_id: str | None = None
    rubric_judge_prompt_version: str | None = None

    def was_hard_failed(self) -> bool:
        return bool(self.hard_fail_reasons)

    def critical_violations(self) -> list[IcdConstraintViolationDetail]:
        return [
            v
            for v in self.icd_constraint_violations
            if v.violation_severity == ConstraintViolationSeverity.CRITICAL
        ]


# ===========================================================================
# Revision loop types
# ===========================================================================


class RevisionAttemptRecord(BaseModel):
    """
    Records everything that happened during a single revision attempt.

    Stored in revision_history on both AcceptedClinicalNoteResult and
    RejectedClinicalNoteResult so the audit artifact has a full trace
    of what was sent, what came back, and how it was evaluated.

    Fields
    ------
    revision_attempt_number
        1 = first revision, 2 = second (and final) revision.
    revision_targets_sent_to_generator
        The plain-text revision targets extracted from the previous critique.
    icd_violations_sent_to_generator
        The structured ICD constraint violations sent alongside the targets.
    revised_note
        The note produced by the revision call.
    critique_after_revision
        The full evaluation critique run on the revised note.
    """

    revision_attempt_number: int = Field(ge=1, le=2)
    revision_targets_sent_to_generator: list[str]
    icd_violations_sent_to_generator: list[IcdConstraintViolationDetail]
    revised_note: GeneratedClinicalNote
    critique_after_revision: NoteEvaluationCritiqueResult


class AcceptedClinicalNoteResult(BaseModel):
    """
    Final output for a note that passed the evaluation stack.

    Contains everything needed to write both the training artifact
    (minimal) and the audit artifact (full provenance).

    Fields
    ------
    seeded_bundle
        The bundle that defined the case.
    bundle_note_writing_constraints
        The constraints derived from the bundle.
    accepted_note
        The final accepted note text and its generation metadata.
    final_critique
        The critique result that produced the 'accept' decision.
    revision_history
        All revision attempts that occurred before acceptance.
        Empty when the note was accepted on first generation.
    required_revision
        False when accepted on first generation, True otherwise.
    """

    seeded_bundle: SeededClinicalBundle
    bundle_note_writing_constraints: ClinicalBundleSemanticConstraints
    accepted_note: GeneratedClinicalNote
    final_critique: NoteEvaluationCritiqueResult
    revision_history: list[RevisionAttemptRecord] = Field(default_factory=list)
    required_revision: bool = False


class RejectedClinicalNoteResult(BaseModel):
    """
    Final output for a note that failed evaluation and was not accepted.

    Preserved in the audit artifact so rejection patterns can be analysed
    to improve prompts and the evaluation stack.

    Fields
    ------
    seeded_bundle
        The bundle that defined the (failed) case.
    rejected_note
        The last generated note text at the point of rejection.
        Present for every rejection — including first-pass hard fails — so
        the failed artifact can always be inspected alongside its critique.
    final_critique
        The full evaluation critique that produced the 'reject' decision.
        Contains the hard_fail_reasons, all rubric scores, and constraint
        violations that explain why the note was not accepted.
    primary_rejection_reason
        A short, human-readable statement of the main reason for rejection.
    all_rejection_reasons
        Every rejection reason collected across all attempts.
    revision_history
        All revision attempts that were made before final rejection.
        Empty when rejected on first generation without revision.
    """

    seeded_bundle: SeededClinicalBundle
    rejected_note: GeneratedClinicalNote
    final_critique: NoteEvaluationCritiqueResult
    primary_rejection_reason: str
    all_rejection_reasons: list[str] = Field(default_factory=list)
    revision_history: list[RevisionAttemptRecord] = Field(default_factory=list)


# ===========================================================================
# Pipeline batch metrics
# ===========================================================================


class PipelineBatchRunMetrics(BaseModel):
    """
    Aggregate quality and throughput metrics across a full pipeline batch run.

    Designed so that iterative prompt improvements show up clearly:
    track mean_combined_score_by_prompt_version, most_frequent_failing_rubric_criteria,
    and revision_success_rate over time to guide prompt and model tuning.

    Fields
    ------
    total_attempted
        Total number of bundle templates attempted in this run.
    total_accepted
        Notes that reached 'accept' decision (including after revision).
    total_rejected
        Notes that reached 'reject' decision.
    acceptance_rate
        total_accepted / total_attempted.
    revision_rate
        Fraction of accepted notes that required at least one revision.
    revision_success_rate
        Of all notes that entered the revision loop, the fraction ultimately
        accepted (not rejected after revision).
    average_combined_score_for_accepted_notes
        Mean combined score (0.0–1.0 scale) across accepted notes.
    average_general_quality_score_for_accepted_notes
        Mean normalized general quality score across accepted notes.
    average_icd_alignment_score_for_accepted_notes
        Mean normalized ICD constraint alignment score across accepted notes.
    average_combined_score_by_archetype
        Mean combined score broken out by archetype string.
    average_combined_score_by_template_id
        Mean combined score broken out by bundle template_id.
    deterministic_precheck_hard_fail_rate
        Fraction of attempts that failed the deterministic pre-check gate.
    support_verifier_fail_rate
        Fraction of attempts that failed the support verification gate.
    near_duplicate_rejection_rate
        Fraction of attempts rejected specifically by the near-duplicate gate.
    icd_code_leakage_rate
        Fraction of attempts rejected because the note contained literal
        ICD code strings.
    most_frequent_failing_rubric_criteria
        Ordered list of rubric criterion names that most frequently scored 0,
        for identifying systematic weaknesses in the generation prompt.
    most_frequent_rejection_reasons
        Dict mapping rejection reason strings to their counts.
    mean_combined_score_by_prompt_version
        Mean combined score broken out by generation_prompt_version,
        for tracking prompt improvement over iterations.
    """

    total_attempted: int
    total_accepted: int
    total_rejected: int
    acceptance_rate: float
    revision_rate: float
    revision_success_rate: float
    average_combined_score_for_accepted_notes: float
    average_general_quality_score_for_accepted_notes: float
    average_icd_alignment_score_for_accepted_notes: float | None
    average_combined_score_by_archetype: dict[str, float] = Field(default_factory=dict)
    average_combined_score_by_template_id: dict[str, float] = Field(default_factory=dict)
    deterministic_precheck_hard_fail_rate: float
    support_verifier_fail_rate: float
    near_duplicate_rejection_rate: float
    icd_code_leakage_rate: float
    most_frequent_failing_rubric_criteria: list[str] = Field(default_factory=list)
    most_frequent_rejection_reasons: dict[str, int] = Field(default_factory=dict)
    mean_combined_score_by_prompt_version: dict[str, float] = Field(default_factory=dict)
