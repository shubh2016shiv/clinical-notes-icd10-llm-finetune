"""
Decision combiner for the clinical note evaluation stack.
"""

from __future__ import annotations

from typing import Literal

from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
    ConstraintViolationSeverity,
)
from clinical_note_generation_v3.core.models.icd_adjudication import (
    FinalIcdCodeAdjudicationOutcome,
)
from clinical_note_generation_v3.core.models.evaluation import (
    ConditionSupportVerificationOutcome,
    DeterministicPreCheckOutcome,
    IcdConstraintAlignmentRubricScores,
    IcdConstraintViolationDetail,
    NoteEvaluationCritiqueResult,
    NoteGeneralQualityRubricScores,
)


class NoteQualityDecisionCombiner:
    """
    Combines evaluation layers into a single accept, revise, or reject decision.
    """

    def __init__(
        self,
        *,
        accept_threshold: float = 0.85,
        revise_threshold: float = 0.60,
        reject_below_threshold: float = 0.45,
    ) -> None:
        self._accept_threshold = accept_threshold
        self._revise_threshold = revise_threshold
        self._reject_below_threshold = reject_below_threshold

    def combine_evaluation_results(
        self,
        *,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        deterministic_precheck_outcome: DeterministicPreCheckOutcome,
        condition_support_verification_outcome: ConditionSupportVerificationOutcome,
        general_quality_rubric_scores: NoteGeneralQualityRubricScores | None,
        icd_constraint_alignment_scores: IcdConstraintAlignmentRubricScores | None,
        icd_adjudication_outcome: FinalIcdCodeAdjudicationOutcome | None = None,
        icd_constraint_violations: list[IcdConstraintViolationDetail] | None = None,
        rubric_judge_prompt_id: str | None = None,
        rubric_judge_prompt_version: str | None = None,
    ) -> NoteEvaluationCritiqueResult:
        icd_constraint_violations = icd_constraint_violations or []

        if not deterministic_precheck_outcome.passed():
            recoverable, unrecoverable = _partition_precheck_failures(
                deterministic_precheck_outcome.failure_reasons
            )
            if recoverable and not unrecoverable:
                # Formatting and lexical defects can usually be fixed by a targeted rewrite.
                return NoteEvaluationCritiqueResult(
                    deterministic_precheck_outcome=deterministic_precheck_outcome,
                    condition_support_verification_outcome=None,
                    general_quality_rubric_scores=None,
                    icd_constraint_alignment_scores=None,
                    icd_constraint_violations=[],
                    icd_adjudication_outcome=None,
                    hard_fail_reasons=[],
                    revision_targets=_build_precheck_revision_targets(recoverable),
                    combined_score=None,
                    final_decision="revise",
                    rubric_judge_prompt_id=rubric_judge_prompt_id,
                    rubric_judge_prompt_version=rubric_judge_prompt_version,
                )
            return NoteEvaluationCritiqueResult(
                deterministic_precheck_outcome=deterministic_precheck_outcome,
                condition_support_verification_outcome=None,
                general_quality_rubric_scores=None,
                icd_constraint_alignment_scores=None,
                icd_constraint_violations=[],
                icd_adjudication_outcome=None,
                hard_fail_reasons=list(deterministic_precheck_outcome.failure_reasons),
                revision_targets=[],
                combined_score=None,
                final_decision="reject",
                rubric_judge_prompt_id=rubric_judge_prompt_id,
                rubric_judge_prompt_version=rubric_judge_prompt_version,
            )

        if general_quality_rubric_scores is None or icd_constraint_alignment_scores is None:
            return NoteEvaluationCritiqueResult(
                deterministic_precheck_outcome=deterministic_precheck_outcome,
                condition_support_verification_outcome=condition_support_verification_outcome,
                general_quality_rubric_scores=general_quality_rubric_scores,
                icd_constraint_alignment_scores=icd_constraint_alignment_scores,
                icd_constraint_violations=icd_constraint_violations,
                icd_adjudication_outcome=icd_adjudication_outcome,
                hard_fail_reasons=["Rubric judging did not produce complete results."],
                revision_targets=["Re-run rubric evaluation with a complete structured output."],
                combined_score=None,
                final_decision="reject",
                rubric_judge_prompt_id=rubric_judge_prompt_id,
                rubric_judge_prompt_version=rubric_judge_prompt_version,
            )

        general_quality_score = general_quality_rubric_scores.normalized_score()
        metadata_alignment_score = icd_constraint_alignment_scores.normalized_score()
        effective_metadata_alignment_score = (
            metadata_alignment_score if metadata_alignment_score is not None else 1.0
        )
        combined_score = (0.6 * general_quality_score) + (0.4 * effective_metadata_alignment_score)

        hard_fail_reasons: list[str] = []
        revision_targets = self._build_revision_targets(
            condition_support_verification_outcome=condition_support_verification_outcome,
            general_quality_rubric_scores=general_quality_rubric_scores,
            icd_constraint_alignment_scores=icd_constraint_alignment_scores,
            icd_constraint_violations=icd_constraint_violations,
        )

        critical_metadata_violation_detected = any(
            violation.violation_severity == ConstraintViolationSeverity.CRITICAL
            for violation in icd_constraint_violations
        )
        if critical_metadata_violation_detected:
            hard_fail_reasons.append("Critical ICD metadata constraint violation detected.")

        if general_quality_rubric_scores.has_any_hard_fail_criterion():
            hard_fail_reasons.append("One or more hard-fail general quality criteria scored zero.")

        if not condition_support_verification_outcome.passed():
            hard_fail_reasons.append("Condition support verification failed.")

        icd_adjudication_passed = bool(
            icd_adjudication_outcome is not None and icd_adjudication_outcome.passed()
        )
        if not icd_adjudication_passed:
            revision_targets.extend(
                _build_icd_adjudication_revision_targets(icd_adjudication_outcome)
            )

        final_decision = self._select_final_decision(
            combined_score=combined_score,
            hard_fail_reasons=hard_fail_reasons,
            revision_targets=revision_targets,
            icd_adjudication_passed=icd_adjudication_passed,
        )
        if final_decision == "reject" and not icd_adjudication_passed:
            hard_fail_reasons.append("Final ICD-10-CM adjudication failed.")

        return NoteEvaluationCritiqueResult(
            deterministic_precheck_outcome=deterministic_precheck_outcome,
            condition_support_verification_outcome=condition_support_verification_outcome,
            general_quality_rubric_scores=general_quality_rubric_scores,
            icd_constraint_alignment_scores=icd_constraint_alignment_scores,
            icd_constraint_violations=icd_constraint_violations,
            icd_adjudication_outcome=icd_adjudication_outcome,
            hard_fail_reasons=hard_fail_reasons,
            revision_targets=revision_targets,
            combined_score=combined_score,
            final_decision=final_decision,
            rubric_judge_prompt_id=rubric_judge_prompt_id,
            rubric_judge_prompt_version=rubric_judge_prompt_version,
        )

    def _build_revision_targets(
        self,
        *,
        condition_support_verification_outcome: ConditionSupportVerificationOutcome,
        general_quality_rubric_scores: NoteGeneralQualityRubricScores,
        icd_constraint_alignment_scores: IcdConstraintAlignmentRubricScores,
        icd_constraint_violations: list[IcdConstraintViolationDetail],
    ) -> list[str]:
        revision_targets: list[str] = []

        if condition_support_verification_outcome.under_supported_conditions:
            revision_targets.append(
                "Strengthen support for these seeded conditions: "
                + ", ".join(condition_support_verification_outcome.under_supported_conditions)
            )

        if condition_support_verification_outcome.unsupported_implied_conditions:
            revision_targets.append(
                "Remove or weaken unsupported extra active conditions: "
                + ", ".join(condition_support_verification_outcome.unsupported_implied_conditions)
            )

        if condition_support_verification_outcome.history_or_negation_drift_detected:
            revision_targets.append(
                "Ensure seeded active conditions read as active rather than historical or negated."
            )

        for criterion_name, criterion_evaluation in general_quality_rubric_scores.__dict__.items():
            if hasattr(criterion_evaluation, "score") and criterion_evaluation.score < 2:
                revision_targets.append(
                    f"Improve {criterion_name.replace('_', ' ')}: {criterion_evaluation.rationale}"
                )

        for (
            criterion_name,
            criterion_evaluation,
        ) in icd_constraint_alignment_scores.__dict__.items():
            if criterion_evaluation is not None and criterion_evaluation.score < 2:
                revision_targets.append(
                    f"Improve {criterion_name.replace('_', ' ')}: {criterion_evaluation.rationale}"
                )

        for icd_constraint_violation in icd_constraint_violations:
            if icd_constraint_violation.fix_instruction_for_revision_prompt:
                revision_targets.append(
                    icd_constraint_violation.fix_instruction_for_revision_prompt
                )

        return list(dict.fromkeys(revision_targets))

    def _select_final_decision(
        self,
        *,
        combined_score: float,
        hard_fail_reasons: list[str],
        revision_targets: list[str],
        icd_adjudication_passed: bool = True,
    ) -> Literal["accept", "revise", "reject"]:
        if hard_fail_reasons:
            return "reject"

        if not icd_adjudication_passed:
            if combined_score >= self._revise_threshold and revision_targets:
                return "revise"
            return "reject"

        if combined_score >= self._accept_threshold:
            return "accept"
        if combined_score >= self._revise_threshold:
            return "revise"
        return "reject"


def _build_icd_adjudication_revision_targets(
    icd_adjudication_outcome: FinalIcdCodeAdjudicationOutcome | None,
) -> list[str]:
    if icd_adjudication_outcome is None:
        return ["Run final ICD-10-CM adjudication before accepting this note."]
    return list(icd_adjudication_outcome.revision_targets)


_RECOVERABLE_PRECHECK_PREFIXES: tuple[str, ...] = (
    "Clinical note is missing required structural sections",
    "Clinical note appears to copy an official ICD description too literally",
)


def _partition_precheck_failures(
    failure_reasons: list[str],
) -> tuple[list[str], list[str]]:
    """
    Split deterministic pre-check failure reasons into recoverable and unrecoverable.

    Recoverable failures (missing structural sections) are formatting defects that a
    single targeted revision can fix.  All other failures — ICD code leakage, note too
    short, near-duplicate, repeated boilerplate — are unrecoverable: a re-prompt of the
    same model against the same bundle is unlikely to produce a fundamentally different
    result, and allowing revision would waste API calls.
    """
    recoverable: list[str] = []
    unrecoverable: list[str] = []
    for reason in failure_reasons:
        if any(reason.startswith(prefix) for prefix in _RECOVERABLE_PRECHECK_PREFIXES):
            recoverable.append(reason)
        else:
            unrecoverable.append(reason)
    return recoverable, unrecoverable


def _build_precheck_revision_targets(recoverable_reasons: list[str]) -> list[str]:
    """
    Convert recoverable deterministic failures into concrete revision targets.
    """
    targets = list(recoverable_reasons)
    if any(
        reason.startswith("Clinical note is missing required structural sections")
        for reason in recoverable_reasons
    ):
        targets.append(
            "Rewrite the note ensuring all required sections are present with clear headings: "
            "Chief Complaint, History of Present Illness (HPI), Past Medical History (PMH), "
            "Medications, Allergies, Physical Examination, Assessment, Plan. "
            "Each section must begin on its own line with a labeled heading."
        )
    if any(
        reason.startswith("Clinical note appears to copy an official ICD description too literally")
        for reason in recoverable_reasons
    ):
        targets.append(
            "Rewrite diagnosis and problem-list wording so it does not copy official ICD-10-CM "
            "short descriptions verbatim. Preserve the same clinical meaning, but use natural "
            "clinician documentation and evidence-forward language instead of ontology wording."
        )
    return targets
