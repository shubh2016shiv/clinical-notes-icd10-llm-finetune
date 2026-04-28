"""
Decision combiner for the clinical note evaluation stack.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
    ConstraintViolationSeverity,
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
        icd_constraint_violations: list[IcdConstraintViolationDetail] | None = None,
        rubric_judge_prompt_id: str | None = None,
        rubric_judge_prompt_version: str | None = None,
    ) -> NoteEvaluationCritiqueResult:
        icd_constraint_violations = icd_constraint_violations or []

        if not deterministic_precheck_outcome.passed():
            return NoteEvaluationCritiqueResult(
                deterministic_precheck_outcome=deterministic_precheck_outcome,
                condition_support_verification_outcome=None,
                general_quality_rubric_scores=None,
                icd_constraint_alignment_scores=None,
                icd_constraint_violations=[],
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

        final_decision = self._select_final_decision(
            combined_score=combined_score,
            hard_fail_reasons=hard_fail_reasons,
            revision_targets=revision_targets,
        )

        return NoteEvaluationCritiqueResult(
            deterministic_precheck_outcome=deterministic_precheck_outcome,
            condition_support_verification_outcome=condition_support_verification_outcome,
            general_quality_rubric_scores=general_quality_rubric_scores,
            icd_constraint_alignment_scores=icd_constraint_alignment_scores,
            icd_constraint_violations=icd_constraint_violations,
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
    ) -> str:
        if hard_fail_reasons:
            return "reject"

        if combined_score >= self._accept_threshold:
            return "accept"
        if combined_score >= self._revise_threshold:
            return "revise"
        return "reject"
