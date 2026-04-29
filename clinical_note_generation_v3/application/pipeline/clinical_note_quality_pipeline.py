"""
End-to-end clinical note quality pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from clinical_note_generation_v3.application.bundle_planner.clinical_bundle_template_sampler import (
    ClinicalBundleTemplateSampler,
)
from clinical_note_generation_v3.application.constraint_extraction.constraint_extraction_orchestrator import (
    BundleConstraintExtractionOrchestrator,
)
from clinical_note_generation_v3.application.evaluation.clinical_note_revision_loop import (
    ClinicalNoteRevisionLoop,
)
from clinical_note_generation_v3.application.evaluation.clinical_note_rubric_judge import (
    ClinicalNoteRubricJudge,
)
from clinical_note_generation_v3.application.evaluation.condition_support_verifier import (
    ConditionSupportVerifier,
)
from clinical_note_generation_v3.application.evaluation.note_quality_decision_combiner import (
    NoteQualityDecisionCombiner,
)
from clinical_note_generation_v3.application.icd_adjudication.final_icd_code_adjudicator import (
    FinalIcdCodeAdjudicator,
)
from clinical_note_generation_v3.application.icd_resolution.icd_condition_to_code_resolver import (
    IcdConditionToCodeResolver,
    IcdResolutionFailedForConditionError,
)
from clinical_note_generation_v3.application.note_generation.seeded_clinical_note_generator import (
    SeededClinicalNoteGenerator,
)
from clinical_note_generation_v3.artifacts.audit_artifact_writer import AuditArtifactWriter
from clinical_note_generation_v3.artifacts.batch_metrics_writer import BatchMetricsWriter
from clinical_note_generation_v3.artifacts.training_artifact_writer import TrainingArtifactWriter
from clinical_note_generation_v3.core.models.bundle import ClinicalBundleTemplate
from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
)
from clinical_note_generation_v3.core.models.evaluation import (
    AcceptedClinicalNoteResult,
    ConditionSupportVerificationOutcome,
    NoteEvaluationCritiqueResult,
    PipelineBatchRunMetrics,
    RejectedClinicalNoteResult,
)
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote
from clinical_note_generation_v3.core.services.deterministic_precheck_runner import (
    DeterministicPreCheckRunner,
)
from clinical_note_generation_v3.core.services.icd_code_set_validator import (
    IcdCodeSetValidationError,
    IcdCodeSetValidator,
)

logger = logging.getLogger(__name__)


class ClinicalNoteQualityPipeline:
    """
    Orchestrates the full bundle-seeded clinical note quality workflow.
    """

    def __init__(
        self,
        *,
        clinical_bundle_template_sampler: ClinicalBundleTemplateSampler,
        icd_condition_to_code_resolver: IcdConditionToCodeResolver,
        bundle_constraint_extraction_orchestrator: BundleConstraintExtractionOrchestrator,
        seeded_clinical_note_generator: SeededClinicalNoteGenerator,
        deterministic_precheck_runner: DeterministicPreCheckRunner,
        condition_support_verifier: ConditionSupportVerifier,
        clinical_note_rubric_judge: ClinicalNoteRubricJudge,
        final_icd_code_adjudicator: FinalIcdCodeAdjudicator,
        icd_code_set_validator: IcdCodeSetValidator,
        note_quality_decision_combiner: NoteQualityDecisionCombiner,
        clinical_note_revision_loop: ClinicalNoteRevisionLoop,
        training_artifact_writer: TrainingArtifactWriter,
        audit_artifact_writer: AuditArtifactWriter,
        batch_metrics_writer: BatchMetricsWriter,
        training_artifact_output_path: Path,
        audit_artifact_output_path: Path,
        batch_metrics_output_path: Path,
        recent_accepted_note_window_size: int = 50,
    ) -> None:
        self._clinical_bundle_template_sampler = clinical_bundle_template_sampler
        self._icd_condition_to_code_resolver = icd_condition_to_code_resolver
        self._bundle_constraint_extraction_orchestrator = bundle_constraint_extraction_orchestrator
        self._seeded_clinical_note_generator = seeded_clinical_note_generator
        self._deterministic_precheck_runner = deterministic_precheck_runner
        self._condition_support_verifier = condition_support_verifier
        self._clinical_note_rubric_judge = clinical_note_rubric_judge
        self._final_icd_code_adjudicator = final_icd_code_adjudicator
        self._icd_code_set_validator = icd_code_set_validator
        self._note_quality_decision_combiner = note_quality_decision_combiner
        self._clinical_note_revision_loop = clinical_note_revision_loop
        self._training_artifact_writer = training_artifact_writer
        self._audit_artifact_writer = audit_artifact_writer
        self._batch_metrics_writer = batch_metrics_writer
        self._training_artifact_output_path = training_artifact_output_path
        self._audit_artifact_output_path = audit_artifact_output_path
        self._batch_metrics_output_path = batch_metrics_output_path
        self._recent_accepted_note_window_size = recent_accepted_note_window_size
        self._recently_used_template_ids: list[str] = []
        self._recent_accepted_note_texts: list[str] = []

    def run_single_template_through_pipeline(
        self,
        *,
        clinical_bundle_template: ClinicalBundleTemplate | None = None,
        write_artifacts: bool = True,
    ) -> AcceptedClinicalNoteResult | RejectedClinicalNoteResult:
        """
        Run the full pipeline for one sampled or provided bundle template.
        """
        selected_clinical_bundle_template = (
            clinical_bundle_template
            if clinical_bundle_template is not None
            else self._clinical_bundle_template_sampler.select_next_template(
                recently_used_template_ids=self._recently_used_template_ids
            )
        )
        self._recently_used_template_ids.append(selected_clinical_bundle_template.template_id)

        seeded_clinical_bundle = (
            self._icd_condition_to_code_resolver.resolve_seeded_clinical_bundle_from_template(
                selected_clinical_bundle_template
            )
        )
        seeded_clinical_bundle = self._icd_code_set_validator.collapse_seeded_bundle_codes(
            seeded_clinical_bundle
        )
        self._icd_code_set_validator.assert_valid_codes(seeded_clinical_bundle.icd_codes)

        bundle_semantic_constraints = (
            self._bundle_constraint_extraction_orchestrator.extract_bundle_semantic_constraints(
                seeded_clinical_bundle
            )
        )

        generated_clinical_note = self._seeded_clinical_note_generator.generate_clinical_note(
            bundle_semantic_constraints=bundle_semantic_constraints,
            generation_attempt_number=1,
        )

        initial_note_evaluation_result = self._evaluate_generated_clinical_note(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
        )

        clinical_note_result = self._clinical_note_revision_loop.run_revision_loop(
            bundle_semantic_constraints=bundle_semantic_constraints,
            initial_generated_clinical_note=generated_clinical_note,
            initial_note_evaluation_result=initial_note_evaluation_result,
            evaluate_generated_clinical_note=lambda revised_clinical_note: self._evaluate_generated_clinical_note(
                generated_clinical_note=revised_clinical_note,
                bundle_semantic_constraints=bundle_semantic_constraints,
            ),
        )

        if isinstance(clinical_note_result, AcceptedClinicalNoteResult):
            self._record_recent_accepted_note_text(clinical_note_result.accepted_note.note_text)
            if write_artifacts:
                self._training_artifact_writer.write_accepted_note_training_rows(
                    accepted_clinical_note_results=[clinical_note_result],
                    output_file_path=self._training_artifact_output_path,
                    append=True,
                )

        if write_artifacts:
            self._audit_artifact_writer.write_clinical_note_audit_rows(
                clinical_note_results=[clinical_note_result],
                output_file_path=self._audit_artifact_output_path,
                append=True,
            )

        return clinical_note_result

    def run_batch_generation_pipeline(
        self,
        *,
        requested_example_count: int,
        write_artifacts: bool = True,
    ) -> tuple[
        list[AcceptedClinicalNoteResult | RejectedClinicalNoteResult],
        PipelineBatchRunMetrics,
    ]:
        """
        Run a batch of pipeline examples and optionally write aggregate artifacts.
        """
        clinical_note_results: list[AcceptedClinicalNoteResult | RejectedClinicalNoteResult] = []
        accepted_clinical_note_results: list[AcceptedClinicalNoteResult] = []

        maximum_sampling_attempts = max(requested_example_count * 3, requested_example_count)
        sampling_attempt_count = 0

        while (
            len(clinical_note_results) < requested_example_count
            and sampling_attempt_count < maximum_sampling_attempts
        ):
            sampling_attempt_count += 1
            try:
                clinical_note_result = self.run_single_template_through_pipeline(
                    write_artifacts=False
                )
            except IcdResolutionFailedForConditionError as icd_resolution_error:
                logger.warning(
                    "Skipping sampled bundle after ICD resolution failure: %s",
                    icd_resolution_error,
                )
                continue
            except IcdCodeSetValidationError as code_set_validation_error:
                logger.warning(
                    "Skipping sampled bundle after ICD code-set validation failure: %s",
                    code_set_validation_error,
                )
                continue
            except Exception as pipeline_iteration_error:
                logger.warning(
                    "Skipping pipeline iteration after unexpected error: %s",
                    pipeline_iteration_error,
                )
                continue

            clinical_note_results.append(clinical_note_result)
            self._print_console_evaluation_metrics(
                clinical_note_result=clinical_note_result,
                attempt_number=sampling_attempt_count,
            )
            if isinstance(clinical_note_result, AcceptedClinicalNoteResult):
                accepted_clinical_note_results.append(clinical_note_result)

        pipeline_batch_run_metrics = self._build_pipeline_batch_run_metrics(
            clinical_note_results=clinical_note_results
        )

        if write_artifacts:
            self._training_artifact_writer.write_accepted_note_training_rows(
                accepted_clinical_note_results=accepted_clinical_note_results,
                output_file_path=self._training_artifact_output_path,
                append=True,
            )
            self._audit_artifact_writer.write_clinical_note_audit_rows(
                clinical_note_results=clinical_note_results,
                output_file_path=self._audit_artifact_output_path,
                append=True,
            )
            self._batch_metrics_writer.write_pipeline_batch_metrics(
                pipeline_batch_run_metrics=pipeline_batch_run_metrics,
                output_file_path=self._batch_metrics_output_path,
            )

        return clinical_note_results, pipeline_batch_run_metrics

    def run_single_pipeline_iteration(
        self,
        *,
        clinical_bundle_template: ClinicalBundleTemplate | None = None,
        write_artifacts: bool = True,
    ) -> AcceptedClinicalNoteResult | RejectedClinicalNoteResult:
        return self.run_single_template_through_pipeline(
            clinical_bundle_template=clinical_bundle_template,
            write_artifacts=write_artifacts,
        )

    def run_pipeline_batch(
        self,
        *,
        requested_example_count: int,
        write_artifacts: bool = True,
    ) -> tuple[
        list[AcceptedClinicalNoteResult | RejectedClinicalNoteResult],
        PipelineBatchRunMetrics,
    ]:
        return self.run_batch_generation_pipeline(
            requested_example_count=requested_example_count,
            write_artifacts=write_artifacts,
        )

    def _evaluate_generated_clinical_note(
        self,
        *,
        generated_clinical_note: GeneratedClinicalNote,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    ):
        deterministic_precheck_outcome = self._deterministic_precheck_runner.run_prechecks(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
            recent_accepted_note_texts=self._recent_accepted_note_texts,
        )

        if not deterministic_precheck_outcome.passed():
            return self._note_quality_decision_combiner.combine_evaluation_results(
                bundle_semantic_constraints=bundle_semantic_constraints,
                deterministic_precheck_outcome=deterministic_precheck_outcome,
                condition_support_verification_outcome=ConditionSupportVerificationOutcome(
                    outcome="fail",
                    verifier_notes="Skipped because deterministic pre-checks hard-failed.",
                ),
                general_quality_rubric_scores=None,
                icd_constraint_alignment_scores=None,
                icd_adjudication_outcome=None,
                icd_constraint_violations=[],
            )

        condition_support_verification_outcome = (
            self._condition_support_verifier.verify_condition_support(
                generated_clinical_note=generated_clinical_note,
                bundle_semantic_constraints=bundle_semantic_constraints,
            )
        )

        (
            general_quality_rubric_scores,
            icd_constraint_alignment_scores,
            icd_constraint_violations,
            rubric_judge_prompt_id,
            rubric_judge_prompt_version,
        ) = self._clinical_note_rubric_judge.judge_generated_clinical_note(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
            condition_support_verification_outcome=condition_support_verification_outcome,
        )
        icd_adjudication_outcome = self._final_icd_code_adjudicator.adjudicate_generated_note(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
        )

        return self._note_quality_decision_combiner.combine_evaluation_results(
            bundle_semantic_constraints=bundle_semantic_constraints,
            deterministic_precheck_outcome=deterministic_precheck_outcome,
            condition_support_verification_outcome=condition_support_verification_outcome,
            general_quality_rubric_scores=general_quality_rubric_scores,
            icd_constraint_alignment_scores=icd_constraint_alignment_scores,
            icd_adjudication_outcome=icd_adjudication_outcome,
            icd_constraint_violations=icd_constraint_violations,
            rubric_judge_prompt_id=rubric_judge_prompt_id,
            rubric_judge_prompt_version=rubric_judge_prompt_version,
        )

    def _record_recent_accepted_note_text(self, accepted_note_text: str) -> None:
        self._recent_accepted_note_texts.append(accepted_note_text)
        if len(self._recent_accepted_note_texts) > self._recent_accepted_note_window_size:
            self._recent_accepted_note_texts = self._recent_accepted_note_texts[
                -self._recent_accepted_note_window_size :
            ]

    def _print_console_evaluation_metrics(
        self,
        *,
        clinical_note_result: AcceptedClinicalNoteResult | RejectedClinicalNoteResult,
        attempt_number: int,
    ) -> None:
        critique = clinical_note_result.final_critique
        support_outcome = (
            critique.condition_support_verification_outcome.outcome
            if critique.condition_support_verification_outcome is not None
            else "not_run"
        )
        general_score = (
            critique.general_quality_rubric_scores.normalized_score()
            if critique.general_quality_rubric_scores is not None
            else None
        )
        icd_alignment_score = (
            critique.icd_constraint_alignment_scores.normalized_score()
            if critique.icd_constraint_alignment_scores is not None
            else None
        )
        adjudication_outcome = critique.icd_adjudication_outcome
        adjudication_label = adjudication_outcome.outcome if adjudication_outcome else "not_run"
        adjudicated_codes = (
            ",".join(adjudication_outcome.adjudicated_icd10_codes)
            if adjudication_outcome
            else "n/a"
        )
        seeded_delta = _format_seeded_code_delta(adjudication_outcome)
        code_set_status = (
            "pass"
            if adjudication_outcome and adjudication_outcome.code_set_validation_outcome.passed()
            else "fail"
            if adjudication_outcome
            else "not_run"
        )
        revision_count = len(clinical_note_result.revision_history)
        print(
            "[Evaluation] "
            f"attempt={attempt_number} "
            f"template={clinical_note_result.seeded_bundle.template_id} "
            f"archetype={clinical_note_result.seeded_bundle.archetype} "
            f"deterministic={critique.deterministic_precheck_outcome.outcome} "
            f"support={support_outcome} "
            f"general={_format_optional_score(general_score)} "
            f"icd_alignment={_format_optional_score(icd_alignment_score)} "
            f"adjudication={adjudication_label} "
            f"codes={adjudicated_codes} "
            f"seeded_delta={seeded_delta} "
            f"icd_rules={code_set_status} "
            f"combined={_format_optional_score(critique.combined_score)} "
            f"decision={critique.final_decision} "
            f"revisions={revision_count}"
        )

    def _build_pipeline_batch_run_metrics(
        self,
        *,
        clinical_note_results: list[AcceptedClinicalNoteResult | RejectedClinicalNoteResult],
    ) -> PipelineBatchRunMetrics:
        total_attempted = len(clinical_note_results)
        accepted_results = [
            result
            for result in clinical_note_results
            if isinstance(result, AcceptedClinicalNoteResult)
        ]
        rejected_results = [
            result
            for result in clinical_note_results
            if isinstance(result, RejectedClinicalNoteResult)
        ]

        total_accepted = len(accepted_results)
        total_rejected = len(rejected_results)

        acceptance_rate = (total_accepted / total_attempted) if total_attempted else 0.0
        revision_rate = (
            sum(1 for result in accepted_results if result.required_revision) / total_accepted
            if total_accepted
            else 0.0
        )

        notes_entering_revision_count = sum(
            1 for result in accepted_results if result.required_revision
        ) + sum(1 for result in rejected_results if result.revision_history)
        revision_success_rate = (
            sum(1 for result in accepted_results if result.required_revision)
            / notes_entering_revision_count
            if notes_entering_revision_count
            else 0.0
        )

        accepted_combined_scores = [
            result.final_critique.combined_score
            for result in accepted_results
            if result.final_critique.combined_score is not None
        ]
        accepted_general_scores = [
            result.final_critique.general_quality_rubric_scores.normalized_score()
            for result in accepted_results
            if result.final_critique.general_quality_rubric_scores is not None
        ]
        accepted_icd_alignment_scores = [
            result.final_critique.icd_constraint_alignment_scores.normalized_score()
            for result in accepted_results
            if (
                result.final_critique.icd_constraint_alignment_scores is not None
                and result.final_critique.icd_constraint_alignment_scores.normalized_score()
                is not None
            )
        ]

        average_combined_score_by_archetype = self._build_average_score_by_archetype(
            accepted_results
        )
        average_combined_score_by_template_id = self._build_average_score_by_template_id(
            accepted_results
        )

        deterministic_precheck_hard_fail_count = 0
        support_verifier_fail_count = 0
        near_duplicate_rejection_count = 0
        icd_code_leakage_rejection_count = 0
        icd_adjudication_fail_count = 0
        code_set_validation_fail_count = 0
        rejection_reason_counter: Counter[str] = Counter()
        rubric_zero_counter: Counter[str] = Counter()
        icd_rule_failure_counter: Counter[str] = Counter()
        mean_combined_score_by_prompt_version = self._build_average_score_by_prompt_version(
            accepted_results
        )

        for result in clinical_note_results:
            for critique in self._collect_all_recorded_critiques(result):
                if not critique.deterministic_precheck_outcome.passed():
                    deterministic_precheck_hard_fail_count += 1
                    for failure_reason in critique.deterministic_precheck_outcome.failure_reasons:
                        lowered = failure_reason.lower()
                        if "similar to a recently accepted note" in lowered:
                            near_duplicate_rejection_count += 1
                        if "literal icd code string" in lowered:
                            icd_code_leakage_rejection_count += 1
                if (
                    critique.condition_support_verification_outcome is not None
                    and not critique.condition_support_verification_outcome.passed()
                ):
                    support_verifier_fail_count += 1
                if critique.general_quality_rubric_scores is not None:
                    for (
                        criterion_name
                    ) in critique.general_quality_rubric_scores.criteria_scoring_zero():
                        rubric_zero_counter[criterion_name] += 1
                if critique.icd_adjudication_outcome is not None:
                    if not critique.icd_adjudication_outcome.passed():
                        icd_adjudication_fail_count += 1
                    validation_outcome = (
                        critique.icd_adjudication_outcome.code_set_validation_outcome
                    )
                    if not validation_outcome.passed():
                        code_set_validation_fail_count += 1
                        for issue in validation_outcome.errors():
                            icd_rule_failure_counter[issue.rule_type] += 1

        for rejected_result in rejected_results:
            rejection_reason_counter[rejected_result.primary_rejection_reason] += 1
            for rejection_reason in rejected_result.all_rejection_reasons:
                rejection_reason_counter[rejection_reason] += 1

        return PipelineBatchRunMetrics(
            total_attempted=total_attempted,
            total_accepted=total_accepted,
            total_rejected=total_rejected,
            acceptance_rate=acceptance_rate,
            revision_rate=revision_rate,
            revision_success_rate=revision_success_rate,
            average_combined_score_for_accepted_notes=(
                sum(score for score in accepted_combined_scores if score is not None)
                / len(accepted_combined_scores)
                if accepted_combined_scores
                else 0.0
            ),
            average_general_quality_score_for_accepted_notes=(
                sum(accepted_general_scores) / len(accepted_general_scores)
                if accepted_general_scores
                else 0.0
            ),
            average_icd_alignment_score_for_accepted_notes=(
                sum(accepted_icd_alignment_scores) / len(accepted_icd_alignment_scores)
                if accepted_icd_alignment_scores
                else None
            ),
            average_combined_score_by_archetype=average_combined_score_by_archetype,
            average_combined_score_by_template_id=average_combined_score_by_template_id,
            deterministic_precheck_hard_fail_rate=(
                deterministic_precheck_hard_fail_count / total_attempted if total_attempted else 0.0
            ),
            support_verifier_fail_rate=(
                support_verifier_fail_count / total_attempted if total_attempted else 0.0
            ),
            near_duplicate_rejection_rate=(
                near_duplicate_rejection_count / total_attempted if total_attempted else 0.0
            ),
            icd_code_leakage_rate=(
                icd_code_leakage_rejection_count / total_attempted if total_attempted else 0.0
            ),
            icd_adjudication_fail_rate=(
                icd_adjudication_fail_count / total_attempted if total_attempted else 0.0
            ),
            code_set_validation_fail_rate=(
                code_set_validation_fail_count / total_attempted if total_attempted else 0.0
            ),
            top_icd_rule_failure_types=[
                rule_type for rule_type, _ in icd_rule_failure_counter.most_common()
            ],
            most_frequent_failing_rubric_criteria=[
                criterion_name for criterion_name, _ in rubric_zero_counter.most_common()
            ],
            most_frequent_rejection_reasons=dict(rejection_reason_counter),
            mean_combined_score_by_prompt_version=mean_combined_score_by_prompt_version,
        )

    def _collect_all_recorded_critiques(
        self,
        result: AcceptedClinicalNoteResult | RejectedClinicalNoteResult,
    ) -> list[NoteEvaluationCritiqueResult]:
        """
        Returns every NoteEvaluationCritiqueResult recorded for a result, in order.

        Revision critiques are listed before the final critique. The
        initial-generation critique is not stored separately when revision occurred,
        but that gap is safe for gate-failure counting: hard-fail gates (deterministic
        precheck, support verifier) unconditionally reject rather than revise, so if
        either gate failed on the initial attempt the initial critique IS the final
        critique and is already included.
        """
        critiques: list[NoteEvaluationCritiqueResult] = [
            revision_record.critique_after_revision for revision_record in result.revision_history
        ]
        critiques.append(result.final_critique)
        return critiques

    def _build_average_score_by_archetype(
        self,
        accepted_results: list[AcceptedClinicalNoteResult],
    ) -> dict[str, float]:
        score_totals_by_archetype: dict[str, list[float]] = {}
        for accepted_result in accepted_results:
            if accepted_result.final_critique.combined_score is None:
                continue
            archetype = accepted_result.seeded_bundle.archetype
            score_totals_by_archetype.setdefault(archetype, []).append(
                accepted_result.final_critique.combined_score
            )
        return {
            archetype: sum(score_values) / len(score_values)
            for archetype, score_values in score_totals_by_archetype.items()
            if score_values
        }

    def _build_average_score_by_template_id(
        self,
        accepted_results: list[AcceptedClinicalNoteResult],
    ) -> dict[str, float]:
        score_totals_by_template_id: dict[str, list[float]] = {}
        for accepted_result in accepted_results:
            if accepted_result.final_critique.combined_score is None:
                continue
            template_id = accepted_result.seeded_bundle.template_id
            score_totals_by_template_id.setdefault(template_id, []).append(
                accepted_result.final_critique.combined_score
            )
        return {
            template_id: sum(score_values) / len(score_values)
            for template_id, score_values in score_totals_by_template_id.items()
            if score_values
        }

    def _build_average_score_by_prompt_version(
        self,
        accepted_results: list[AcceptedClinicalNoteResult],
    ) -> dict[str, float]:
        score_totals_by_prompt_version: dict[str, list[float]] = {}
        for accepted_result in accepted_results:
            if accepted_result.final_critique.combined_score is None:
                continue
            prompt_version = accepted_result.accepted_note.generation_prompt_version
            score_totals_by_prompt_version.setdefault(prompt_version, []).append(
                accepted_result.final_critique.combined_score
            )
        return {
            prompt_version: sum(score_values) / len(score_values)
            for prompt_version, score_values in score_totals_by_prompt_version.items()
            if score_values
        }


def _format_optional_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"


def _format_seeded_code_delta(adjudication_outcome) -> str:
    if adjudication_outcome is None:
        return "n/a"
    added = "+[" + ",".join(adjudication_outcome.added_icd10_codes) + "]"
    removed = "-[" + ",".join(adjudication_outcome.removed_seeded_icd10_codes) + "]"
    return f"{added}{removed}"
