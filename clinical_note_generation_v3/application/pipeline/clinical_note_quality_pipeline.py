"""
End-to-end clinical note quality pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

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
from clinical_note_generation_v3.persistence_clinical_notes_postgresql import (
    AcceptedClinicalNotePersistenceError,
    PostgreSqlAcceptedClinicalNotesPersistence,
)
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
from clinical_note_generation_v3.core.exceptions import ClinicalNoteGenerationError
from clinical_note_generation_v3.core.observability import PipelineTraceCollector
from clinical_note_generation_v3.core.services.deterministic_precheck_runner import (
    DeterministicPreCheckRunner,
)
from clinical_note_generation_v3.application.evaluation.icd_compliance_reasoning_reviewer import (
    IcdComplianceReasoningReviewer,
)
from clinical_note_generation_v3.application.reporting.console_progress_reporter import (
    ConsoleProgressReporter,
)
from clinical_note_generation_v3.core.services.icd_code_set_validator import (
    IcdCodeSetValidationError,
    IcdCodeSetValidator,
)

logger = logging.getLogger(__name__)
ClinicalNoteResultT = TypeVar(
    "ClinicalNoteResultT",
    AcceptedClinicalNoteResult,
    RejectedClinicalNoteResult,
)


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
        accepted_note_persistence: PostgreSqlAcceptedClinicalNotesPersistence | None = None,
        accept_threshold_generated_notes: float = 0.97,
        icd_compliance_reasoning_reviewer: IcdComplianceReasoningReviewer | None = None,
        icd_compliance_reviewer_max_regeneration_loops: int = 3,
        log_verbosity: bool = False,
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
        self._accepted_note_persistence = accepted_note_persistence
        self._accept_threshold_generated_notes = accept_threshold_generated_notes
        self._icd_compliance_reasoning_reviewer = icd_compliance_reasoning_reviewer
        self._icd_compliance_reviewer_max_regeneration_loops = (
            icd_compliance_reviewer_max_regeneration_loops
        )
        self._log_verbosity = log_verbosity
        self._recently_used_template_ids: list[str] = []
        self._recent_accepted_note_texts: list[str] = []

    def run_single_template_through_pipeline(
        self,
        *,
        clinical_bundle_template: ClinicalBundleTemplate | None = None,
        write_artifacts: bool = True,
        note_sequence_number: int | None = None,
        total_requested_count: int | None = None,
    ) -> AcceptedClinicalNoteResult | RejectedClinicalNoteResult:
        """
        Run the full pipeline for one sampled or provided bundle template.
        """
        progress_reporter = self._build_progress_reporter(
            note_sequence_number=note_sequence_number,
            total_requested_count=total_requested_count,
        )
        correlation_id = (
            progress_reporter.correlation_id if progress_reporter is not None else str(uuid4())
        )
        trace_collector = PipelineTraceCollector(
            correlation_id=correlation_id,
            enabled=self._log_verbosity,
            logger_name=__name__,
        )
        current_stage_number = 1
        current_stage_name = "Start clinical note generation"

        try:
            if progress_reporter is not None:
                progress_reporter.begin_note()
                progress_reporter.stage(1, current_stage_name)
                progress_reporter.detail("Preparing a new synthetic encounter for generation.")
            self._trace_event(
                trace_collector,
                event_type="stage_started",
                stage_number=1,
                stage_name=current_stage_name,
                payload={"write_artifacts": write_artifacts},
            )

            current_stage_number = 2
            current_stage_name = "Select the next clinical bundle template"
            if progress_reporter is not None:
                progress_reporter.stage(2, current_stage_name)
            selected_clinical_bundle_template = (
                clinical_bundle_template
                if clinical_bundle_template is not None
                else self._clinical_bundle_template_sampler.select_next_template(
                    recently_used_template_ids=self._recently_used_template_ids
                )
            )
            self._recently_used_template_ids.append(selected_clinical_bundle_template.template_id)
            if progress_reporter is not None:
                progress_reporter.detail(
                    "Selected template "
                    f"{selected_clinical_bundle_template.template_id} for "
                    f"{selected_clinical_bundle_template.archetype}."
                )
            self._trace_event(
                trace_collector,
                event_type="stage_passed",
                stage_number=2,
                stage_name=current_stage_name,
                payload={
                    "template_id": selected_clinical_bundle_template.template_id,
                    "archetype": selected_clinical_bundle_template.archetype,
                    "encounter_context": selected_clinical_bundle_template.encounter_context,
                },
            )

            current_stage_number = 3
            current_stage_name = "Resolve ICD-10 codes for the seeded conditions"
            if progress_reporter is not None:
                progress_reporter.stage(
                    3,
                    current_stage_name,
                    model_label=self._icd_condition_to_code_resolver.configured_model_label,
                )
            seeded_clinical_bundle = (
                self._icd_condition_to_code_resolver.resolve_seeded_clinical_bundle_from_template(
                    selected_clinical_bundle_template
                )
            )
            seeded_clinical_bundle = self._icd_code_set_validator.collapse_seeded_bundle_codes(
                seeded_clinical_bundle
            )
            self._icd_code_set_validator.assert_valid_codes(seeded_clinical_bundle.icd_codes)
            if progress_reporter is not None:
                progress_reporter.detail(
                    "Resolved seeded ICD-10 codes: " + ", ".join(seeded_clinical_bundle.icd_codes)
                )
            self._trace_event(
                trace_collector,
                event_type="stage_passed",
                stage_number=3,
                stage_name=current_stage_name,
                payload={
                    "seeded_icd10_codes": list(seeded_clinical_bundle.icd_codes),
                    "resolved_conditions": [
                        {
                            "condition_name": condition.condition_name,
                            "icd_code": condition.icd_code,
                            "icd_short_description": condition.icd_short_description,
                        }
                        for condition in seeded_clinical_bundle.resolved_conditions
                    ],
                },
            )

            current_stage_number = 4
            current_stage_name = "Build note-writing constraints from the resolved bundle"
            if progress_reporter is not None:
                progress_reporter.stage(4, current_stage_name)
            bundle_semantic_constraints = (
                self._bundle_constraint_extraction_orchestrator.extract_bundle_semantic_constraints(
                    seeded_clinical_bundle
                )
            )
            if progress_reporter is not None:
                progress_reporter.detail(
                    "Prepared "
                    f"{len(bundle_semantic_constraints.per_code_note_writing_constraints)} "
                    "code-specific writing constraint records."
                )
            self._trace_event(
                trace_collector,
                event_type="stage_passed",
                stage_number=4,
                stage_name=current_stage_name,
                payload={
                    "constraint_count": len(
                        bundle_semantic_constraints.per_code_note_writing_constraints
                    ),
                },
            )

            current_stage_number = 5
            current_stage_name = "Generate the first clinical note draft"
            if progress_reporter is not None:
                progress_reporter.stage(
                    5,
                    current_stage_name,
                    model_label=self._seeded_clinical_note_generator.configured_model_label,
                )
            generated_clinical_note = self._seeded_clinical_note_generator.generate_clinical_note(
                bundle_semantic_constraints=bundle_semantic_constraints,
                generation_attempt_number=1,
                correlation_id=correlation_id,
            )
            if progress_reporter is not None:
                progress_reporter.detail(
                    "Created the first draft clinical note for this encounter."
                )
            self._trace_event(
                trace_collector,
                event_type="stage_passed",
                stage_number=5,
                stage_name=current_stage_name,
                payload={
                    "note_character_count": len(generated_clinical_note.note_text),
                    "generation_prompt_id": generated_clinical_note.generation_prompt_id,
                    "generation_prompt_version": generated_clinical_note.generation_prompt_version,
                },
            )

            generated_clinical_note = self._run_icd_compliance_review_loop(
                generated_clinical_note=generated_clinical_note,
                bundle_semantic_constraints=bundle_semantic_constraints,
                progress_reporter=progress_reporter,
                trace_collector=trace_collector,
            )

            initial_note_evaluation_result = self._evaluate_generated_clinical_note(
                generated_clinical_note=generated_clinical_note,
                bundle_semantic_constraints=bundle_semantic_constraints,
                progress_reporter=progress_reporter,
                log_stages=True,
                trace_collector=trace_collector,
            )

            current_stage_number = 11
            current_stage_name = "Run the revision loop when the note is fixable"
            if progress_reporter is not None:
                progress_reporter.stage(
                    11,
                    current_stage_name,
                    model_label=self._seeded_clinical_note_generator.configured_model_label,
                )
            clinical_note_result = self._clinical_note_revision_loop.run_revision_loop(
                bundle_semantic_constraints=bundle_semantic_constraints,
                initial_generated_clinical_note=generated_clinical_note,
                initial_note_evaluation_result=initial_note_evaluation_result,
                evaluate_generated_clinical_note=lambda revised_clinical_note: self._evaluate_generated_clinical_note(
                    generated_clinical_note=revised_clinical_note,
                    bundle_semantic_constraints=bundle_semantic_constraints,
                    progress_reporter=progress_reporter,
                    log_stages=False,
                    trace_collector=trace_collector,
                ),
                progress_reporter=progress_reporter,
            )

            current_stage_number = 12
            current_stage_name = "Finalize the decision for this clinical note"
            if progress_reporter is not None:
                progress_reporter.stage(12, current_stage_name)
                if isinstance(clinical_note_result, AcceptedClinicalNoteResult):
                    progress_reporter.final_decision(
                        "Accepted",
                        "The clinical note passed the quality workflow and is accepted.",
                    )
                else:
                    progress_reporter.final_decision(
                        "Rejected",
                        clinical_note_result.primary_rejection_reason,
                    )
            self._trace_event(
                trace_collector,
                event_type="stage_passed",
                stage_number=12,
                stage_name=current_stage_name,
                status=(
                    "pass"
                    if isinstance(clinical_note_result, AcceptedClinicalNoteResult)
                    else "fail"
                ),
                payload={
                    "result_status": (
                        "accepted"
                        if isinstance(clinical_note_result, AcceptedClinicalNoteResult)
                        else "rejected"
                    ),
                    "final_decision": clinical_note_result.final_critique.final_decision,
                    "hard_fail_reasons": list(
                        clinical_note_result.final_critique.hard_fail_reasons
                    ),
                },
            )

            clinical_note_result = self._attach_correlation_id_to_result(
                clinical_note_result=clinical_note_result,
                correlation_id=correlation_id,
            )

            current_stage_number = 13
            current_stage_name = (
                "Write the accepted note to PostgreSQL or explain why it is skipped"
            )
            if progress_reporter is not None:
                progress_reporter.stage(13, current_stage_name)
            if isinstance(clinical_note_result, AcceptedClinicalNoteResult):
                accepted_clinical_note_result = clinical_note_result
                self._record_recent_accepted_note_text(
                    accepted_clinical_note_result.accepted_note.note_text
                )
                accepted_clinical_note_result = self._attach_pipeline_trace_to_result(
                    clinical_note_result=accepted_clinical_note_result,
                    pipeline_trace=trace_collector.snapshot(),
                )
                self._persist_accepted_note_if_eligible(
                    accepted_clinical_note_result,
                    progress_reporter=progress_reporter,
                    trace_collector=trace_collector,
                )
                if write_artifacts:
                    self._training_artifact_writer.write_accepted_note_training_rows(
                        accepted_clinical_note_results=self._training_eligible_results(
                            [accepted_clinical_note_result]
                        ),
                        output_file_path=self._training_artifact_output_path,
                        append=True,
                    )
                clinical_note_result = accepted_clinical_note_result
            else:
                if progress_reporter is not None:
                    progress_reporter.detail("Rejected notes are not written to PostgreSQL.")
                self._trace_event(
                    trace_collector,
                    event_type="stage_passed",
                    stage_number=13,
                    stage_name=current_stage_name,
                    payload={"persistence_action": "skipped_rejected_note"},
                )

            clinical_note_result.pipeline_trace = trace_collector.snapshot()

            if write_artifacts:
                self._audit_artifact_writer.write_clinical_note_audit_rows(
                    clinical_note_results=[clinical_note_result],
                    output_file_path=self._audit_artifact_output_path,
                    append=True,
                )

            if progress_reporter is not None:
                progress_reporter.end_note()

            return clinical_note_result
        except Exception as error:
            self._trace_event(
                trace_collector,
                event_type="pipeline_exception",
                stage_number=current_stage_number,
                stage_name=current_stage_name,
                status="error",
                payload={"error_type": error.__class__.__name__, "error_message": str(error)},
            )
            if isinstance(
                error, (ClinicalNoteGenerationError, AcceptedClinicalNotePersistenceError)
            ):
                raise
            raise ClinicalNoteGenerationError(
                "Unexpected clinical note pipeline failure.",
                correlation_id=correlation_id,
                stage_number=current_stage_number,
                stage_name=current_stage_name,
                details={"error_type": error.__class__.__name__, "error_message": str(error)},
            ) from error
        finally:
            trace_collector.close()

    def run_batch_generation_pipeline(
        self,
        *,
        requested_example_count: int,
        write_artifacts: bool = True,
        show_progress: bool = False,
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
            note_sequence_number = len(clinical_note_results) + 1
            try:
                clinical_note_result = self.run_single_template_through_pipeline(
                    write_artifacts=False,
                    note_sequence_number=(note_sequence_number if show_progress else None),
                    total_requested_count=(requested_example_count if show_progress else None),
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
            except AcceptedClinicalNotePersistenceError:
                raise
            except Exception as pipeline_iteration_error:
                logger.warning(
                    "Skipping pipeline iteration after unexpected error: %s",
                    pipeline_iteration_error,
                )
                continue

            clinical_note_results.append(clinical_note_result)
            if isinstance(clinical_note_result, AcceptedClinicalNoteResult):
                accepted_clinical_note_results.append(clinical_note_result)

        pipeline_batch_run_metrics = self._build_pipeline_batch_run_metrics(
            clinical_note_results=clinical_note_results
        )

        if write_artifacts:
            self._training_artifact_writer.write_accepted_note_training_rows(
                accepted_clinical_note_results=self._training_eligible_results(
                    accepted_clinical_note_results
                ),
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
        show_progress: bool = False,
    ) -> tuple[
        list[AcceptedClinicalNoteResult | RejectedClinicalNoteResult],
        PipelineBatchRunMetrics,
    ]:
        return self.run_batch_generation_pipeline(
            requested_example_count=requested_example_count,
            write_artifacts=write_artifacts,
            show_progress=show_progress,
        )

    def _run_icd_compliance_review_loop(
        self,
        *,
        generated_clinical_note: GeneratedClinicalNote,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        progress_reporter: ConsoleProgressReporter | None = None,
        trace_collector: PipelineTraceCollector | None = None,
    ) -> GeneratedClinicalNote:
        """
        Run the GPT-5 Nano ICD-10 compliance review loop before the main eval stack.

        For each loop iteration the reviewer returns note_fix_needed + fixes in one
        call. If fixes are present the generator regenerates the note using them as
        revision_targets. The method only returns notes that have already been
        reviewed; it never performs a final regeneration that goes unchecked.
        """
        if progress_reporter is not None:
            progress_reporter.stage(
                6,
                "Review ICD-10 compliance and regenerate when fixes are needed",
                model_label=(
                    f"{self._icd_compliance_reasoning_reviewer.provider_name}/"
                    f"{self._icd_compliance_reasoning_reviewer.model_name}"
                    if self._icd_compliance_reasoning_reviewer is not None
                    else None
                ),
            )
        self._trace_event(
            trace_collector,
            event_type="stage_started",
            stage_number=6,
            stage_name="Review ICD-10 compliance and regenerate when fixes are needed",
            payload={
                "reviewer_enabled": self._icd_compliance_reasoning_reviewer is not None,
                "max_regeneration_loops": self._icd_compliance_reviewer_max_regeneration_loops,
            },
        )
        if self._icd_compliance_reasoning_reviewer is None:
            if progress_reporter is not None:
                progress_reporter.detail("ICD-10 compliance review is disabled for this run.")
            self._trace_event(
                trace_collector,
                event_type="stage_passed",
                stage_number=6,
                stage_name="Review ICD-10 compliance and regenerate when fixes are needed",
                payload={"reviewer_enabled": False},
            )
            return generated_clinical_note

        seeded_bundle = bundle_semantic_constraints.seeded_bundle
        condition_icd_pairs = [
            (rc.condition_name, rc.icd_code, rc.icd_short_description)
            for rc in seeded_bundle.resolved_conditions
        ]

        current_note = generated_clinical_note
        for loop_number in range(1, self._icd_compliance_reviewer_max_regeneration_loops + 1):
            review_result = self._icd_compliance_reasoning_reviewer.review_icd_compliance(
                clinical_note_text=current_note.note_text,
                seeded_icd_codes=seeded_bundle.icd_codes,
                active_conditions=seeded_bundle.active_condition_names,
                condition_icd_pairs=condition_icd_pairs,
            )

            if not review_result.note_fix_needed:
                if progress_reporter is not None:
                    progress_reporter.detail(
                        f"Review pass {loop_number}/{self._icd_compliance_reviewer_max_regeneration_loops}: "
                        "the draft already supports the intended ICD-10 coding."
                    )
                self._trace_event(
                    trace_collector,
                    event_type="review_pass",
                    stage_number=6,
                    stage_name="Review ICD-10 compliance and regenerate when fixes are needed",
                    substage=f"review_loop_{loop_number}",
                    payload={"note_fix_needed": False, "fixes": []},
                )
                break

            if loop_number == self._icd_compliance_reviewer_max_regeneration_loops:
                if progress_reporter is not None:
                    progress_reporter.detail(
                        f"Review pass {loop_number}/{self._icd_compliance_reviewer_max_regeneration_loops}: "
                        "fixes are still needed, but the maximum review loops have been exhausted."
                    )
                logger.warning(
                    "ICD compliance reviewer exhausted %d loop(s) for template %s; "
                    "returning the last reviewed note without another regeneration.",
                    self._icd_compliance_reviewer_max_regeneration_loops,
                    seeded_bundle.template_id,
                )
                self._trace_event(
                    trace_collector,
                    event_type="review_pass",
                    stage_number=6,
                    stage_name="Review ICD-10 compliance and regenerate when fixes are needed",
                    substage=f"review_loop_{loop_number}",
                    status="warn",
                    payload={
                        "note_fix_needed": True,
                        "fixes": list(review_result.fixes),
                        "review_loops_exhausted": True,
                    },
                )
                break

            if progress_reporter is not None:
                progress_reporter.detail(
                    f"Review pass {loop_number}/{self._icd_compliance_reviewer_max_regeneration_loops}: "
                    f"{len(review_result.fixes)} targeted ICD-10 fix instructions were returned."
                )
            logger.info(
                "ICD compliance reviewer requested %d fix(es) on loop %d/%d for template %s",
                len(review_result.fixes),
                loop_number,
                self._icd_compliance_reviewer_max_regeneration_loops,
                seeded_bundle.template_id,
            )
            self._trace_event(
                trace_collector,
                event_type="review_pass",
                stage_number=6,
                stage_name="Review ICD-10 compliance and regenerate when fixes are needed",
                substage=f"review_loop_{loop_number}",
                payload={
                    "note_fix_needed": True,
                    "fix_count": len(review_result.fixes),
                    "fixes": list(review_result.fixes),
                },
            )

            current_note = self._seeded_clinical_note_generator.generate_revised_clinical_note(
                bundle_semantic_constraints=bundle_semantic_constraints,
                previous_generated_clinical_note=current_note,
                revision_targets=review_result.fixes,
                metadata_constraint_violations=[],
                generation_attempt_number=loop_number + 1,
                correlation_id=current_note.correlation_id,
            )
            if progress_reporter is not None:
                progress_reporter.detail("Regenerated the draft using the ICD-10 compliance fixes.")

        self._trace_event(
            trace_collector,
            event_type="stage_passed",
            stage_number=6,
            stage_name="Review ICD-10 compliance and regenerate when fixes are needed",
            payload={"final_note_character_count": len(current_note.note_text)},
        )
        return current_note

    def _evaluate_generated_clinical_note(
        self,
        *,
        generated_clinical_note: GeneratedClinicalNote,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        progress_reporter: ConsoleProgressReporter | None = None,
        log_stages: bool = True,
        trace_collector: PipelineTraceCollector | None = None,
    ):
        if progress_reporter is not None and log_stages:
            progress_reporter.stage(7, "Run deterministic quality checks")
        self._trace_event(
            trace_collector,
            event_type="stage_started",
            stage_number=7,
            stage_name="Run deterministic quality checks",
        )
        deterministic_precheck_outcome = self._deterministic_precheck_runner.run_prechecks(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
            recent_accepted_note_texts=self._recent_accepted_note_texts,
        )
        if progress_reporter is not None and log_stages:
            if deterministic_precheck_outcome.passed():
                progress_reporter.detail("Deterministic quality checks passed.")
            else:
                progress_reporter.detail(
                    "Deterministic quality checks failed: "
                    + "; ".join(deterministic_precheck_outcome.failure_reasons)
                )
        self._trace_event(
            trace_collector,
            event_type="stage_passed",
            stage_number=7,
            stage_name="Run deterministic quality checks",
            status="pass" if deterministic_precheck_outcome.passed() else "fail",
            payload=deterministic_precheck_outcome.model_dump(mode="json"),
        )

        if not deterministic_precheck_outcome.passed():
            if progress_reporter is not None and log_stages:
                progress_reporter.stage(8, "Verify support for each intended condition")
                progress_reporter.detail(
                    "Skipped because deterministic quality checks already failed."
                )
                progress_reporter.stage(
                    9,
                    "Score note quality and ICD alignment",
                    model_label=self._clinical_note_rubric_judge.configured_model_label,
                )
                progress_reporter.detail(
                    "Skipped because deterministic quality checks already failed."
                )
                progress_reporter.stage(
                    10,
                    "Adjudicate the final ICD-10 codes from the note text",
                    model_label=self._final_icd_code_adjudicator.configured_model_label,
                )
                progress_reporter.detail(
                    "Skipped because deterministic quality checks already failed."
                )
            self._trace_event(
                trace_collector,
                event_type="stage_skipped",
                stage_number=8,
                stage_name="Verify support for each intended condition",
                payload={"skip_reason": "deterministic_precheck_hard_fail"},
            )
            self._trace_event(
                trace_collector,
                event_type="stage_skipped",
                stage_number=9,
                stage_name="Score note quality and ICD alignment",
                payload={"skip_reason": "deterministic_precheck_hard_fail"},
            )
            self._trace_event(
                trace_collector,
                event_type="stage_skipped",
                stage_number=10,
                stage_name="Adjudicate the final ICD-10 codes from the note text",
                payload={"skip_reason": "deterministic_precheck_hard_fail"},
            )
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

        if progress_reporter is not None and log_stages:
            progress_reporter.stage(
                8,
                "Verify support for each intended condition",
                model_label=self._condition_support_verifier.configured_model_label,
            )
        self._trace_event(
            trace_collector,
            event_type="stage_started",
            stage_number=8,
            stage_name="Verify support for each intended condition",
            payload={"model_label": self._condition_support_verifier.configured_model_label},
        )
        condition_support_verification_outcome = (
            self._condition_support_verifier.verify_condition_support(
                generated_clinical_note=generated_clinical_note,
                bundle_semantic_constraints=bundle_semantic_constraints,
            )
        )
        if progress_reporter is not None and log_stages:
            progress_reporter.detail(
                "Condition support review outcome: "
                f"{condition_support_verification_outcome.outcome}."
            )
        self._trace_event(
            trace_collector,
            event_type="stage_passed",
            stage_number=8,
            stage_name="Verify support for each intended condition",
            status="pass" if condition_support_verification_outcome.passed() else "fail",
            payload=condition_support_verification_outcome.model_dump(mode="json"),
        )

        if progress_reporter is not None and log_stages:
            progress_reporter.stage(
                9,
                "Score note quality and ICD alignment",
                model_label=self._clinical_note_rubric_judge.configured_model_label,
            )
        self._trace_event(
            trace_collector,
            event_type="stage_started",
            stage_number=9,
            stage_name="Score note quality and ICD alignment",
            payload={"model_label": self._clinical_note_rubric_judge.configured_model_label},
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
        if progress_reporter is not None and log_stages:
            general_score = (
                general_quality_rubric_scores.normalized_score()
                if general_quality_rubric_scores is not None
                else None
            )
            icd_alignment_score = (
                icd_constraint_alignment_scores.normalized_score()
                if icd_constraint_alignment_scores is not None
                else None
            )
            progress_reporter.detail(
                "Quality score: "
                f"{_format_optional_score(general_score)} | "
                "ICD alignment score: "
                f"{_format_optional_score(icd_alignment_score)}."
            )
        self._trace_event(
            trace_collector,
            event_type="stage_passed",
            stage_number=9,
            stage_name="Score note quality and ICD alignment",
            payload={
                "general_quality_rubric_scores": (
                    general_quality_rubric_scores.model_dump(mode="json")
                    if general_quality_rubric_scores is not None
                    else None
                ),
                "icd_constraint_alignment_scores": (
                    icd_constraint_alignment_scores.model_dump(mode="json")
                    if icd_constraint_alignment_scores is not None
                    else None
                ),
                "icd_constraint_violations": [
                    violation.model_dump(mode="json") for violation in icd_constraint_violations
                ],
                "rubric_judge_prompt_id": rubric_judge_prompt_id,
                "rubric_judge_prompt_version": rubric_judge_prompt_version,
            },
        )

        if progress_reporter is not None and log_stages:
            progress_reporter.stage(
                10,
                "Adjudicate the final ICD-10 codes from the note text",
                model_label=self._final_icd_code_adjudicator.configured_model_label,
            )
        self._trace_event(
            trace_collector,
            event_type="stage_started",
            stage_number=10,
            stage_name="Adjudicate the final ICD-10 codes from the note text",
            payload={"model_label": self._final_icd_code_adjudicator.configured_model_label},
        )
        icd_adjudication_outcome = self._final_icd_code_adjudicator.adjudicate_generated_note(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
        )
        if progress_reporter is not None and log_stages:
            progress_reporter.detail(
                "ICD-10 adjudication outcome: "
                f"{icd_adjudication_outcome.outcome}. Final codes: "
                f"{', '.join(icd_adjudication_outcome.adjudicated_icd10_codes) or 'none'}."
            )
        self._trace_event(
            trace_collector,
            event_type="stage_passed",
            stage_number=10,
            stage_name="Adjudicate the final ICD-10 codes from the note text",
            status="pass" if icd_adjudication_outcome.passed() else "fail",
            payload=icd_adjudication_outcome.model_dump(mode="json"),
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

    def _persist_accepted_note_if_eligible(
        self,
        accepted_clinical_note_result: AcceptedClinicalNoteResult,
        progress_reporter: ConsoleProgressReporter | None = None,
        trace_collector: PipelineTraceCollector | None = None,
    ) -> None:
        if self._accepted_note_persistence is None:
            if progress_reporter is not None:
                progress_reporter.detail(
                    "PostgreSQL persistence is disabled, so no database write was attempted."
                )
            self._trace_event(
                trace_collector,
                event_type="stage_passed",
                stage_number=13,
                stage_name="Write the accepted note to PostgreSQL or explain why it is skipped",
                payload={"persistence_action": "disabled"},
            )
            return

        if not self._meets_generated_note_acceptance_bar(accepted_clinical_note_result):
            if progress_reporter is not None:
                progress_reporter.detail(
                    "Accepted by the pipeline but not eligible for PostgreSQL because the "
                    "combined score is below the persistence threshold."
                )
            self._trace_event(
                trace_collector,
                event_type="stage_passed",
                stage_number=13,
                stage_name="Write the accepted note to PostgreSQL or explain why it is skipped",
                payload={
                    "persistence_action": "skipped_below_threshold",
                    "combined_score": accepted_clinical_note_result.final_critique.combined_score,
                    "threshold": self._accept_threshold_generated_notes,
                },
            )
            return

        if progress_reporter is not None:
            progress_reporter.detail("Writing the accepted note to PostgreSQL.")
        self._trace_event(
            trace_collector,
            event_type="postgresql_write_attempt",
            stage_number=13,
            stage_name="Write the accepted note to PostgreSQL or explain why it is skipped",
            payload={
                "correlation_id": accepted_clinical_note_result.correlation_id,
                "combined_score": accepted_clinical_note_result.final_critique.combined_score,
            },
        )
        self._accepted_note_persistence.persist_accepted_note(accepted_clinical_note_result)
        if progress_reporter is not None:
            progress_reporter.detail("Accepted note stored in PostgreSQL.")
        self._trace_event(
            trace_collector,
            event_type="stage_passed",
            stage_number=13,
            stage_name="Write the accepted note to PostgreSQL or explain why it is skipped",
            payload={"persistence_action": "persisted"},
        )

    def _training_eligible_results(
        self,
        accepted_clinical_note_results: list[AcceptedClinicalNoteResult],
    ) -> list[AcceptedClinicalNoteResult]:
        return [
            result
            for result in accepted_clinical_note_results
            if self._meets_generated_note_acceptance_bar(result)
        ]

    def _meets_generated_note_acceptance_bar(
        self,
        accepted_clinical_note_result: AcceptedClinicalNoteResult,
    ) -> bool:
        combined_score = accepted_clinical_note_result.final_critique.combined_score
        return (
            combined_score is not None and combined_score >= self._accept_threshold_generated_notes
        )

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

    def _build_progress_reporter(
        self,
        *,
        note_sequence_number: int | None,
        total_requested_count: int | None,
    ) -> ConsoleProgressReporter | None:
        if note_sequence_number is None or total_requested_count is None:
            return None
        return ConsoleProgressReporter(
            note_number=note_sequence_number,
            total_count=total_requested_count,
            correlation_id=str(uuid4()),
        )

    def _attach_correlation_id_to_result(
        self,
        *,
        clinical_note_result: AcceptedClinicalNoteResult | RejectedClinicalNoteResult,
        correlation_id: str,
    ) -> AcceptedClinicalNoteResult | RejectedClinicalNoteResult:
        if isinstance(clinical_note_result, AcceptedClinicalNoteResult):
            clinical_note_result.correlation_id = correlation_id
            clinical_note_result.accepted_note.correlation_id = correlation_id
            return clinical_note_result
        clinical_note_result.correlation_id = correlation_id
        clinical_note_result.rejected_note.correlation_id = correlation_id
        return clinical_note_result

    def _attach_pipeline_trace_to_result(
        self,
        *,
        clinical_note_result: ClinicalNoteResultT,
        pipeline_trace: list[dict[str, Any]],
    ) -> ClinicalNoteResultT:
        clinical_note_result.pipeline_trace = list(pipeline_trace)
        return clinical_note_result

    def _trace_event(
        self,
        trace_collector: PipelineTraceCollector | None,
        *,
        event_type: str,
        stage_number: int | None = None,
        stage_name: str | None = None,
        substage: str | None = None,
        status: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if trace_collector is None:
            return
        trace_collector.record(
            event_type=event_type,
            stage_number=stage_number,
            stage_name=stage_name,
            substage=substage,
            status=status,
            payload=payload,
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
            normalized_score
            for result in accepted_results
            if result.final_critique.icd_constraint_alignment_scores is not None
            for normalized_score in [
                result.final_critique.icd_constraint_alignment_scores.normalized_score()
            ]
            if normalized_score is not None
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
