"""
Factory for the end-to-end clinical note quality pipeline.
"""

from __future__ import annotations

from pathlib import Path

from clinical_note_generation_v3.application.bundle_planner.clinical_bundle_template_registry import (
    ClinicalBundleTemplateRegistry,
)
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
from clinical_note_generation_v3.application.icd_resolution.icd_condition_to_code_resolver import (
    IcdConditionToCodeResolver,
)
from clinical_note_generation_v3.application.note_generation.seeded_clinical_note_generator import (
    SeededClinicalNoteGenerator,
)
from clinical_note_generation_v3.application.pipeline.clinical_note_quality_pipeline import (
    ClinicalNoteQualityPipeline,
)
from clinical_note_generation_v3.artifacts.audit_artifact_writer import AuditArtifactWriter
from clinical_note_generation_v3.artifacts.batch_metrics_writer import BatchMetricsWriter
from clinical_note_generation_v3.artifacts.training_artifact_writer import TrainingArtifactWriter
from clinical_note_generation_v3.config.settings import V3PipelineSettings
from clinical_note_generation_v3.core.services.deterministic_precheck_runner import (
    DeterministicPreCheckRunner,
)


def create_default_clinical_note_quality_pipeline(
    *,
    bundle_template_directory: Path | None = None,
    training_artifact_output_path: Path | None = None,
    audit_artifact_output_path: Path | None = None,
    batch_metrics_output_path: Path | None = None,
) -> ClinicalNoteQualityPipeline:
    """
    Create a fully wired end-to-end pipeline using default settings.
    """
    settings = V3PipelineSettings()

    clinical_bundle_template_registry = ClinicalBundleTemplateRegistry()
    clinical_bundle_template_registry.load_templates_from_directory(
        bundle_template_directory or settings.bundle_template_directory
    )
    clinical_bundle_template_sampler = ClinicalBundleTemplateSampler(
        clinical_bundle_template_registry,
        random_seed=settings.random_seed,
    )

    icd_condition_to_code_resolver = IcdConditionToCodeResolver.from_default_settings(
        candidate_count_per_condition=settings.candidate_count
    )
    bundle_constraint_extraction_orchestrator = (
        BundleConstraintExtractionOrchestrator.from_default_settings()
    )
    seeded_clinical_note_generator = SeededClinicalNoteGenerator.from_default_settings(
        random_seed=settings.random_seed
    )
    deterministic_precheck_runner = DeterministicPreCheckRunner(
        minimum_note_character_count=settings.minimum_note_character_count,
        near_duplicate_similarity_threshold=settings.near_duplicate_similarity_threshold,
    )
    condition_support_verifier = ConditionSupportVerifier.from_default_settings()
    clinical_note_rubric_judge = ClinicalNoteRubricJudge.from_default_settings()
    note_quality_decision_combiner = NoteQualityDecisionCombiner(
        accept_threshold=settings.accept_score_threshold,
        revise_threshold=settings.revise_score_threshold,
        reject_below_threshold=settings.reject_below_score_threshold,
    )
    clinical_note_revision_loop = ClinicalNoteRevisionLoop(
        seeded_clinical_note_generator=seeded_clinical_note_generator,
        max_revision_attempts=settings.max_revision_attempts,
    )

    return ClinicalNoteQualityPipeline(
        clinical_bundle_template_sampler=clinical_bundle_template_sampler,
        icd_condition_to_code_resolver=icd_condition_to_code_resolver,
        bundle_constraint_extraction_orchestrator=bundle_constraint_extraction_orchestrator,
        seeded_clinical_note_generator=seeded_clinical_note_generator,
        deterministic_precheck_runner=deterministic_precheck_runner,
        condition_support_verifier=condition_support_verifier,
        clinical_note_rubric_judge=clinical_note_rubric_judge,
        note_quality_decision_combiner=note_quality_decision_combiner,
        clinical_note_revision_loop=clinical_note_revision_loop,
        training_artifact_writer=TrainingArtifactWriter(),
        audit_artifact_writer=AuditArtifactWriter(),
        batch_metrics_writer=BatchMetricsWriter(),
        training_artifact_output_path=(
            training_artifact_output_path or settings.training_artifact_output_path
        ),
        audit_artifact_output_path=(
            audit_artifact_output_path or settings.audit_artifact_output_path
        ),
        batch_metrics_output_path=(batch_metrics_output_path or settings.batch_metrics_output_path),
        recent_accepted_note_window_size=settings.recent_accepted_note_window_size,
    )


def create_clinical_note_quality_pipeline() -> ClinicalNoteQualityPipeline:
    return create_default_clinical_note_quality_pipeline()
