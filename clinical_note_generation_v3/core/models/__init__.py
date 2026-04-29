"""
Public re-exports for core/models.

Import everything you need from here rather than from the individual files,
so internal file reorganisation does not break import sites.

  from clinical_note_generation_v3.core.models import (
      ClinicalBundleTemplate,
      IcdSeededClinicalBundle,
      ICDCodeRecord,
      CandidateCode,
      ...
  )
"""

# --- Official ICD code record types (preserved from icd_codes.py) ---
from .icd_codes import (
    ICDCodeRecord,
    CandidateCode,
    SelectedCode,
    RejectedCandidate,
    MissingCandidateCode,
)

from .icd_adjudication import (
    ClinicalDiagnosisMention,
    FinalIcdCodeAdjudicationOutcome,
    IcdCodeSetValidationIssue,
    IcdCodeSetValidationOutcome,
)
from .icd_rules import IcdTabularRuleNode

# --- Bundle types ---
from .bundle import (
    ClinicalBundleTemplate,
    ResolvedConditionCode,
    SeededClinicalBundle,
    ResolvedConditionWithIcdCode,
    IcdSeededClinicalBundle,
)

# --- Constraint types ---
from .constraints import (
    ConstraintViolationSeverity,
    ConstraintExtractionSource,
    ExtractedSemanticSignal,
    IcdCodeSemanticConstraints,
    ClinicalBundleSemanticConstraints,
    IcdCodeNoteWritingConstraints,
    ClinicalBundleNoteWritingConstraints,
)

# --- Note types ---
from .note import (
    GenerationModelInfo,
    GeneratedClinicalNote,
    ClinicalNoteGenerationAttempt,
)

# --- Evaluation types ---
from .evaluation import (
    # Layer 1
    DeterministicPreCheckOutcome,
    # Layer 2
    ConditionSupportVerificationOutcome,
    # Layer 3
    RubricCriterionEvaluation,
    NoteGeneralQualityRubricScores,
    IcdConstraintAlignmentRubricScores,
    IcdConstraintViolationDetail,
    # Combined
    NoteEvaluationCritiqueResult,
    # Revision and final results
    RevisionAttemptRecord,
    AcceptedClinicalNoteResult,
    RejectedClinicalNoteResult,
    # Batch metrics
    PipelineBatchRunMetrics,
)

__all__ = [
    # ICD code record types
    "ICDCodeRecord",
    "CandidateCode",
    "SelectedCode",
    "RejectedCandidate",
    "MissingCandidateCode",
    "ClinicalDiagnosisMention",
    "FinalIcdCodeAdjudicationOutcome",
    "IcdCodeSetValidationIssue",
    "IcdCodeSetValidationOutcome",
    "IcdTabularRuleNode",
    # Bundle
    "ClinicalBundleTemplate",
    "ResolvedConditionCode",
    "SeededClinicalBundle",
    "ResolvedConditionWithIcdCode",
    "IcdSeededClinicalBundle",
    # Constraints
    "ConstraintViolationSeverity",
    "ConstraintExtractionSource",
    "ExtractedSemanticSignal",
    "IcdCodeSemanticConstraints",
    "ClinicalBundleSemanticConstraints",
    "IcdCodeNoteWritingConstraints",
    "ClinicalBundleNoteWritingConstraints",
    # Note
    "GenerationModelInfo",
    "GeneratedClinicalNote",
    "ClinicalNoteGenerationAttempt",
    # Evaluation — pre-checks
    "DeterministicPreCheckOutcome",
    # Evaluation — support verification
    "ConditionSupportVerificationOutcome",
    # Evaluation — rubric
    "RubricCriterionEvaluation",
    "NoteGeneralQualityRubricScores",
    "IcdConstraintAlignmentRubricScores",
    "IcdConstraintViolationDetail",
    # Evaluation — critique result
    "NoteEvaluationCritiqueResult",
    # Revision and final results
    "RevisionAttemptRecord",
    "AcceptedClinicalNoteResult",
    "RejectedClinicalNoteResult",
    # Batch metrics
    "PipelineBatchRunMetrics",
]
