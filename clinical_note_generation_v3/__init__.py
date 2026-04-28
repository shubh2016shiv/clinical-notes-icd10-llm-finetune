"""
Clinical Note Generation V3 — bundle-seeded clinical note quality pipeline.

ARCHITECTURE:
┌──────────────────────────────────────────────────────────────────────────┐
│  scripts/  │  diagnostics/  │  artifacts/                                │ ← Entrypoints / Output
├──────────────────────────────────────────────────────────────────────────┤
│                          application/                                    │ ← Orchestration
│  bundle_planner/  │  icd_resolution/  │  note_generator/                 │
│  evaluation/      │  pipeline/                                           │
├──────────────────────────────────────────────────────────────────────────┤
│                         infrastructure/                                  │ ← External Adapters
│  data_preprocessing/ │ llm_provider/ │ embedding_provider/ │ vector_store/│
├──────────────────────────────────────────────────────────────────────────┤
│                            core/                                         │ ← Business Logic
│  models/  │  ports/  │  services/  │  retrieval/                        │
└──────────────────────────────────────────────────────────────────────────┘
                       config/  (inward-only reads)

DATA FLOW:
  ClinicalBundleTemplate  (from bundle_planner registry)
      -> IcdSeededClinicalBundle  (via icd_resolution layer)
      -> ClinicalBundleNoteWritingConstraints  (via constraint extractor)
      -> GeneratedClinicalNote  (via seeded note generator)
      -> NoteEvaluationCritiqueResult  (via evaluation stack)
      -> AcceptedClinicalNoteResult  (via revision loop)
      -> training artifact + audit artifact  (via artifact writers)

KEY DESIGN:
  - Bundle-first: ICD codes are resolved before any note is written
  - Generator renders, not decides: the bundle defines the case; the generator
    only expresses it in realistic clinical prose
  - Tiered evaluation: deterministic gates -> support verifier -> rubric judge
  - Constraint-anchored: the same IcdCodeNoteWritingConstraints object drives
    generation, verification, and rubric scoring

BUILD STATUS (per phase):
  Phase 1 — core/models/          DONE
  Phase 2 — bundle registry data  pending
  Phase 3 — ICD resolution layer  pending
  Phase 4 — constraint extractor  pending
  Phase 5 — note generator        pending
  Phase 6 — evaluation stack      pending
  Phase 7 — revision loop         pending
  Phase 8 — pipeline              pending
  Phase 9 — artifact writers      pending
"""

# Phase 1 models are the only complete layer right now.
# Remaining layers will be re-exported here as they are built.
from clinical_note_generation_v3.core.models import (
    ClinicalBundleTemplate,
    IcdSeededClinicalBundle,
    ClinicalBundleNoteWritingConstraints,
    GeneratedClinicalNote,
    AcceptedClinicalNoteResult,
    RejectedClinicalNoteResult,
    PipelineBatchRunMetrics,
)

__all__ = [
    "ClinicalBundleTemplate",
    "IcdSeededClinicalBundle",
    "ClinicalBundleNoteWritingConstraints",
    "GeneratedClinicalNote",
    "AcceptedClinicalNoteResult",
    "RejectedClinicalNoteResult",
    "PipelineBatchRunMetrics",
]
