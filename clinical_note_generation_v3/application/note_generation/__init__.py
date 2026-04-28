"""
Seeded clinical note generation layer.

This package contains:
  - prompt building for bundle-seeded clinical note generation
  - the generator that calls the configured LLM and returns a
    GeneratedClinicalNote with model provenance and fake patient context
"""

from .seeded_clinical_note_prompt_builder import (
    build_seeded_clinical_note_generation_prompt,
    build_seeded_clinical_note_generation_response_schema,
    build_seeded_clinical_note_revision_prompt,
)
from .seeded_clinical_note_generator import (
    FakePatientIdentity,
    SeededClinicalNoteGenerator,
)

__all__ = [
    "build_seeded_clinical_note_generation_prompt",
    "build_seeded_clinical_note_generation_response_schema",
    "build_seeded_clinical_note_revision_prompt",
    "FakePatientIdentity",
    "SeededClinicalNoteGenerator",
]
