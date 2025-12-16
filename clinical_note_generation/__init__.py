"""
Clinical Note Generation Module

A production-grade system for generating realistic, de-identified clinical notes
with validated ICD-10 code assignments for model fine-tuning.

Architecture Overview:
    clinical_note_generation/
    ├── core/           → Domain models, enums, configuration (Layer 0 - Pure)
    ├── repository/     → ICD-10 data access (Layer 1 - Infrastructure)
    ├── selection/      → Code selection strategies (Layer 2 - Business Logic)
    ├── generation/     → Note generation (Layer 3 - Business Logic)
    ├── validation/     → Note validation (Layer 4 - Business Logic)
    ├── clients/        → LLM client abstractions (Layer 5 - Infrastructure)
    └── pipeline.py     → Main orchestrator (Layer 6 - Public API)

Quick Start:
    from clinical_note_generation import ClinicalNotePipeline

    pipeline = ClinicalNotePipeline.from_environment()
    notes = pipeline.generate_notes(count=10, profile="chronic_care")

Author: Shubham Singh
Date: December 2025
"""

__version__ = "1.0.0"
__author__ = "Shubham Singh"

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================
# These are the only classes/functions that external code should import.
# All internal implementation details are hidden behind this facade.

# Main Entry Point
from clinical_note_generation.pipeline import ClinicalNotePipeline

# Core Models
from clinical_note_generation.core.models import (
    ICD10Code,
    ClinicalNote,
    ValidationResult,
)

# Enums
from clinical_note_generation.core.enums import (
    ClinicalNoteType,
    ValidationStatus,
    SelectionStrategy,
    GenerationProfile,
)

# Configuration
from clinical_note_generation.core.config import PipelineConfiguration

__all__ = [
    # Main Entry Point (use this!)
    "ClinicalNotePipeline",
    # Core Models
    "ICD10Code",
    "ClinicalNote",
    "ValidationResult",
    # Enums
    "ClinicalNoteType",
    "ValidationStatus",
    "SelectionStrategy",
    "GenerationProfile",
    # Configuration
    "PipelineConfiguration",
]
