"""
Core Layer - Domain Models, Enums, and Configuration

This layer contains PURE, side-effect-free components that form the foundation
of the clinical note generation system. No external dependencies beyond
standard library and dataclasses.

Submodules:
    models.py   → Data structures (ICD10Code, ClinicalNote, ValidationResult)
    enums.py    → Enumerations (ClinicalNoteType, ValidationStatus)
    config.py   → Configuration dataclass
    exceptions.py → Domain-specific exceptions

Dependency Rule:
    This layer depends on NOTHING else in the package.
    All other layers may depend on this layer.

Author: Shubham Singh
Date: December 2025
"""

from clinical_note_generation.core.models import (
    ICD10Code,
    ClinicalNote,
    ValidationResult,
)
from clinical_note_generation.core.enums import (
    ClinicalNoteType,
    ValidationStatus,
    SelectionStrategy,
)
from clinical_note_generation.core.config import PipelineConfiguration
from clinical_note_generation.core.exceptions import (
    ClinicalNoteGenerationError,
    ICD10CodeError,
    ValidationError,
    ConfigurationError,
)

__all__ = [
    # Models
    "ICD10Code",
    "ClinicalNote",
    "ValidationResult",
    # Enums
    "ClinicalNoteType",
    "ValidationStatus",
    "SelectionStrategy",
    # Configuration
    "PipelineConfiguration",
    # Exceptions
    "ClinicalNoteGenerationError",
    "ICD10CodeError",
    "ValidationError",
    "ConfigurationError",
]
