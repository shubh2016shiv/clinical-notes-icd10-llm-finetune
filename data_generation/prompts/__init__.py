"""
Prompts module for clinical notes generation.

Contains centralized prompt templates for AI model interactions.
"""

from .generate_clinical_note_prompt import get_clinical_note_generation_prompt
from .validation_prompt import get_icd10_validation_prompt

__all__ = ["get_clinical_note_generation_prompt", "get_icd10_validation_prompt"]

