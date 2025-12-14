"""
Generation Layer - Clinical Note Generation

This layer handles the core task of generating clinical note text
using LLM APIs with ICD-10 code guidance.

Submodules:
    note_generator.py → Main generator class
    prompt_builder.py → Prompt construction and templates

Dependency Rule:
    This layer depends on: core, clients (LLM)
    This layer is used by: pipeline (orchestrator)

Author: Shubham Singh
Date: December 2025
"""

from clinical_note_generation.generation.note_generator import NoteGenerator
from clinical_note_generation.generation.prompt_builder import PromptBuilder

__all__ = [
    "NoteGenerator",
    "PromptBuilder",
]
