"""
Selection Layer - ICD-10 Code Selection Strategies

This layer provides intelligent selection of ICD-10 codes for clinical note
generation. Key capability: selecting medically coherent code combinations.

Submodules:
    code_selector.py → Main unified selector (facade)
    strategies.py    → Individual selection strategy implementations

Dependency Rule:
    This layer depends on: core, repository
    This layer is used by: generation

Author: Shubham Singh
Date: December 2025
"""

from clinical_note_generation.selection.code_selector import CodeSelector
from clinical_note_generation.selection.strategies import (
    SelectionStrategyBase,
    RandomSelectionStrategy,
    ProfileBasedSelectionStrategy,
)

__all__ = [
    "CodeSelector",
    "SelectionStrategyBase",
    "RandomSelectionStrategy",
    "ProfileBasedSelectionStrategy",
]
