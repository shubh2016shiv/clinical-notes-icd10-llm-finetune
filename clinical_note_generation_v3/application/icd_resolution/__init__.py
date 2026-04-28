"""
application/icd_resolution — resolves human-readable condition names to official
ICD-10-CM codes before any note is written.

Public API
----------
  IcdResolutionFailedForConditionError
      Raised when a condition cannot be resolved to a valid billable code.
      Catch this in the pipeline orchestrator to skip or retry the bundle.

  IcdConditionToCodeResolver
      The main resolver.  Drives BM25 retrieval → LLM selection → validation.
      Construct via IcdConditionToCodeResolver.from_default_settings() for
      standard usage, or wire dependencies manually for testing.
"""

from .icd_condition_to_code_resolver import (
    IcdResolutionFailedForConditionError,
    IcdConditionToCodeResolver,
)
from .icd_single_condition_resolution_prompt_builder import (
    build_icd_code_selection_prompt_for_single_condition,
)

__all__ = [
    "IcdResolutionFailedForConditionError",
    "IcdConditionToCodeResolver",
    "build_icd_code_selection_prompt_for_single_condition",
]
