"""
bundle_planner — loads and samples clinical bundle templates.

Public API for this sub-package:

  from clinical_note_generation_v3.application.bundle_planner import (
      ClinicalBundleTemplateRegistry,
      ClinicalBundleTemplateSampler,
  )

The registry reads JSONL files from data/bundle_templates/.
The sampler selects templates with archetype rotation for diverse batches.
"""

from .clinical_bundle_template_registry import ClinicalBundleTemplateRegistry
from .clinical_bundle_template_sampler import ClinicalBundleTemplateSampler

__all__ = [
    "ClinicalBundleTemplateRegistry",
    "ClinicalBundleTemplateSampler",
]
