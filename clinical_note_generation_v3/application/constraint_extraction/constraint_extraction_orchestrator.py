"""
Constraint extraction orchestrator.

Drives the IcdCodeSemanticConstraintExtractor over every resolved condition in
a SeededClinicalBundle, assembling the results into a single
ClinicalBundleSemanticConstraints object.

This is the application-layer entry point for Phase 4 of the pipeline.  It
wires the core-layer extractor to infrastructure (LLM client) and calls it
once per resolved condition.

Public classes
--------------
  BundleConstraintExtractionOrchestrator
"""

from __future__ import annotations

import logging

from clinical_note_generation_v3.core.models.bundle import SeededClinicalBundle
from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
    IcdCodeSemanticConstraints,
)
from clinical_note_generation_v3.core.ports.llm_generation_port import JSONGenerationClient
from clinical_note_generation_v3.core.services.icd_code_semantic_constraint_extractor import (
    IcdCodeSemanticConstraintExtractor,
)

logger = logging.getLogger(__name__)


class BundleConstraintExtractionOrchestrator:
    """
    Orchestrates per-code constraint extraction for an entire clinical bundle.

    Takes a SeededClinicalBundle (all conditions already resolved to ICD
    codes) and returns a ClinicalBundleSemanticConstraints covering every
    resolved condition.

    Parameters
    ----------
    llm_escalation_client
        Optional LLM client for Tier-3 escalation in the constraint extractor.
        Pass None for fast offline or testing scenarios — only tiers 1+2 will
        run and LLM escalation will be silently skipped for any code that has
        no deterministic constraints.
    """

    def __init__(
        self,
        llm_escalation_client: JSONGenerationClient | None = None,
    ) -> None:
        self._constraint_extractor = IcdCodeSemanticConstraintExtractor(
            llm_escalation_client=llm_escalation_client,
        )

    def extract_bundle_semantic_constraints(
        self,
        seeded_bundle: SeededClinicalBundle,
    ) -> ClinicalBundleSemanticConstraints:
        """
        Extract note-writing constraints for every resolved condition in the bundle.

        Each resolved condition is processed in order; the resulting
        IcdCodeSemanticConstraints list mirrors the order of
        seeded_bundle.resolved_conditions.

        Parameters
        ----------
        seeded_bundle
            The SeededClinicalBundle produced by the ICD resolution layer.
            Must have at least one entry in resolved_conditions.

        Returns
        -------
        ClinicalBundleSemanticConstraints
            Ready to be passed to the note generator and then to the evaluator.
        """
        resolved_condition_constraints: list[IcdCodeSemanticConstraints] = []

        for resolved_condition in seeded_bundle.resolved_conditions:
            logger.debug(
                "Extracting constraints for code %s (%s)",
                resolved_condition.icd_code,
                resolved_condition.icd_short_description[:50],
            )
            resolved_condition_constraint = (
                self._constraint_extractor.extract_semantic_constraints_for_resolved_condition(
                    resolved_condition
                )
            )
            resolved_condition_constraints.append(resolved_condition_constraint)
            logger.debug(
                "  -> chapter_family=%s, laterality=%s, encounter_type=%s, "
                "must_include_count=%d, must_not_imply_count=%d",
                resolved_condition_constraint.chapter_family,
                resolved_condition_constraint.laterality,
                resolved_condition_constraint.encounter_type,
                len(resolved_condition_constraint.must_include_in_note),
                len(resolved_condition_constraint.must_not_imply_in_note),
            )

        return ClinicalBundleSemanticConstraints(
            seeded_bundle=seeded_bundle,
            per_code_note_writing_constraints=resolved_condition_constraints,
        )

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_default_settings(cls) -> "BundleConstraintExtractionOrchestrator":
        """
        Creates the orchestrator wired to the default V3 pipeline LLM client.

        Uses DeepSeek as primary and OpenAI as fallback for LLM escalation.
        """
        from clinical_note_generation_v3.infrastructure.llm_provider.llm_client_factory import (
            create_default_json_generation_client,
        )

        return cls(llm_escalation_client=create_default_json_generation_client())

    # ------------------------------------------------------------------
    # Backward-compatible wrapper during naming transition
    # ------------------------------------------------------------------

    def extract_constraints_for_seeded_bundle(
        self,
        seeded_bundle: SeededClinicalBundle,
    ) -> ClinicalBundleSemanticConstraints:
        return self.extract_bundle_semantic_constraints(seeded_bundle)
