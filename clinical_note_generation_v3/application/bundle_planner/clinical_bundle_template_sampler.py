"""
Clinical bundle template sampler.

Selects ClinicalBundleTemplate objects from a ClinicalBundleTemplateRegistry
with built-in archetype rotation so that a batch of generated notes covers a
diverse set of encounter types rather than clustering on one archetype.

Design responsibilities
-----------------------
  - This class maintains the ARCHETYPE rotation state between calls.
  - The CALLER (pipeline orchestrator) maintains the list of recently used
    template IDs and passes it in on each call.  This keeps the sampler
    stateless with respect to what has already been generated.

Archetype rotation strategy
---------------------------
  On initialisation, the sampler shuffles all known archetypes into a rotation
  order.  Each call to sample_next_template() advances one step through this
  rotation, favouring the current archetype while avoiding recently used IDs.
  When all templates in the current archetype have been recently used, it falls
  back to any available template and still advances the rotation pointer so the
  next call tries a fresh archetype.

Typical usage
-------------
  registry = ClinicalBundleTemplateRegistry.from_default_bundle_data_directory()
  sampler  = ClinicalBundleTemplateSampler(registry, random_seed=42)

  # Inside a pipeline batch loop:
  recently_used_ids: list[str] = []
  for _ in range(50):
      template = sampler.sample_next_template(recently_used_template_ids=recently_used_ids)
      recently_used_ids.append(template.template_id)
      ...

  # Or sample a whole batch at once:
  diverse_batch = sampler.sample_diverse_batch_of_templates(requested_count=20)
"""

from __future__ import annotations

import random

from clinical_note_generation_v3.application.bundle_planner.clinical_bundle_template_registry import (
    ClinicalBundleTemplateRegistry,
)
from clinical_note_generation_v3.core.models.bundle import ClinicalBundleTemplate


class ClinicalBundleTemplateSampler:
    """
    Selects templates from a ClinicalBundleTemplateRegistry with archetype
    rotation to ensure diverse encounter-type coverage across a batch run.

    Parameters
    ----------
    registry
        A populated ClinicalBundleTemplateRegistry.  Must contain at least
        one template; raises ValueError if empty.
    random_seed
        Optional integer seed for reproducible sampling.  Useful for testing
        and for generating the same dataset across runs.
    """

    def __init__(
        self,
        registry: ClinicalBundleTemplateRegistry,
        random_seed: int | None = None,
    ) -> None:
        if registry.is_empty():
            raise ValueError(
                "ClinicalBundleTemplateSampler requires a non-empty registry. "
                "Load templates before constructing the sampler."
            )
        self._registry = registry
        self._random_number_generator = random.Random(random_seed)
        self._archetype_rotation_sequence = self._build_shuffled_archetype_rotation_sequence()
        self._current_position_in_archetype_rotation = 0

    # ------------------------------------------------------------------
    # Primary sampling interface
    # ------------------------------------------------------------------

    def select_next_template(
        self,
        recently_used_template_ids: list[str] | None = None,
    ) -> ClinicalBundleTemplate:
        """
        Returns the next template to generate a note for.

        Selects from the archetype currently due in the rotation sequence,
        preferring templates whose IDs are not in recently_used_template_ids.
        Always advances the rotation pointer so the next call will favour a
        different archetype.

        If every template in the current archetype was recently used, the
        sampler tries the next archetype in the rotation before falling back
        to a globally unconstrained random pick.

        Parameters
        ----------
        recently_used_template_ids
            Template IDs to avoid if possible.  The caller is responsible for
            tracking this list across calls.  Pass None or [] to disable
            recency filtering.
        """
        recently_used_id_set = set(recently_used_template_ids or [])

        selected_template = self._try_picking_from_archetype_rotation_with_recency_filter(
            recently_used_id_set
        )

        if selected_template is None:
            selected_template = self._pick_any_template_ignoring_recency()

        return selected_template

    def select_diverse_batch_of_templates(
        self,
        requested_count: int,
    ) -> list[ClinicalBundleTemplate]:
        """
        Returns a list of templates with guaranteed archetype rotation diversity.

        No template_id appears twice in the batch unless the registry has
        fewer unique templates than requested_count, in which case templates
        are allowed to repeat after exhaustion.

        Parameters
        ----------
        requested_count
            How many templates to include in the batch.
        """
        if requested_count <= 0:
            raise ValueError(f"requested_count must be a positive integer, got {requested_count}.")

        accumulated_template_ids_used_in_this_batch: list[str] = []
        selected_templates: list[ClinicalBundleTemplate] = []

        for _ in range(requested_count):
            next_template = self.select_next_template(
                recently_used_template_ids=accumulated_template_ids_used_in_this_batch,
            )
            selected_templates.append(next_template)
            if next_template.template_id not in accumulated_template_ids_used_in_this_batch:
                accumulated_template_ids_used_in_this_batch.append(next_template.template_id)

        return selected_templates

    # ------------------------------------------------------------------
    # Rotation state inspection (useful for diagnostics and testing)
    # ------------------------------------------------------------------

    def current_rotation_archetype(self) -> str:
        """Returns the archetype that the next call to sample_next_template will favour."""
        position = self._current_position_in_archetype_rotation % len(
            self._archetype_rotation_sequence
        )
        return self._archetype_rotation_sequence[position]

    def list_rotation_sequence(self) -> list[str]:
        """Returns a copy of the shuffled archetype rotation sequence."""
        return list(self._archetype_rotation_sequence)

    def count_completed_rotation_steps(self) -> int:
        """Returns how many archetype steps have been taken since initialisation."""
        return self._current_position_in_archetype_rotation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_shuffled_archetype_rotation_sequence(self) -> list[str]:
        """
        Creates the initial shuffled archetype list that the sampler cycles through.
        Shuffling ensures there is no systematic bias toward one archetype at the
        start of every batch.
        """
        archetypes = self._registry.list_archetypes()
        self._random_number_generator.shuffle(archetypes)
        return archetypes

    def _try_picking_from_archetype_rotation_with_recency_filter(
        self,
        recently_used_id_set: set[str],
    ) -> ClinicalBundleTemplate | None:
        """
        Attempts to find a non-recently-used template starting from the current
        archetype in the rotation and walking forward through the sequence until
        one is found or all archetypes are exhausted.

        Returns None only when every template in every archetype appears in
        recently_used_id_set.
        """
        total_archetypes = len(self._archetype_rotation_sequence)

        for steps_tried in range(total_archetypes):
            rotation_index = (
                self._current_position_in_archetype_rotation + steps_tried
            ) % total_archetypes

            archetype_at_this_position = self._archetype_rotation_sequence[rotation_index]
            candidate_templates = self._registry.find_templates_by_archetype(
                archetype_at_this_position
            )
            available_templates = [
                t for t in candidate_templates if t.template_id not in recently_used_id_set
            ]

            if available_templates:
                selected = self._random_number_generator.choice(available_templates)
                self._advance_to_next_archetype_in_rotation()
                return selected

        self._advance_to_next_archetype_in_rotation()
        return None

    def _pick_any_template_ignoring_recency(self) -> ClinicalBundleTemplate:
        """
        Last-resort fallback used when every template has been recently used.
        Picks uniformly at random from the full template list.
        """
        all_templates = self._registry.list_templates()
        return self._random_number_generator.choice(all_templates)

    def _advance_to_next_archetype_in_rotation(self) -> None:
        """Moves the rotation pointer forward by one position."""
        self._current_position_in_archetype_rotation += 1

    # ------------------------------------------------------------------
    # Backward-compatible wrappers during naming transition
    # ------------------------------------------------------------------

    def sample_next_template(
        self,
        recently_used_template_ids: list[str] | None = None,
    ) -> ClinicalBundleTemplate:
        return self.select_next_template(recently_used_template_ids=recently_used_template_ids)

    def sample_diverse_batch_of_templates(
        self,
        requested_count: int,
    ) -> list[ClinicalBundleTemplate]:
        return self.select_diverse_batch_of_templates(requested_count=requested_count)

    def current_archetype_in_rotation(self) -> str:
        return self.current_rotation_archetype()

    def rotation_sequence(self) -> list[str]:
        return self.list_rotation_sequence()

    def rotation_steps_completed(self) -> int:
        return self.count_completed_rotation_steps()
