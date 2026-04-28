"""
Clinical bundle template registry.

Loads ClinicalBundleTemplate objects from JSONL data files and exposes
query methods so the rest of the pipeline can find templates by archetype,
complexity tier, or ID without re-parsing files on every access.

Typical usage
-------------
  registry = ClinicalBundleTemplateRegistry.from_default_bundle_data_directory()
  templates = registry.get_templates_by_archetype("acute_injury_initial_encounter")
  template   = registry.find_template_by_id("acute_injury_01")

JSONL file format
-----------------
  One JSON object per line, each matching the ClinicalBundleTemplate schema.
  Blank lines and lines starting with '#' are silently skipped.
  Duplicate template_id values across any combination of files raise ValueError.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from clinical_note_generation_v3.core.models.bundle import ClinicalBundleTemplate


class ClinicalBundleTemplateRegistry:
    """
    In-memory registry of clinical bundle templates loaded from JSONL files.

    All templates are indexed by template_id (for O(1) lookup) and by
    archetype (for fast filtering without scanning the full list).
    Duplicate template_id values are rejected at load time.
    """

    def __init__(self) -> None:
        self._templates_indexed_by_id: dict[str, ClinicalBundleTemplate] = {}
        self._template_ids_grouped_by_archetype: dict[str, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_templates_from_jsonl_file(self, jsonl_file_path: Path) -> int:
        """
        Parses every non-blank line in a JSONL file as a ClinicalBundleTemplate
        and registers it.

        Returns the number of templates successfully loaded from this file.

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        ValueError
            When a line cannot be parsed as a valid ClinicalBundleTemplate, or
            when a template_id duplicates one already in the registry.
        """
        if not jsonl_file_path.exists():
            raise FileNotFoundError(f"Bundle template JSONL file not found: {jsonl_file_path}")

        templates_loaded_from_this_file = 0

        with open(jsonl_file_path, encoding="utf-8") as jsonl_file:
            for line_number, raw_line in enumerate(jsonl_file, start=1):
                stripped_line = raw_line.strip()

                if not stripped_line or stripped_line.startswith("#"):
                    continue

                try:
                    parsed_template = ClinicalBundleTemplate.model_validate_json(stripped_line)
                except Exception as parse_error:
                    raise ValueError(
                        f"Failed to parse bundle template at "
                        f"{jsonl_file_path}:{line_number} — {parse_error}"
                    ) from parse_error

                self._register_template(parsed_template)
                templates_loaded_from_this_file += 1

        return templates_loaded_from_this_file

    def load_templates_from_directory(self, directory_path: Path) -> int:
        """
        Loads every *.jsonl file found in directory_path (non-recursive, sorted).

        Returns the total number of templates loaded across all files.

        Raises
        ------
        FileNotFoundError
            When directory_path does not exist or is not a directory.
        ValueError
            When any line fails to parse, or a duplicate template_id is found.
        """
        if not directory_path.is_dir():
            raise FileNotFoundError(f"Bundle templates directory not found: {directory_path}")

        jsonl_files = sorted(directory_path.glob("*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(
                f"No *.jsonl files found in bundle templates directory: {directory_path}"
            )

        total_templates_loaded = 0
        for jsonl_file in jsonl_files:
            total_templates_loaded += self.load_templates_from_jsonl_file(jsonl_file)

        return total_templates_loaded

    def _register_template(self, template: ClinicalBundleTemplate) -> None:
        """Adds one template to both internal indexes. Raises if template_id is duplicate."""
        if template.template_id in self._templates_indexed_by_id:
            raise ValueError(
                f"Duplicate template_id detected: '{template.template_id}'. "
                f"Every template must have a unique ID across all JSONL files."
            )
        self._templates_indexed_by_id[template.template_id] = template
        self._template_ids_grouped_by_archetype[template.archetype].append(template.template_id)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def list_templates(self) -> list[ClinicalBundleTemplate]:
        """Returns every registered template, in insertion order."""
        return list(self._templates_indexed_by_id.values())

    def find_templates_by_archetype(self, archetype: str) -> list[ClinicalBundleTemplate]:
        """
        Returns all templates whose archetype field matches the given string.
        Returns an empty list when the archetype is not found — does not raise.
        """
        matching_ids = self._template_ids_grouped_by_archetype.get(archetype, [])
        return [self._templates_indexed_by_id[tid] for tid in matching_ids]

    def find_templates_by_complexity_tier(
        self, complexity_tier: int
    ) -> list[ClinicalBundleTemplate]:
        """Returns all templates at the specified complexity tier (1, 2, or 3)."""
        return [
            template
            for template in self._templates_indexed_by_id.values()
            if template.complexity_tier == complexity_tier
        ]

    def get_template_by_id(self, template_id: str) -> ClinicalBundleTemplate | None:
        """
        Returns the template with this ID, or None if no template has this ID.
        O(1) lookup.
        """
        return self._templates_indexed_by_id.get(template_id)

    def get_template_by_id_or_raise(self, template_id: str) -> ClinicalBundleTemplate:
        """
        Returns the template with this ID.

        Raises
        ------
        KeyError
            When no template with this ID exists in the registry.
        """
        template = self.get_template_by_id(template_id)
        if template is None:
            raise KeyError(
                f"No bundle template with ID '{template_id}' found in registry. "
                f"Known IDs: {sorted(self._templates_indexed_by_id.keys())}"
            )
        return template

    def list_archetypes(self) -> list[str]:
        """Returns a sorted list of every archetype string present in the registry."""
        return sorted(self._template_ids_grouped_by_archetype.keys())

    def total_template_count(self) -> int:
        """Returns the total number of templates currently in the registry."""
        return len(self._templates_indexed_by_id)

    def template_count_by_archetype(self) -> dict[str, int]:
        """Returns a dict mapping each archetype to its template count."""
        return {
            archetype: len(ids)
            for archetype, ids in self._template_ids_grouped_by_archetype.items()
        }

    def is_empty(self) -> bool:
        """True when no templates have been loaded yet."""
        return len(self._templates_indexed_by_id) == 0

    # ------------------------------------------------------------------
    # Convenience factory — builds from the package's own data directory
    # ------------------------------------------------------------------

    @classmethod
    def from_default_bundle_data_directory(cls) -> "ClinicalBundleTemplateRegistry":
        """
        Creates a registry pre-loaded from the package's default
        data/bundle_templates/ directory, located relative to this file:

          clinical_note_generation_v3/data/bundle_templates/

        This is the standard way to get a fully populated registry without
        having to know where the data files live.
        """
        default_data_directory = (
            Path(
                __file__
            ).parent.parent.parent  # application/bundle_planner/  # application/  # clinical_note_generation_v3/
            / "data"
            / "bundle_templates"
        )
        registry = cls()
        registry.load_templates_from_directory(default_data_directory)
        return registry

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def build_summary_report(self) -> str:
        """Returns a human-readable summary of what is loaded in the registry."""
        lines = [
            f"ClinicalBundleTemplateRegistry — {self.total_template_count()} templates total",
            "",
            "Templates per archetype:",
        ]
        for archetype, count in sorted(self.template_count_by_archetype().items()):
            lines.append(f"  {archetype:<45} {count} template(s)")

        tier_counts = {1: 0, 2: 0, 3: 0}
        for template in self.list_templates():
            tier_counts[template.complexity_tier] = tier_counts.get(template.complexity_tier, 0) + 1
        lines.append("")
        lines.append("Templates per complexity tier:")
        for tier, count in sorted(tier_counts.items()):
            lines.append(f"  Tier {tier}: {count} template(s)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Backward-compatible wrappers during naming transition
    # ------------------------------------------------------------------

    def load_all_templates_from_jsonl_file(self, jsonl_file_path: Path) -> int:
        return self.load_templates_from_jsonl_file(jsonl_file_path)

    def load_all_templates_from_directory(self, directory_path: Path) -> int:
        return self.load_templates_from_directory(directory_path)

    def _register_single_template(self, template: ClinicalBundleTemplate) -> None:
        self._register_template(template)

    def get_all_templates(self) -> list[ClinicalBundleTemplate]:
        return self.list_templates()

    def get_templates_by_archetype(self, archetype: str) -> list[ClinicalBundleTemplate]:
        return self.find_templates_by_archetype(archetype)

    def get_templates_by_complexity_tier(
        self, complexity_tier: int
    ) -> list[ClinicalBundleTemplate]:
        return self.find_templates_by_complexity_tier(complexity_tier)

    def find_template_by_id(self, template_id: str) -> ClinicalBundleTemplate | None:
        return self.get_template_by_id(template_id)

    def all_known_archetypes(self) -> list[str]:
        return self.list_archetypes()

    def summary_report(self) -> str:
        return self.build_summary_report()
