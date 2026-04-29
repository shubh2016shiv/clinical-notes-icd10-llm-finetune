"""
ICD-10-CM tabular rule models.

These models are intentionally deterministic data containers.  They represent
the structured tabular notes loaded from the official ICD-10-CM XML release.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


ICD_TABULAR_NOTE_TYPES: tuple[str, ...] = (
    "inclusionTerm",
    "includes",
    "excludes1",
    "excludes2",
    "codeFirst",
    "useAdditionalCode",
    "codeAlso",
    "notes",
    "instruction",
    "sevenChrNote",
)


class IcdTabularRuleNode(BaseModel):
    """
    Structured tabular rule context for one ICD-10-CM category/code.
    """

    code: str
    description: str = ""
    parent_code: str | None = None
    ancestor_codes: list[str] = Field(default_factory=list)
    direct_notes_by_type: dict[str, list[str]] = Field(default_factory=dict)
    ancestor_notes_by_type: dict[str, list[str]] = Field(default_factory=dict)

    def direct_notes(self, note_type: str) -> list[str]:
        return list(self.direct_notes_by_type.get(note_type, []))

    def inherited_notes(self, note_type: str) -> list[str]:
        return list(self.ancestor_notes_by_type.get(note_type, []))

    def applicable_notes(self, note_type: str) -> list[str]:
        return self.inherited_notes(note_type) + self.direct_notes(note_type)


class IcdRuleCacheManifest(BaseModel):
    """
    Metadata written beside the generated deterministic ICD rule cache.
    """

    schema_version: int = 1
    source_xml_filename: str
    source_xml_size: int
    node_count: int
