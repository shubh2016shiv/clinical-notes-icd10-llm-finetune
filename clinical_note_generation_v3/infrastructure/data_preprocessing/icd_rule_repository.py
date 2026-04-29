"""
Official ICD-10-CM tabular rule repository.

The production pipeline requires the structured April 1, 2026 ICD-10-CM XML
tabular file.  XSD/PDF assets are useful for humans, but they are not enough
for deterministic code-set validation.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from clinical_note_generation_v3.config.settings import V3PipelineSettings
from clinical_note_generation_v3.core.models.icd_rules import (
    ICD_TABULAR_NOTE_TYPES,
    IcdRuleCacheManifest,
    IcdTabularRuleNode,
)


MANDATORY_XML_REMEDIATION = (
    "Official ICD-10-CM tabular XML is required for medically sound code-set "
    "validation. Download icd10cm-April-1-2026-XML.zip from the official CDC "
    "release and extract the tabular XML into official_icd10cm_2026_april_1."
)

_CODE_LIKE_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?$")


class MissingIcdRuleDataError(RuntimeError):
    """Raised when structured ICD XML rule data is unavailable."""


class IcdRuleRepository:
    """
    In-memory repository of ICD-10-CM tabular rules loaded from official XML.
    """

    _CACHE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        nodes_by_code: dict[str, IcdTabularRuleNode],
        source_xml_path: Path,
    ) -> None:
        self._nodes_by_code = nodes_by_code
        self.source_xml_path = source_xml_path

    @property
    def total_nodes(self) -> int:
        return len(self._nodes_by_code)

    @classmethod
    def from_default_settings(
        cls,
        settings: V3PipelineSettings | None = None,
    ) -> "IcdRuleRepository":
        settings = settings or V3PipelineSettings()
        xml_path = _find_tabular_xml_file(settings)
        cache_path = settings.icd_rule_cache_path
        if cache_path.exists():
            cached_repository = cls.from_cache_file(cache_path=cache_path, source_xml_path=xml_path)
            if cached_repository.total_nodes > 0:
                return cached_repository
        return cls.from_xml_file(xml_path)

    @classmethod
    def from_xml_file(cls, xml_path: Path) -> "IcdRuleRepository":
        if not xml_path.exists():
            raise MissingIcdRuleDataError(MANDATORY_XML_REMEDIATION)

        root = ET.parse(xml_path).getroot()
        nodes_by_code: dict[str, IcdTabularRuleNode] = {}
        inherited_notes: dict[str, list[str]] = {
            note_type: [] for note_type in ICD_TABULAR_NOTE_TYPES
        }

        for child in root:
            if _local_name(child.tag) in {"chapter", "section", "diag"}:
                _walk_tabular_element(
                    child,
                    parent_code=None,
                    ancestor_codes=[],
                    inherited_notes_by_type=inherited_notes,
                    nodes_by_code=nodes_by_code,
                )

        return cls(nodes_by_code=nodes_by_code, source_xml_path=xml_path)

    @classmethod
    def from_cache_file(
        cls,
        *,
        cache_path: Path,
        source_xml_path: Path,
    ) -> "IcdRuleRepository":
        if not cache_path.exists():
            raise MissingIcdRuleDataError(MANDATORY_XML_REMEDIATION)
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        manifest = payload.get("manifest", {})
        if manifest.get("schema_version") != cls._CACHE_SCHEMA_VERSION:
            raise MissingIcdRuleDataError(
                f"{MANDATORY_XML_REMEDIATION} Existing rule cache has an unsupported schema."
            )
        nodes_by_code = {
            code: IcdTabularRuleNode.model_validate(node_payload)
            for code, node_payload in payload.get("nodes_by_code", {}).items()
        }
        return cls(nodes_by_code=nodes_by_code, source_xml_path=source_xml_path)

    def write_cache_file(self, cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = IcdRuleCacheManifest(
            schema_version=self._CACHE_SCHEMA_VERSION,
            source_xml_filename=self.source_xml_path.name,
            source_xml_size=self.source_xml_path.stat().st_size,
            node_count=self.total_nodes,
        )
        payload = {
            "manifest": manifest.model_dump(mode="json"),
            "nodes_by_code": {
                code: node.model_dump(mode="json") for code, node in self._nodes_by_code.items()
            },
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def get_rule_node(self, code: str) -> IcdTabularRuleNode | None:
        normalized_code = normalize_icd_code(code)
        if normalized_code in self._nodes_by_code:
            return self._nodes_by_code[normalized_code]

        # Billable codes may be below the explicit tabular category.  Walk up
        # from specific code to category so ancestor/category notes still apply.
        undotted = normalized_code.replace(".", "")
        for end_index in range(len(undotted) - 1, 2, -1):
            candidate = normalize_icd_code(undotted[:end_index])
            if candidate in self._nodes_by_code:
                return self._nodes_by_code[candidate]
        return self._nodes_by_code.get(undotted[:3])

    def applicable_notes(self, code: str, note_type: str) -> list[str]:
        node = self.get_rule_node(code)
        return node.applicable_notes(note_type) if node else []

    def all_nodes(self) -> list[IcdTabularRuleNode]:
        return list(self._nodes_by_code.values())


def normalize_icd_code(code: str) -> str:
    cleaned = code.strip().upper().replace(" ", "")
    if not cleaned:
        return ""
    if "." in cleaned or len(cleaned) <= 3:
        return cleaned
    return f"{cleaned[:3]}.{cleaned[3:]}"


def _find_tabular_xml_file(settings: V3PipelineSettings) -> Path:
    exact_path = settings.official_icd_directory / settings.icd_tabular_xml_filename
    if exact_path.exists():
        return exact_path
    candidates = [
        path
        for path in settings.official_icd_directory.rglob("*.xml")
        if "tabular" in path.name.lower() and not path.name.lower().endswith(".xsd")
    ]
    if candidates:
        return sorted(candidates, key=lambda path: path.name.lower())[0]
    raise MissingIcdRuleDataError(MANDATORY_XML_REMEDIATION)


def _walk_tabular_element(
    element: ET.Element,
    *,
    parent_code: str | None,
    ancestor_codes: list[str],
    inherited_notes_by_type: dict[str, list[str]],
    nodes_by_code: dict[str, IcdTabularRuleNode],
) -> None:
    direct_notes_by_type = _extract_direct_notes_by_type(element)
    element_code = _extract_code_from_element(element)
    description = _direct_child_text(element, "desc")

    current_inherited_notes = _merge_notes(inherited_notes_by_type, direct_notes_by_type)
    current_parent_code = parent_code
    current_ancestor_codes = list(ancestor_codes)

    if element_code is not None:
        normalized_code = normalize_icd_code(element_code)
        if _looks_like_single_code(normalized_code):
            nodes_by_code[normalized_code] = IcdTabularRuleNode(
                code=normalized_code,
                description=description,
                parent_code=parent_code,
                ancestor_codes=list(ancestor_codes),
                direct_notes_by_type=direct_notes_by_type,
                ancestor_notes_by_type={
                    key: list(values) for key, values in inherited_notes_by_type.items()
                },
            )
            current_parent_code = normalized_code
            current_ancestor_codes = [*ancestor_codes, normalized_code]

    for child in element:
        if _local_name(child.tag) in {"chapter", "section", "diag"}:
            _walk_tabular_element(
                child,
                parent_code=current_parent_code,
                ancestor_codes=current_ancestor_codes,
                inherited_notes_by_type=current_inherited_notes,
                nodes_by_code=nodes_by_code,
            )


def _extract_code_from_element(element: ET.Element) -> str | None:
    code_text = _direct_child_text(element, "name")
    if not code_text:
        return None
    first_token = code_text.strip().split()[0]
    if "-" in first_token:
        return None
    return first_token


def _extract_direct_notes_by_type(element: ET.Element) -> dict[str, list[str]]:
    notes_by_type: dict[str, list[str]] = {note_type: [] for note_type in ICD_TABULAR_NOTE_TYPES}
    for child in element:
        child_name = _local_name(child.tag)
        if child_name not in ICD_TABULAR_NOTE_TYPES:
            continue
        notes_by_type[child_name].extend(_extract_note_texts(child))
    return {key: values for key, values in notes_by_type.items() if values}


def _extract_note_texts(note_container: ET.Element) -> list[str]:
    note_children = [child for child in note_container if _local_name(child.tag) == "note"]
    source_elements: Iterable[ET.Element] = note_children or [note_container]
    return [
        cleaned
        for cleaned in (_clean_text(" ".join(element.itertext())) for element in source_elements)
        if cleaned
    ]


def _merge_notes(
    inherited_notes_by_type: dict[str, list[str]],
    direct_notes_by_type: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged = {key: list(values) for key, values in inherited_notes_by_type.items()}
    for note_type, note_values in direct_notes_by_type.items():
        merged.setdefault(note_type, []).extend(note_values)
    return merged


def _direct_child_text(element: ET.Element, child_name: str) -> str:
    for child in element:
        if _local_name(child.tag) == child_name:
            return _clean_text(" ".join(child.itertext()))
    return ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _looks_like_single_code(code: str) -> bool:
    return bool(_CODE_LIKE_PATTERN.match(code))
