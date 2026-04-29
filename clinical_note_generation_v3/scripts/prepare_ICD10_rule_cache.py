#!/usr/bin/env python3
"""
Prepare the ICD-10-CM tabular rule cache used by v3 code-set validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_note_generation_v3.config.settings import V3PipelineSettings  # noqa: E402
from clinical_note_generation_v3.infrastructure.data_preprocessing.icd_rule_repository import (  # noqa: E402
    IcdRuleRepository,
    MANDATORY_XML_REMEDIATION,
    MissingIcdRuleDataError,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the v3 ICD-10-CM tabular rule cache from official XML."
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=None,
        help="Override the official tabular XML path.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Override the generated rule cache output path.",
    )
    args = parser.parse_args()

    settings = V3PipelineSettings()
    xml_path = args.xml_path or _find_default_tabular_xml(settings)
    cache_path = args.cache_path or settings.icd_rule_cache_path

    print("Clinical Note Generation v3 ICD-10-CM Rule Cache Build")
    print("=" * 60)
    print(f"Tabular XML: {xml_path}")
    print(f"Cache path: {cache_path}")
    print("=" * 60)

    try:
        repository = IcdRuleRepository.from_xml_file(xml_path)
    except MissingIcdRuleDataError:
        print(f"ERROR: {MANDATORY_XML_REMEDIATION}")
        return 1

    repository.write_cache_file(cache_path)
    print(f"Built ICD rule cache with {repository.total_nodes} tabular nodes.")
    return 0


def _find_default_tabular_xml(settings: V3PipelineSettings) -> Path:
    exact_path = settings.official_icd_tabular_xml_path
    if exact_path.exists():
        return exact_path
    candidates = [
        path
        for path in settings.official_icd_directory.rglob("*.xml")
        if "tabular" in path.name.lower() and not path.name.lower().endswith(".xsd")
    ]
    return sorted(candidates, key=lambda path: path.name.lower())[0] if candidates else exact_path


if __name__ == "__main__":
    raise SystemExit(main())
