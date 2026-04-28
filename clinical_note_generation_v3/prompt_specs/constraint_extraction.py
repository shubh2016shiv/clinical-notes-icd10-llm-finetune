"""
Centralized prompt spec for ICD semantic constraint extraction escalation.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.constraints import ExtractedSemanticSignal
from clinical_note_generation_v3.prompt_specs.contracts import (
    CONSTRAINT_EXTRACTION_RESPONSE_SCHEMA,
    JSON_ONLY_OUTPUT_CONTRACT,
    PromptSpec,
)

CONSTRAINT_EXTRACTION_PROMPT_ID = "icd_constraint_extraction_escalation"
CONSTRAINT_EXTRACTION_PROMPT_VERSION = "v1_centralized"


def build_constraint_extraction_prompt_spec(
    *,
    icd_code: str,
    long_description: str,
    short_description: str,
    signals_already_extracted: list[ExtractedSemanticSignal],
) -> PromptSpec:
    if signals_already_extracted:
        signal_summary = "\n".join(
            f"- {signal.signal_type}: {signal.extracted_value}"
            for signal in signals_already_extracted
        )
    else:
        signal_summary = "- (none)"

    system_prompt = (
        "You are a clinical documentation specialist and expert ICD-10-CM coder. "
        "Derive note-writing obligations and prohibitions for a note writer. "
        + JSON_ONLY_OUTPUT_CONTRACT
    )
    user_prompt = f"""
ICD-10-CM Code: {icd_code}
Long description: {long_description}
Short description: {short_description}

Signals already extracted (do not repeat them):
{signal_summary}

Return only a JSON object in this format:
{{
  "must_include_in_note": ["<concrete clinical element>"],
  "must_not_imply_in_note": ["<concrete prohibition>"]
}}

Rules:
1. must_include_in_note should contain 2 to 5 concrete clinical documentation elements.
2. must_not_imply_in_note should contain 1 to 3 concrete prohibitions that would contradict the diagnosis.
3. Do not repeat signals already extracted above.
""".strip()
    return PromptSpec(
        prompt_id=CONSTRAINT_EXTRACTION_PROMPT_ID,
        prompt_version=CONSTRAINT_EXTRACTION_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=CONSTRAINT_EXTRACTION_RESPONSE_SCHEMA,
    )
