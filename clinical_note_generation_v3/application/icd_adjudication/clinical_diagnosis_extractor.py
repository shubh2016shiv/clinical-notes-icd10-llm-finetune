"""
LLM-assisted diagnosis extraction for final ICD adjudication.
"""

from __future__ import annotations

from typing import cast

from clinical_note_generation_v3.core.models.constraints import ClinicalBundleSemanticConstraints
from clinical_note_generation_v3.core.models.icd_adjudication import (
    ClinicalDiagnosisMention,
    DiagnosisMentionStatus,
)
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote
from clinical_note_generation_v3.core.ports.llm_generation_port import JSONGenerationClient
from clinical_note_generation_v3.prompt_specs.registry import (
    build_diagnosis_extraction_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.rendering import compose_chat_prompt

_ALLOWED_STATUSES: set[str] = set(DiagnosisMentionStatus.__args__)


class ClinicalDiagnosisExtractor:
    """
    Extracts active and non-active diagnosis mentions from a clinical note.
    """

    def __init__(self, *, llm_json_generation_client: JSONGenerationClient) -> None:
        self._llm_json_generation_client = llm_json_generation_client

    def extract_diagnoses(
        self,
        *,
        generated_clinical_note: GeneratedClinicalNote,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    ) -> tuple[list[ClinicalDiagnosisMention], str, str, str]:
        prompt_spec = build_diagnosis_extraction_prompt_spec(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
        )
        prompt = compose_chat_prompt(
            system_prompt=prompt_spec.system_prompt,
            user_prompt=prompt_spec.user_prompt,
        )
        response = self._llm_json_generation_client.generate_json(
            prompt,
            response_schema=prompt_spec.response_schema,
        )
        mentions = [
            _build_diagnosis_mention(payload)
            for payload in response.get("diagnoses", [])
            if isinstance(payload, dict)
        ]
        rationale = str(response.get("extraction_rationale", ""))
        return mentions, rationale, prompt_spec.prompt_id, prompt_spec.prompt_version

    @property
    def configured_model_label(self) -> str:
        return (
            f"{self._llm_json_generation_client.provider_name}/"
            f"{self._llm_json_generation_client.model_name}"
        )

    @classmethod
    def from_default_settings(cls) -> "ClinicalDiagnosisExtractor":
        from clinical_note_generation_v3.infrastructure.llm_provider.llm_client_factory import (
            create_default_json_generation_client,
        )

        return cls(llm_json_generation_client=create_default_json_generation_client())


def _build_diagnosis_mention(payload: dict) -> ClinicalDiagnosisMention:
    raw_status = str(payload.get("status", "incidental")).strip().lower()
    status = cast(
        DiagnosisMentionStatus,
        raw_status if raw_status in _ALLOWED_STATUSES else "incidental",
    )
    # Trust the LLM's should_code judgment directly.  Z-category personal-history
    # and surveillance codes (e.g. Z85.x) have status="historical" yet ARE the
    # correct billable code for the encounter — the old `and status == "active"`
    # guard was silently overriding the LLM's correct should_code=True for those
    # cases and causing downstream cascade failures.
    should_code = bool(payload.get("should_code", False))
    return ClinicalDiagnosisMention(
        diagnosis_name=str(payload.get("diagnosis_name", "")).strip(),
        status=status,
        evidence=str(payload.get("evidence", "")).strip(),
        should_code=should_code,
        coding_rationale=str(payload.get("coding_rationale", "")).strip(),
    )
