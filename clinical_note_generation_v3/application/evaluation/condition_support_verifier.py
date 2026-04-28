"""
Condition support verifier.

Checks whether a generated note adequately expresses the seeded active
conditions and whether it drifts into unsupported additional diagnoses.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
)
from clinical_note_generation_v3.core.models.evaluation import (
    ConditionSupportVerificationOutcome,
)
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote
from clinical_note_generation_v3.core.ports.llm_generation_port import JSONGenerationClient
from clinical_note_generation_v3.prompt_specs.registry import (
    build_condition_support_verifier_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.rendering import compose_chat_prompt


class ConditionSupportVerifier:
    """
    LLM-assisted verifier for seeded condition support coverage.
    """

    def __init__(
        self,
        *,
        llm_json_generation_client: JSONGenerationClient,
    ) -> None:
        self._llm_json_generation_client = llm_json_generation_client

    def verify_condition_support(
        self,
        *,
        generated_clinical_note: GeneratedClinicalNote,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    ) -> ConditionSupportVerificationOutcome:
        prompt_spec = build_condition_support_verifier_prompt_spec(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
        )
        verification_prompt = compose_chat_prompt(
            system_prompt=prompt_spec.system_prompt,
            user_prompt=prompt_spec.user_prompt,
        )
        verification_response = self._llm_json_generation_client.generate_json(
            verification_prompt,
            response_schema=prompt_spec.response_schema,
        )

        return ConditionSupportVerificationOutcome(
            outcome=str(verification_response.get("outcome", "fail")),
            under_supported_conditions=list(
                verification_response.get("under_supported_conditions", [])
            ),
            unsupported_implied_conditions=list(
                verification_response.get("unsupported_implied_conditions", [])
            ),
            history_or_negation_drift_detected=bool(
                verification_response.get("history_or_negation_drift_detected", False)
            ),
            verifier_notes=str(verification_response.get("verifier_notes", "")),
            verifier_prompt_id=prompt_spec.prompt_id,
            verifier_prompt_version=prompt_spec.prompt_version,
        )

    @classmethod
    def from_default_settings(cls) -> "ConditionSupportVerifier":
        from clinical_note_generation_v3.infrastructure.llm_provider.llm_client_factory import (
            create_default_json_generation_client,
        )

        return cls(llm_json_generation_client=create_default_json_generation_client())
