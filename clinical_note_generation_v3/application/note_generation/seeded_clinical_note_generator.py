"""
Seeded clinical note generator.

Builds fake patient identity context, creates the seeded generation prompt,
calls the configured JSON-capable LLM client, and returns a
GeneratedClinicalNote with provenance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
)
from clinical_note_generation_v3.core.models.evaluation import IcdConstraintViolationDetail
from clinical_note_generation_v3.core.models.note import (
    GeneratedClinicalNote,
    GenerationModelInfo,
)
from clinical_note_generation_v3.core.ports.llm_generation_port import JSONGenerationClient
from clinical_note_generation_v3.prompt_specs.registry import (
    build_clinical_note_generation_prompt_spec,
    build_clinical_note_revision_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.rendering import compose_chat_prompt


@dataclass(frozen=True)
class FakePatientIdentity:
    """
    Fictitious patient identity used for synthetic note generation.
    """

    patient_name: str
    medical_record_number: str
    date_of_birth: str


class SeededClinicalNoteGenerator:
    """
    Generates one synthetic clinical note for a frozen seeded clinical bundle.
    """

    def __init__(
        self,
        *,
        llm_json_generation_client: JSONGenerationClient,
        generation_prompt_version: str = "seeded_clinical_note_v1",
        random_seed: int | None = None,
    ) -> None:
        self._llm_json_generation_client = llm_json_generation_client
        self._generation_prompt_version = generation_prompt_version
        self._random_number_generator = random.Random(random_seed)

    def generate_clinical_note(
        self,
        *,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        generation_attempt_number: int = 1,
    ) -> GeneratedClinicalNote:
        """
        Generate a structured synthetic clinical note for the given frozen bundle.
        """
        fake_patient_identity = self._create_fake_patient_identity(
            bundle_semantic_constraints=bundle_semantic_constraints
        )

        prompt_spec = build_clinical_note_generation_prompt_spec(
            bundle_semantic_constraints=bundle_semantic_constraints,
            fake_patient_name=fake_patient_identity.patient_name,
            fake_patient_medical_record_number=fake_patient_identity.medical_record_number,
            fake_patient_date_of_birth=fake_patient_identity.date_of_birth,
            generation_attempt_number=generation_attempt_number,
        )
        generation_prompt = compose_chat_prompt(
            system_prompt=prompt_spec.system_prompt,
            user_prompt=prompt_spec.user_prompt,
        )
        generated_note_response = self._llm_json_generation_client.generate_json(
            generation_prompt,
            response_schema=prompt_spec.response_schema,
        )

        return self._build_generated_clinical_note_from_response(
            generated_note_response=generated_note_response,
            fake_patient_identity=fake_patient_identity,
            prompt_id=prompt_spec.prompt_id,
            prompt_version=prompt_spec.prompt_version,
        )

    def generate_revised_clinical_note(
        self,
        *,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        previous_generated_clinical_note: GeneratedClinicalNote,
        revision_targets: list[str],
        metadata_constraint_violations: list[IcdConstraintViolationDetail],
        generation_attempt_number: int,
    ) -> GeneratedClinicalNote:
        """
        Generate a revised clinical note while preserving the fixed case definition.
        """
        metadata_fix_instructions = [
            metadata_constraint_violation.fix_instruction_for_revision_prompt
            for metadata_constraint_violation in metadata_constraint_violations
        ]

        prompt_spec = build_clinical_note_revision_prompt_spec(
            bundle_semantic_constraints=bundle_semantic_constraints,
            previous_clinical_note_text=previous_generated_clinical_note.note_text,
            revision_targets=revision_targets,
            metadata_fix_instructions=metadata_fix_instructions,
            fake_patient_name=previous_generated_clinical_note.fake_patient_name,
            fake_patient_medical_record_number=previous_generated_clinical_note.fake_patient_mrn,
            fake_patient_date_of_birth=previous_generated_clinical_note.fake_patient_date_of_birth,
            generation_attempt_number=generation_attempt_number,
        )
        revision_prompt = compose_chat_prompt(
            system_prompt=prompt_spec.system_prompt,
            user_prompt=prompt_spec.user_prompt,
        )
        generated_note_response = self._llm_json_generation_client.generate_json(
            revision_prompt,
            response_schema=prompt_spec.response_schema,
        )

        return self._build_generated_clinical_note_from_response(
            generated_note_response=generated_note_response,
            fake_patient_identity=FakePatientIdentity(
                patient_name=previous_generated_clinical_note.fake_patient_name,
                medical_record_number=previous_generated_clinical_note.fake_patient_mrn,
                date_of_birth=previous_generated_clinical_note.fake_patient_date_of_birth,
            ),
            prompt_id=prompt_spec.prompt_id,
            prompt_version=prompt_spec.prompt_version,
        )

    def _create_fake_patient_identity(
        self,
        *,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    ) -> FakePatientIdentity:
        seeded_bundle = bundle_semantic_constraints.seeded_bundle

        patient_name = self._build_fake_patient_name()
        medical_record_number = self._build_fake_medical_record_number()
        date_of_birth = self._build_fake_date_of_birth_for_archetype(seeded_bundle.archetype)

        return FakePatientIdentity(
            patient_name=patient_name,
            medical_record_number=medical_record_number,
            date_of_birth=date_of_birth,
        )

    def _build_fake_patient_name(self) -> str:
        given_names = [
            "Maya",
            "Aarav",
            "Elena",
            "Daniel",
            "Priya",
            "Jordan",
            "Amara",
            "Isaac",
            "Nina",
            "Marcus",
            "Leah",
            "Rohan",
            "Sofia",
            "Ethan",
            "Anika",
            "David",
            "Tara",
            "Noah",
        ]
        family_names = [
            "Patel",
            "Nguyen",
            "Carter",
            "Sharma",
            "Brooks",
            "Ibrahim",
            "Reed",
            "Banerjee",
            "Lopez",
            "Khan",
            "Morris",
            "Singh",
            "Foster",
            "Das",
            "Walker",
            "Ali",
            "Price",
            "Raman",
        ]
        return (
            f"{self._random_number_generator.choice(given_names)} "
            f"{self._random_number_generator.choice(family_names)}"
        )

    def _build_fake_medical_record_number(self) -> str:
        random_digits = "".join(str(self._random_number_generator.randint(0, 9)) for _ in range(8))
        return f"MRN-{random_digits}"

    def _build_fake_date_of_birth_for_archetype(self, bundle_archetype: str) -> str:
        if "pediatric" in bundle_archetype.lower():
            minimum_age_years = 2
            maximum_age_years = 12
        else:
            minimum_age_years = 18
            maximum_age_years = 84

        selected_age_years = self._random_number_generator.randint(
            minimum_age_years,
            maximum_age_years,
        )
        selected_extra_days = self._random_number_generator.randint(0, 364)
        approximate_birth_date = date.today() - timedelta(
            days=(selected_age_years * 365) + selected_extra_days
        )
        return approximate_birth_date.isoformat()

    def _build_generated_clinical_note_from_response(
        self,
        *,
        generated_note_response: dict,
        fake_patient_identity: FakePatientIdentity,
        prompt_id: str,
        prompt_version: str,
    ) -> GeneratedClinicalNote:
        clinical_note_text = generated_note_response.get("clinical_note_text")
        if not isinstance(clinical_note_text, str) or not clinical_note_text.strip():
            raise ValueError(
                "Seeded clinical note generation response did not include a non-empty "
                "'clinical_note_text' field."
            )

        generation_model_info = GenerationModelInfo(
            provider_name=self._llm_json_generation_client.provider_name,
            model_name=self._llm_json_generation_client.model_name,
        )

        return GeneratedClinicalNote(
            note_text=clinical_note_text.strip(),
            generation_prompt_id=prompt_id,
            generation_prompt_version=prompt_version,
            generation_model_info=generation_model_info,
            fake_patient_name=fake_patient_identity.patient_name,
            fake_patient_mrn=fake_patient_identity.medical_record_number,
            fake_patient_date_of_birth=fake_patient_identity.date_of_birth,
        )

    @classmethod
    def from_default_settings(
        cls,
        *,
        generation_prompt_version: str = "seeded_clinical_note_v1",
        random_seed: int | None = None,
    ) -> "SeededClinicalNoteGenerator":
        """
        Create a generator wired to the default V3 provider configuration.
        """
        from clinical_note_generation_v3.config.settings import V3PipelineSettings
        from clinical_note_generation_v3.infrastructure.llm_provider.llm_client_factory import (
            create_default_json_generation_client,
        )

        settings = V3PipelineSettings()

        return cls(
            llm_json_generation_client=create_default_json_generation_client(settings),
            generation_prompt_version=generation_prompt_version,
            random_seed=random_seed if random_seed is not None else settings.random_seed,
        )
