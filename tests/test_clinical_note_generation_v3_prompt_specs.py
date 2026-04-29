from __future__ import annotations

from clinical_note_generation_v3.application.evaluation.clinical_note_rubric_judge import (
    ClinicalNoteRubricJudge,
)
from clinical_note_generation_v3.application.evaluation.condition_support_verifier import (
    ConditionSupportVerifier,
)
from clinical_note_generation_v3.application.icd_resolution.icd_condition_to_code_resolver import (
    IcdConditionToCodeResolver,
)
from clinical_note_generation_v3.application.note_generation.seeded_clinical_note_generator import (
    SeededClinicalNoteGenerator,
)
from clinical_note_generation_v3.core.models.bundle import (
    ResolvedConditionCode,
    SeededClinicalBundle,
)
from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
    IcdCodeSemanticConstraints,
)
from clinical_note_generation_v3.core.models.evaluation import (
    ConditionSupportVerificationOutcome,
)
from clinical_note_generation_v3.core.models.icd_codes import CandidateCode, ICDCodeRecord
from clinical_note_generation_v3.core.services.icd_code_semantic_constraint_extractor import (
    IcdCodeSemanticConstraintExtractor,
)
from clinical_note_generation_v3.prompt_specs.note_generation import (
    GENERATION_PROMPT_ID,
)
from clinical_note_generation_v3.prompt_specs.registry import (
    build_clinical_note_generation_prompt_spec,
    build_clinical_note_revision_prompt_spec,
    build_condition_support_verifier_prompt_spec,
    build_constraint_extraction_prompt_spec,
    build_diagnosis_extraction_prompt_spec,
    build_icd_resolution_prompt_spec,
    build_rubric_judge_prompt_spec,
)


class StubJSONClient:
    provider_name = "stub"
    model_name = "stub-model"

    def __init__(self, response: dict) -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.last_schema: dict | None = None

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        self.last_prompt = prompt
        self.last_schema = response_schema
        return self.response


class StubRetriever:
    def __init__(self, candidates: list[CandidateCode]) -> None:
        self._candidates = candidates

    def retrieve(
        self, clinical_note: str, focus_terms: list[str], limit: int
    ) -> list[CandidateCode]:
        return self._candidates[:limit]


class StubRepository:
    def __init__(self, record: ICDCodeRecord) -> None:
        self._record = record

    def get_code(self, code: str):
        if code.strip().upper().replace(".", "") == self._record.code:
            return self._record
        return None


def _make_constraints() -> ClinicalBundleSemanticConstraints:
    seeded_bundle = SeededClinicalBundle(
        template_id="behavioral_health_09",
        archetype="behavioral_health_outpatient_followup",
        encounter_context="outpatient psychiatry follow-up visit",
        active_condition_names=["adjustment disorder with mixed anxiety and depressed mood"],
        resolved_conditions=[
            ResolvedConditionCode(
                condition_name="adjustment disorder with mixed anxiety and depressed mood",
                icd_code="F43.23",
                icd_short_description="Adj disorder w anx+depressd mood",
                icd_long_description="Adjustment disorder with mixed anxiety and depressed mood",
            )
        ],
        allowed_distractors=["suicidal ideation"],
        trap_patterns=["Do not imply psychosis."],
    )
    return ClinicalBundleSemanticConstraints(
        seeded_bundle=seeded_bundle,
        per_code_note_writing_constraints=[
            IcdCodeSemanticConstraints(
                icd_code="F43.23",
                icd_short_description="Adj disorder w anx+depressd mood",
                temporal_states=["chronic"],
                chapter_family="behavioral_health",
                must_include_in_note=[
                    "presenting psychiatric symptoms",
                    "mental status examination findings",
                ],
                must_not_imply_in_note=["psychotic features"],
            )
        ],
    )


def _make_generated_note():
    generator = SeededClinicalNoteGenerator(
        llm_json_generation_client=StubJSONClient(
            {"clinical_note_text": "Assessment and plan note."}
        ),
        random_seed=7,
    )
    return generator.generate_clinical_note(
        bundle_semantic_constraints=_make_constraints(),
        generation_attempt_number=1,
    )


def test_prompt_registry_builds_all_v3_prompt_families() -> None:
    constraints = _make_constraints()
    generated_note = _make_generated_note()
    verifier_outcome = ConditionSupportVerificationOutcome(outcome="pass")
    retrieved_candidates = [
        CandidateCode(
            candidate_id=1,
            code="F43.23",
            description="Adjustment disorder with mixed anxiety and depressed mood",
            source="bm25",
            score=0.99,
        )
    ]

    prompt_specs = [
        build_clinical_note_generation_prompt_spec(
            bundle_semantic_constraints=constraints,
            fake_patient_name="Test Person",
            fake_patient_medical_record_number="MRN-12345678",
            fake_patient_date_of_birth="1980-01-01",
            generation_attempt_number=1,
        ),
        build_clinical_note_revision_prompt_spec(
            bundle_semantic_constraints=constraints,
            previous_clinical_note_text="Old note",
            revision_targets=["Improve support."],
            metadata_fix_instructions=["None."],
            fake_patient_name="Test Person",
            fake_patient_medical_record_number="MRN-12345678",
            fake_patient_date_of_birth="1980-01-01",
            generation_attempt_number=2,
        ),
        build_condition_support_verifier_prompt_spec(
            generated_clinical_note=generated_note,
            bundle_semantic_constraints=constraints,
        ),
        build_rubric_judge_prompt_spec(
            generated_clinical_note=generated_note,
            bundle_semantic_constraints=constraints,
            condition_support_verification_outcome=verifier_outcome,
        ),
        build_icd_resolution_prompt_spec(
            condition_name="adjustment disorder with mixed anxiety and depressed mood",
            bundle_archetype=constraints.seeded_bundle.archetype,
            encounter_context=constraints.seeded_bundle.encounter_context,
            retrieved_candidates=retrieved_candidates,
        ),
        build_constraint_extraction_prompt_spec(
            icd_code="F43.23",
            long_description="Adjustment disorder with mixed anxiety and depressed mood",
            short_description="Adj disorder w anx+depressd mood",
            signals_already_extracted=[],
        ),
        build_diagnosis_extraction_prompt_spec(
            generated_clinical_note=generated_note,
            bundle_semantic_constraints=constraints,
        ),
    ]

    assert all(prompt_spec.prompt_id for prompt_spec in prompt_specs)
    assert all(prompt_spec.prompt_version for prompt_spec in prompt_specs)
    assert all(prompt_spec.response_schema is not None for prompt_spec in prompt_specs)


def test_generation_prompt_removes_visible_official_description_context() -> None:
    prompt_spec = build_clinical_note_generation_prompt_spec(
        bundle_semantic_constraints=_make_constraints(),
        fake_patient_name="Test Person",
        fake_patient_medical_record_number="MRN-12345678",
        fake_patient_date_of_birth="1980-01-01",
        generation_attempt_number=1,
    )

    assert "Official description context" not in prompt_spec.user_prompt
    assert "Adj disorder w anx+depressd mood" not in prompt_spec.user_prompt
    assert '"clinical_note_text"' in prompt_spec.system_prompt


def test_revision_prompt_includes_frozen_case_and_target_only_rules() -> None:
    prompt_spec = build_clinical_note_revision_prompt_spec(
        bundle_semantic_constraints=_make_constraints(),
        previous_clinical_note_text="Prior note text",
        revision_targets=["Strengthen symptom support."],
        metadata_fix_instructions=["None."],
        fake_patient_name="Test Person",
        fake_patient_medical_record_number="MRN-12345678",
        fake_patient_date_of_birth="1980-01-01",
        generation_attempt_number=2,
    )

    assert (
        "Laterality, encounter stage, temporal state, and patient identity"
        in prompt_spec.system_prompt
    )
    assert "Apply only the requested revision targets." in prompt_spec.system_prompt
    assert "Strengthen symptom support." in prompt_spec.user_prompt


def test_seeded_generator_records_centralized_prompt_lineage() -> None:
    stub_client = StubJSONClient({"clinical_note_text": "Assessment and plan note."})
    generator = SeededClinicalNoteGenerator(
        llm_json_generation_client=stub_client,
        random_seed=7,
    )

    generated_note = generator.generate_clinical_note(
        bundle_semantic_constraints=_make_constraints(),
        generation_attempt_number=1,
    )

    assert generated_note.generation_prompt_id == GENERATION_PROMPT_ID
    assert generated_note.generation_prompt_version
    assert stub_client.last_schema is not None
    assert "<system>" in (stub_client.last_prompt or "")


def test_support_verifier_records_prompt_lineage() -> None:
    stub_client = StubJSONClient(
        {
            "outcome": "pass",
            "under_supported_conditions": [],
            "unsupported_implied_conditions": [],
            "history_or_negation_drift_detected": False,
            "verifier_notes": "Looks supported.",
        }
    )
    verifier = ConditionSupportVerifier(llm_json_generation_client=stub_client)

    outcome = verifier.verify_condition_support(
        generated_clinical_note=_make_generated_note(),
        bundle_semantic_constraints=_make_constraints(),
    )

    assert outcome.verifier_prompt_id == "condition_support_verifier"
    assert outcome.verifier_prompt_version == "v1_centralized"


def test_rubric_judge_records_prompt_lineage() -> None:
    stub_client = StubJSONClient(
        {
            "general_quality_rubric_scores": {
                "condition_support_coverage": {"score": 2, "rationale": "ok"},
                "internal_consistency": {"score": 2, "rationale": "ok"},
                "clinical_realism": {"score": 2, "rationale": "ok"},
                "encounter_structure_quality": {"score": 2, "rationale": "ok"},
                "evidence_specificity": {"score": 2, "rationale": "ok"},
                "distractor_handling": {"score": 2, "rationale": "ok"},
                "assessment_to_plan_linkage": {"score": 2, "rationale": "ok"},
                "language_naturalness": {"score": 2, "rationale": "ok"},
                "diversity_contribution": {"score": 2, "rationale": "ok"},
                "training_utility": {"score": 2, "rationale": "ok"},
            },
            "icd_constraint_alignment_scores": {
                "specificity_alignment": None,
                "laterality_alignment": None,
                "encounter_stage_alignment": None,
                "temporal_state_alignment": {"score": 2, "rationale": "ok"},
                "with_without_complication_alignment": None,
                "chapter_style_alignment": {"score": 2, "rationale": "ok"},
                "must_not_imply_compliance": {"score": 2, "rationale": "ok"},
            },
            "icd_constraint_violations": [],
        }
    )
    judge = ClinicalNoteRubricJudge(llm_json_generation_client=stub_client)

    result = judge.judge_generated_clinical_note(
        generated_clinical_note=_make_generated_note(),
        bundle_semantic_constraints=_make_constraints(),
        condition_support_verification_outcome=ConditionSupportVerificationOutcome(outcome="pass"),
    )

    assert result[3] == "rubric_judge"
    assert result[4] == "v1_centralized"


def test_icd_resolver_records_prompt_lineage() -> None:
    stub_repository = StubRepository(
        ICDCodeRecord(
            code="F4323",
            is_billable=True,
            short_description="Adj disorder w anx+depressd mood",
            long_description="Adjustment disorder with mixed anxiety and depressed mood",
        )
    )
    stub_client = StubJSONClient(
        {
            "selected_icd_code": "F43.23",
            "selected_icd_description": "Adjustment disorder with mixed anxiety and depressed mood",
            "selection_rationale": "Best match.",
            "resolution_succeeded": True,
        }
    )
    resolver = IcdConditionToCodeResolver(
        icd_official_repository=stub_repository,
        hybrid_icd_candidate_retriever=StubRetriever(
            [
                CandidateCode(
                    candidate_id=1,
                    code="F43.23",
                    description="Adjustment disorder with mixed anxiety and depressed mood",
                    source="bm25",
                    score=0.99,
                )
            ]
        ),
        llm_json_client=stub_client,
    )

    resolved = resolver.resolve_condition_name_to_icd_code(
        condition_name="adjustment disorder with mixed anxiety and depressed mood",
        bundle_archetype="behavioral_health_outpatient_followup",
        encounter_context="outpatient psychiatry follow-up visit",
    )

    assert resolved.resolver_prompt_id == "icd_single_condition_resolution"
    assert resolved.resolver_prompt_version == "v1_centralized"


def test_constraint_extractor_records_prompt_lineage_when_escalated() -> None:
    extractor = IcdCodeSemanticConstraintExtractor(
        llm_escalation_client=StubJSONClient(
            {
                "must_include_in_note": ["documented symptom cluster"],
                "must_not_imply_in_note": ["psychosis"],
            }
        )
    )
    resolved_condition = ResolvedConditionCode(
        condition_name="rare condition",
        icd_code="Q99.9",
        icd_short_description="Rare condition NOS",
        icd_long_description="Rare condition not otherwise specified",
    )

    derived_signals, prompt_id, prompt_version = extractor._run_llm_escalation_and_return_signals(
        resolved_condition=resolved_condition,
        signals_already_extracted=[],
    )

    assert prompt_id == "icd_constraint_extraction_escalation"
    assert prompt_version == "v1_centralized"
    assert any(signal.extracted_value == "documented symptom cluster" for signal in derived_signals)
