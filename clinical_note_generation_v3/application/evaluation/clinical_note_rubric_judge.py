"""
Rubric-based judge for seeded clinical notes.
"""

from __future__ import annotations

import logging

from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
    ConstraintViolationSeverity,
)
from clinical_note_generation_v3.core.models.evaluation import (
    ConditionSupportVerificationOutcome,
    IcdConstraintAlignmentRubricScores,
    IcdConstraintViolationDetail,
    NoteGeneralQualityRubricScores,
    RubricCriterionEvaluation,
)
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote
from clinical_note_generation_v3.core.ports.llm_generation_port import JSONGenerationClient
from clinical_note_generation_v3.prompt_specs.registry import (
    build_rubric_judge_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.rendering import compose_chat_prompt


class ClinicalNoteRubricJudge:
    """
    LLM-based rubric judge for note quality and metadata alignment.
    """

    def __init__(
        self,
        *,
        llm_json_generation_client: JSONGenerationClient,
    ) -> None:
        self._llm_json_generation_client = llm_json_generation_client

    def judge_generated_clinical_note(
        self,
        *,
        generated_clinical_note: GeneratedClinicalNote,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        condition_support_verification_outcome: ConditionSupportVerificationOutcome,
    ) -> tuple[
        NoteGeneralQualityRubricScores,
        IcdConstraintAlignmentRubricScores,
        list[IcdConstraintViolationDetail],
        str,
        str,
    ]:
        prompt_spec = build_rubric_judge_prompt_spec(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
            condition_support_verification_outcome=condition_support_verification_outcome,
        )
        judgment_prompt = compose_chat_prompt(
            system_prompt=prompt_spec.system_prompt,
            user_prompt=prompt_spec.user_prompt,
        )
        judgment_response = self._llm_json_generation_client.generate_json(
            judgment_prompt,
            response_schema=prompt_spec.response_schema,
        )

        _missing_or_empty = {
            k
            for k in {"general_quality_rubric_scores", "icd_constraint_alignment_scores"}
            if not judgment_response.get(k)
        }
        if _missing_or_empty:
            logging.getLogger(__name__).error(
                "Rubric judge response is missing or empty for keys: %s. "
                "Keys returned: %s. Response snippet: %.800s",
                _missing_or_empty,
                list(judgment_response.keys()),
                str(judgment_response),
            )
            return (None, None, [], prompt_spec.prompt_id, prompt_spec.prompt_version)

        general_quality_rubric_scores = self._build_general_quality_rubric_scores(
            judgment_response.get("general_quality_rubric_scores", {})
        )
        icd_constraint_alignment_scores = self._build_icd_constraint_alignment_scores(
            judgment_response.get("icd_constraint_alignment_scores", {})
        )
        icd_constraint_violations = self._build_icd_constraint_violations(
            judgment_response.get("icd_constraint_violations", [])
        )

        return (
            general_quality_rubric_scores,
            icd_constraint_alignment_scores,
            icd_constraint_violations,
            prompt_spec.prompt_id,
            prompt_spec.prompt_version,
        )

    def _build_general_quality_rubric_scores(
        self,
        response_payload: dict,
    ) -> NoteGeneralQualityRubricScores:
        return NoteGeneralQualityRubricScores(
            condition_support_coverage=self._build_single_rubric_criterion(
                response_payload.get("condition_support_coverage")
            ),
            internal_consistency=self._build_single_rubric_criterion(
                response_payload.get("internal_consistency")
            ),
            clinical_realism=self._build_single_rubric_criterion(
                response_payload.get("clinical_realism")
            ),
            encounter_structure_quality=self._build_single_rubric_criterion(
                response_payload.get("encounter_structure_quality")
            ),
            evidence_specificity=self._build_single_rubric_criterion(
                response_payload.get("evidence_specificity")
            ),
            distractor_handling=self._build_single_rubric_criterion(
                response_payload.get("distractor_handling")
            ),
            assessment_to_plan_linkage=self._build_single_rubric_criterion(
                response_payload.get("assessment_to_plan_linkage")
            ),
            language_naturalness=self._build_single_rubric_criterion(
                response_payload.get("language_naturalness")
            ),
            diversity_contribution=self._build_single_rubric_criterion(
                response_payload.get("diversity_contribution")
            ),
            training_utility=self._build_single_rubric_criterion(
                response_payload.get("training_utility")
            ),
        )

    def _build_icd_constraint_alignment_scores(
        self,
        response_payload: dict,
    ) -> IcdConstraintAlignmentRubricScores:
        return IcdConstraintAlignmentRubricScores(
            specificity_alignment=self._build_optional_rubric_criterion(
                response_payload.get("specificity_alignment")
            ),
            laterality_alignment=self._build_optional_rubric_criterion(
                response_payload.get("laterality_alignment")
            ),
            encounter_stage_alignment=self._build_optional_rubric_criterion(
                response_payload.get("encounter_stage_alignment")
            ),
            temporal_state_alignment=self._build_optional_rubric_criterion(
                response_payload.get("temporal_state_alignment")
            ),
            with_without_complication_alignment=self._build_optional_rubric_criterion(
                response_payload.get("with_without_complication_alignment")
            ),
            chapter_style_alignment=self._build_optional_rubric_criterion(
                response_payload.get("chapter_style_alignment")
            ),
            must_not_imply_compliance=self._build_optional_rubric_criterion(
                response_payload.get("must_not_imply_compliance")
            ),
        )

    def _build_icd_constraint_violations(
        self,
        response_payload: list[dict],
    ) -> list[IcdConstraintViolationDetail]:
        violations: list[IcdConstraintViolationDetail] = []
        for violation_payload in response_payload:
            if not isinstance(violation_payload, dict):
                continue
            try:
                violations.append(
                    IcdConstraintViolationDetail(
                        violated_constraint_type=str(
                            violation_payload.get("violated_constraint_type", "")
                        ),
                        violation_severity=ConstraintViolationSeverity(
                            str(violation_payload.get("violation_severity", "advisory")).lower()
                        ),
                        what_was_expected=str(violation_payload.get("what_was_expected", "")),
                        what_was_observed_in_note=str(
                            violation_payload.get("what_was_observed_in_note", "")
                        ),
                        fix_instruction_for_revision_prompt=str(
                            violation_payload.get("fix_instruction_for_revision_prompt", "")
                        ),
                        source_icd_code=str(violation_payload.get("source_icd_code", "")),
                    )
                )
            except Exception:
                continue
        return violations

    def _build_single_rubric_criterion(
        self,
        criterion_payload: dict | None,
    ) -> RubricCriterionEvaluation:
        if not isinstance(criterion_payload, dict):
            return RubricCriterionEvaluation(
                score=0,
                rationale="No rubric evaluation was returned for this criterion.",
                is_missing=True,
            )
        raw_score = criterion_payload.get("score")
        return RubricCriterionEvaluation(
            score=int(raw_score) if raw_score is not None else 0,
            rationale=str(criterion_payload.get("rationale", "")),
            is_missing=False,
        )

    def _build_optional_rubric_criterion(
        self,
        criterion_payload: dict | None,
    ) -> RubricCriterionEvaluation | None:
        if criterion_payload is None:
            return None
        return self._build_single_rubric_criterion(criterion_payload)

    @classmethod
    def from_default_settings(cls) -> "ClinicalNoteRubricJudge":
        from clinical_note_generation_v3.infrastructure.llm_provider.llm_client_factory import (
            create_default_json_generation_client,
        )

        return cls(llm_json_generation_client=create_default_json_generation_client())
