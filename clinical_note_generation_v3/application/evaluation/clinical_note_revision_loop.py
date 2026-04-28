"""
Revision loop for seeded clinical note generation.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from clinical_note_generation_v3.application.note_generation.seeded_clinical_note_generator import (
    SeededClinicalNoteGenerator,
)
from clinical_note_generation_v3.core.models.constraints import ClinicalBundleSemanticConstraints
from clinical_note_generation_v3.core.models.evaluation import (
    AcceptedClinicalNoteResult,
    NoteEvaluationCritiqueResult,
    RejectedClinicalNoteResult,
    RevisionAttemptRecord,
)
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote


_LATERALITY_OPPOSITE: dict[str, str] = {
    "left": "right",
    "right": "left",
}

_ENCOUNTER_WRONG_STAGE_TERMS: dict[str, list[str]] = {
    "initial": [
        "follow-up visit",
        "follow up visit",
        "return visit",
        "subsequent encounter",
        "established patient follow",
    ],
    "subsequent": [
        "new onset",
        "first presentation",
        "presenting for the first time",
        "initial evaluation for new",
    ],
    "sequela": [
        "acute injury",
        "new injury",
        "fresh injury",
        "acute fracture",
        "initial injury presentation",
    ],
}

_TEMPORAL_CONTRAINDICATIONS: dict[str, list[str]] = {
    "acute": [
        "chronic condition",
        "long-standing",
        "long standing",
        "chronic management of",
        "years of poorly controlled",
    ],
    "chronic": [
        "new onset",
        "sudden onset",
        "acute onset",
        "first occurrence",
        "first episode of",
    ],
    "in_remission": [
        "active symptoms",
        "active episode",
        "current episode",
        "current exacerbation",
        "currently symptomatic",
    ],
    "history": [
        "current active",
        "active diagnosis",
        "currently experiencing",
        "current episode of",
    ],
    "recurrent": [],
}

_CONDITION_STOPWORDS: frozenset[str] = frozenset(
    {
        "with",
        "without",
        "and",
        "the",
        "for",
        "due",
        "from",
        "type",
        "into",
        "over",
        "under",
        "that",
        "this",
        "about",
        "which",
        "other",
        "unspecified",
        "initial",
        "subsequent",
        "sequela",
        "encounter",
        "bilateral",
        "right",
        "left",
        "acute",
        "chronic",
    }
)


def _extract_key_terms(condition_name: str) -> list[str]:
    words = re.findall(r"[a-z]+", condition_name.lower())
    return [w for w in words if len(w) >= 5 and w not in _CONDITION_STOPWORDS]


class ClinicalNoteRevisionLoop:
    """
    Runs up to two targeted revision attempts for a generated note.
    """

    def __init__(
        self,
        *,
        seeded_clinical_note_generator: SeededClinicalNoteGenerator,
        max_revision_attempts: int = 2,
    ) -> None:
        self._seeded_clinical_note_generator = seeded_clinical_note_generator
        self._max_revision_attempts = max_revision_attempts

    def run_revision_loop(
        self,
        *,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        initial_generated_clinical_note: GeneratedClinicalNote,
        initial_note_evaluation_result: NoteEvaluationCritiqueResult,
        evaluate_generated_clinical_note: Callable[
            [GeneratedClinicalNote], NoteEvaluationCritiqueResult
        ],
    ) -> AcceptedClinicalNoteResult | RejectedClinicalNoteResult:
        revision_history: list[RevisionAttemptRecord] = []

        if initial_note_evaluation_result.final_decision == "accept":
            return AcceptedClinicalNoteResult(
                seeded_bundle=bundle_semantic_constraints.seeded_bundle,
                bundle_note_writing_constraints=bundle_semantic_constraints,
                accepted_note=initial_generated_clinical_note,
                final_critique=initial_note_evaluation_result,
                revision_history=[],
                required_revision=False,
            )

        current_generated_clinical_note = initial_generated_clinical_note
        current_note_evaluation_result = initial_note_evaluation_result

        for revision_attempt_number in range(1, self._max_revision_attempts + 1):
            if current_note_evaluation_result.final_decision != "revise":
                break

            revised_clinical_note = self._seeded_clinical_note_generator.generate_revised_clinical_note(
                bundle_semantic_constraints=bundle_semantic_constraints,
                previous_generated_clinical_note=current_generated_clinical_note,
                revision_targets=current_note_evaluation_result.revision_targets,
                metadata_constraint_violations=current_note_evaluation_result.icd_constraint_violations,
                generation_attempt_number=revision_attempt_number + 1,
            )

            post_revision_drift_violations = self._detect_post_revision_drift(
                bundle_semantic_constraints=bundle_semantic_constraints,
                previous_generated_clinical_note=current_generated_clinical_note,
                revised_clinical_note=revised_clinical_note,
            )
            if post_revision_drift_violations:
                revised_note_evaluation_result = NoteEvaluationCritiqueResult(
                    deterministic_precheck_outcome=current_note_evaluation_result.deterministic_precheck_outcome,
                    condition_support_verification_outcome=current_note_evaluation_result.condition_support_verification_outcome,
                    general_quality_rubric_scores=current_note_evaluation_result.general_quality_rubric_scores,
                    icd_constraint_alignment_scores=current_note_evaluation_result.icd_constraint_alignment_scores,
                    icd_constraint_violations=current_note_evaluation_result.icd_constraint_violations,
                    hard_fail_reasons=post_revision_drift_violations,
                    revision_targets=[],
                    combined_score=current_note_evaluation_result.combined_score,
                    final_decision="reject",
                )
            else:
                revised_note_evaluation_result = evaluate_generated_clinical_note(
                    revised_clinical_note
                )

            revision_history.append(
                RevisionAttemptRecord(
                    revision_attempt_number=revision_attempt_number,
                    revision_targets_sent_to_generator=list(
                        current_note_evaluation_result.revision_targets
                    ),
                    icd_violations_sent_to_generator=list(
                        current_note_evaluation_result.icd_constraint_violations
                    ),
                    revised_note=revised_clinical_note,
                    critique_after_revision=revised_note_evaluation_result,
                )
            )

            current_generated_clinical_note = revised_clinical_note
            current_note_evaluation_result = revised_note_evaluation_result

            if current_note_evaluation_result.final_decision == "accept":
                return AcceptedClinicalNoteResult(
                    seeded_bundle=bundle_semantic_constraints.seeded_bundle,
                    bundle_note_writing_constraints=bundle_semantic_constraints,
                    accepted_note=current_generated_clinical_note,
                    final_critique=current_note_evaluation_result,
                    revision_history=revision_history,
                    required_revision=True,
                )

        return RejectedClinicalNoteResult(
            seeded_bundle=bundle_semantic_constraints.seeded_bundle,
            rejected_note=current_generated_clinical_note,
            final_critique=current_note_evaluation_result,
            primary_rejection_reason=self._select_primary_rejection_reason(
                current_note_evaluation_result
            ),
            all_rejection_reasons=list(current_note_evaluation_result.hard_fail_reasons),
            revision_history=revision_history,
        )

    def _detect_post_revision_drift(
        self,
        *,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        previous_generated_clinical_note: GeneratedClinicalNote,
        revised_clinical_note: GeneratedClinicalNote,
    ) -> list[str]:
        violations: list[str] = []

        if (
            previous_generated_clinical_note.fake_patient_name
            != revised_clinical_note.fake_patient_name
        ):
            violations.append("Revision changed the fake patient name, which must remain fixed.")
        if (
            previous_generated_clinical_note.fake_patient_mrn
            != revised_clinical_note.fake_patient_mrn
        ):
            violations.append("Revision changed the fake patient MRN, which must remain fixed.")
        if (
            previous_generated_clinical_note.fake_patient_date_of_birth
            != revised_clinical_note.fake_patient_date_of_birth
        ):
            violations.append(
                "Revision changed the fake patient date of birth, which must remain fixed."
            )

        note_lower = revised_clinical_note.note_text.lower()

        for code_constraint in bundle_semantic_constraints.codes_with_laterality_constraints():
            required = code_constraint.laterality
            if not required or required == "bilateral":
                continue
            opposite = _LATERALITY_OPPOSITE.get(required)
            if opposite and opposite in note_lower and required not in note_lower:
                violations.append(
                    f"Laterality drift for {code_constraint.icd_code}: "
                    f"required '{required}' but revised note contains '{opposite}' without '{required}'."
                )

        for code_constraint in bundle_semantic_constraints.codes_with_encounter_type_constraints():
            required_stage = code_constraint.encounter_type
            wrong_terms = _ENCOUNTER_WRONG_STAGE_TERMS.get(required_stage or "", [])
            for term in wrong_terms:
                if term in note_lower:
                    violations.append(
                        f"Encounter stage drift for {code_constraint.icd_code}: "
                        f"required '{required_stage}' stage but revised note implies '{term}'."
                    )
                    break

        for code_constraint in bundle_semantic_constraints.per_code_note_writing_constraints:
            for temporal_state in code_constraint.temporal_states:
                contraindications = _TEMPORAL_CONTRAINDICATIONS.get(temporal_state, [])
                for term in contraindications:
                    if term in note_lower:
                        violations.append(
                            f"Temporal state drift for {code_constraint.icd_code}: "
                            f"required '{temporal_state}' but revised note implies '{term}'."
                        )
                        break

        for condition_name in bundle_semantic_constraints.seeded_bundle.active_condition_names:
            key_terms = _extract_key_terms(condition_name)
            if key_terms and not any(term in note_lower for term in key_terms):
                violations.append(
                    f"Active condition '{condition_name}' appears to have been dropped from the revised note."
                )

        return violations

    def _select_primary_rejection_reason(
        self,
        note_evaluation_result: NoteEvaluationCritiqueResult,
    ) -> str:
        if note_evaluation_result.hard_fail_reasons:
            return note_evaluation_result.hard_fail_reasons[0]
        return "Clinical note did not reach acceptance after the allowed revision attempts."
