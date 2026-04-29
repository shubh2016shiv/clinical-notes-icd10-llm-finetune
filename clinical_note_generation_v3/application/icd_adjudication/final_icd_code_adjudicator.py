"""
Final coder-style ICD-10-CM adjudication for generated notes.
"""

from __future__ import annotations

from clinical_note_generation_v3.application.icd_adjudication.clinical_diagnosis_extractor import (
    ClinicalDiagnosisExtractor,
)
from clinical_note_generation_v3.application.icd_resolution.icd_condition_to_code_resolver import (
    IcdConditionToCodeResolver,
    IcdResolutionFailedForConditionError,
)
from clinical_note_generation_v3.core.models.constraints import ClinicalBundleSemanticConstraints
from clinical_note_generation_v3.core.models.icd_adjudication import (
    FinalIcdCodeAdjudicationOutcome,
    IcdCodeSetValidationIssue,
    IcdCodeSetValidationOutcome,
)
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote
from clinical_note_generation_v3.core.services.icd_code_set_validator import (
    IcdCodeSetValidator,
)


class FinalIcdCodeAdjudicator:
    """
    Converts a final note into medically validated ICD-10-CM training labels.
    """

    def __init__(
        self,
        *,
        clinical_diagnosis_extractor: ClinicalDiagnosisExtractor,
        icd_condition_to_code_resolver: IcdConditionToCodeResolver,
        icd_code_set_validator: IcdCodeSetValidator,
    ) -> None:
        self._clinical_diagnosis_extractor = clinical_diagnosis_extractor
        self._icd_condition_to_code_resolver = icd_condition_to_code_resolver
        self._icd_code_set_validator = icd_code_set_validator

    def adjudicate_generated_note(
        self,
        *,
        generated_clinical_note: GeneratedClinicalNote,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    ) -> FinalIcdCodeAdjudicationOutcome:
        seeded_codes = list(bundle_semantic_constraints.seeded_bundle.icd_codes)
        (
            diagnosis_mentions,
            extraction_rationale,
            prompt_id,
            prompt_version,
        ) = self._clinical_diagnosis_extractor.extract_diagnoses(
            generated_clinical_note=generated_clinical_note,
            bundle_semantic_constraints=bundle_semantic_constraints,
        )

        # Use the LLM's should_code as the sole gate for what to resolve and code.
        # Requiring status == "active" here is a redundant and incorrect second filter
        # that excludes Z-category personal-history codes (status="historical") which
        # ARE the correct billable codes for the encounter (e.g. Z85.x in surveillance).
        active_reportable_mentions = [
            mention
            for mention in diagnosis_mentions
            if mention.should_code and mention.diagnosis_name
        ]
        resolved_codes: list[str] = []
        unresolved_active_diagnoses: list[str] = []
        resolution_rationales: list[str] = []

        for diagnosis_mention in active_reportable_mentions:
            try:
                resolved_condition = self._icd_condition_to_code_resolver.resolve_condition_name_to_icd_code(
                    condition_name=diagnosis_mention.diagnosis_name,
                    bundle_archetype=bundle_semantic_constraints.seeded_bundle.archetype,
                    encounter_context=bundle_semantic_constraints.seeded_bundle.encounter_context,
                )
            except IcdResolutionFailedForConditionError as error:
                unresolved_active_diagnoses.append(diagnosis_mention.diagnosis_name)
                resolution_rationales.append(str(error))
                continue

            resolved_codes.append(resolved_condition.icd_code)
            resolution_rationales.append(
                f"{diagnosis_mention.diagnosis_name} -> {resolved_condition.icd_code}"
            )

        code_set_validation_outcome = self._icd_code_set_validator.validate_codes(resolved_codes)
        code_set_validation_outcome = _add_adjudication_consistency_issues(
            validation_outcome=code_set_validation_outcome,
            seeded_codes=seeded_codes,
            unresolved_active_diagnoses=unresolved_active_diagnoses,
        )
        adjudicated_codes = list(code_set_validation_outcome.deduplicated_codes)
        seeded_code_set = set(
            self._icd_code_set_validator.validate_codes(seeded_codes).deduplicated_codes
        )
        adjudicated_code_set = set(adjudicated_codes)
        added_codes = sorted(adjudicated_code_set - seeded_code_set)
        removed_seeded_codes = sorted(seeded_code_set - adjudicated_code_set)
        revision_targets = _build_revision_targets(
            code_set_validation_outcome=code_set_validation_outcome,
            added_codes=added_codes,
            removed_seeded_codes=removed_seeded_codes,
            unresolved_active_diagnoses=unresolved_active_diagnoses,
        )
        outcome = "pass" if code_set_validation_outcome.passed() else "fail"

        return FinalIcdCodeAdjudicationOutcome(
            outcome=outcome,
            seeded_icd10_codes=sorted(seeded_code_set),
            adjudicated_icd10_codes=adjudicated_codes,
            added_icd10_codes=added_codes,
            removed_seeded_icd10_codes=removed_seeded_codes,
            diagnosis_mentions=diagnosis_mentions,
            code_set_validation_outcome=code_set_validation_outcome,
            adjudication_rationale="; ".join(
                item for item in [extraction_rationale, *resolution_rationales] if item
            ),
            revision_targets=revision_targets,
            unresolved_active_diagnoses=unresolved_active_diagnoses,
            adjudicator_prompt_id=prompt_id,
            adjudicator_prompt_version=prompt_version,
        )


def _add_adjudication_consistency_issues(
    *,
    validation_outcome: IcdCodeSetValidationOutcome,
    seeded_codes: list[str],
    unresolved_active_diagnoses: list[str],
) -> IcdCodeSetValidationOutcome:
    issues = list(validation_outcome.issues)
    seeded_deduplicated_codes = list(dict.fromkeys(_normalize_seeded_codes(seeded_codes)))
    adjudicated_code_set = set(validation_outcome.deduplicated_codes)

    for seeded_code in seeded_deduplicated_codes:
        if seeded_code in adjudicated_code_set:
            continue
        issues.append(
            IcdCodeSetValidationIssue(
                severity="error",
                rule_type="missing_seeded_code",
                source_code=seeded_code,
                message=(
                    f"Seeded ICD-10-CM code {seeded_code} was not recovered from "
                    "the final note adjudication."
                ),
                remediation=_missing_seeded_code_remediation(seeded_code),
            )
        )

    for diagnosis_name in unresolved_active_diagnoses:
        issues.append(
            IcdCodeSetValidationIssue(
                severity="error",
                rule_type="unresolved_active_diagnosis",
                message=f"Active reportable diagnosis could not be resolved: {diagnosis_name}.",
                remediation=(
                    "Revise the note to remove unsupported extra active diagnoses or make "
                    "the diagnosis specific enough for ICD-10-CM resolution."
                ),
            )
        )

    return validation_outcome.model_copy(update={"issues": issues})


def _normalize_seeded_codes(seeded_codes: list[str]) -> list[str]:
    from clinical_note_generation_v3.infrastructure.data_preprocessing.icd_rule_repository import (
        normalize_icd_code,
    )

    return [normalize_icd_code(code) for code in seeded_codes]


def _missing_seeded_code_remediation(seeded_code: str) -> str:
    """
    Return a revision-target string that is appropriate for the code category.

    Z-category personal-history and status codes are historically resolved conditions
    that are still the reason for the current encounter.  Telling the revision model
    to make them "clearly active and reportable" causes the model to drift the note
    toward describing the condition as an ongoing active disease — the opposite of
    what is clinically and coding-rule correct.
    """
    normalized = seeded_code.strip().upper()
    if normalized.startswith("Z"):
        return (
            f"The personal-history or status code {seeded_code} was not recovered "
            "from the adjudication.  Ensure the note clearly documents the condition "
            "as a historical/background diagnosis that is the reason for the current "
            "surveillance or follow-up encounter — do NOT reframe it as an active "
            "current disease.  It should appear in Past Medical History and/or as the "
            "stated reason for the visit (e.g., 'follow-up after curative resection', "
            "'routine surveillance for prior malignancy')."
        )
    return (
        f"Seeded code {seeded_code} was not recovered from the final note adjudication. "
        "Revise the note so the seeded diagnosis is clearly present, currently managed, "
        "and documentationally supported — it must be recoverable as an active reportable "
        "condition for this encounter."
    )


def _build_revision_targets(
    *,
    code_set_validation_outcome: IcdCodeSetValidationOutcome,
    added_codes: list[str],
    removed_seeded_codes: list[str],
    unresolved_active_diagnoses: list[str],
) -> list[str]:
    revision_targets: list[str] = []
    for issue in code_set_validation_outcome.errors():
        revision_targets.append(issue.remediation or issue.message)
    if added_codes:
        revision_targets.append(
            "Review extra diagnoses introduced by the note that were not seeded: "
            + ", ".join(added_codes)
            + ". Remove or clarify any that are not clinically supported by the encounter."
        )
    if removed_seeded_codes:
        for code in removed_seeded_codes:
            revision_targets.append(_missing_seeded_code_remediation(code))
    if unresolved_active_diagnoses:
        revision_targets.append(
            "Remove or clarify unresolved active diagnoses: "
            + ", ".join(unresolved_active_diagnoses)
        )
    return list(dict.fromkeys(revision_targets))
