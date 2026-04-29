"""
Deterministic ICD-10-CM code-set validation.
"""

from __future__ import annotations

import re
from collections import Counter

from clinical_note_generation_v3.core.models.bundle import (
    ResolvedConditionCode,
    SeededClinicalBundle,
)
from clinical_note_generation_v3.core.models.icd_adjudication import (
    IcdCodeSetValidationIssue,
    IcdCodeSetValidationOutcome,
)
from clinical_note_generation_v3.core.services.guideline_rule_pack_engine import (
    run_all_guideline_rules,
)
from clinical_note_generation_v3.infrastructure.data_preprocessing.icd_rule_repository import (
    IcdRuleRepository,
    normalize_icd_code,
)


class IcdCodeSetValidationError(RuntimeError):
    """
    Raised when a deterministic ICD code-set gate fails before note generation.
    """

    def __init__(self, outcome: IcdCodeSetValidationOutcome) -> None:
        self.outcome = outcome
        message = "; ".join(issue.message for issue in outcome.errors())
        super().__init__(message or "ICD-10-CM code-set validation failed.")


class IcdCodeSetValidator:
    """
    Validates final ICD-10-CM code sets against official tabular rules.
    """

    def __init__(
        self,
        *,
        icd_rule_repository: IcdRuleRepository,
        official_icd_repository=None,
    ) -> None:
        self._icd_rule_repository = icd_rule_repository
        self._official_icd_repository = official_icd_repository

    def collapse_duplicate_resolved_conditions(
        self,
        resolved_conditions: list[ResolvedConditionCode],
    ) -> list[ResolvedConditionCode]:
        collapsed_by_code: dict[str, ResolvedConditionCode] = {}
        for resolved_condition in resolved_conditions:
            normalized_code = normalize_icd_code(resolved_condition.icd_code)
            source_names = (
                list(resolved_condition.source_condition_names)
                if resolved_condition.source_condition_names
                else [resolved_condition.condition_name]
            )
            if normalized_code not in collapsed_by_code:
                collapsed_by_code[normalized_code] = resolved_condition.model_copy(
                    update={
                        "icd_code": normalized_code,
                        "source_condition_names": source_names,
                    }
                )
                continue

            existing = collapsed_by_code[normalized_code]
            merged_source_names = list(
                dict.fromkeys([*existing.source_condition_names, *source_names])
            )
            collapsed_by_code[normalized_code] = existing.model_copy(
                update={
                    "condition_name": " + ".join(merged_source_names),
                    "source_condition_names": merged_source_names,
                }
            )
        return list(collapsed_by_code.values())

    def collapse_seeded_bundle_codes(
        self,
        seeded_bundle: SeededClinicalBundle,
    ) -> SeededClinicalBundle:
        return seeded_bundle.model_copy(
            update={
                "resolved_conditions": self.collapse_duplicate_resolved_conditions(
                    seeded_bundle.resolved_conditions
                )
            }
        )

    def validate_codes(self, codes: list[str]) -> IcdCodeSetValidationOutcome:
        original_codes = list(codes)
        normalized_codes = [normalize_icd_code(code) for code in original_codes if code.strip()]
        deduplicated_codes = list(dict.fromkeys(normalized_codes))
        duplicate_codes = [code for code, count in Counter(normalized_codes).items() if count > 1]

        issues: list[IcdCodeSetValidationIssue] = []
        for duplicate_code in duplicate_codes:
            issues.append(
                IcdCodeSetValidationIssue(
                    severity="warning",
                    rule_type="duplicate",
                    source_code=duplicate_code,
                    message=f"Duplicate ICD-10-CM code {duplicate_code} was collapsed.",
                    remediation="Preserve one code and keep all source conditions as provenance.",
                )
            )

        issues.extend(self._validate_code_existence_and_billability(deduplicated_codes))
        issues.extend(self._validate_excludes1_conflicts(deduplicated_codes))
        issues.extend(self._collect_instructional_warnings(deduplicated_codes))
        issues.extend(run_all_guideline_rules(deduplicated_codes))

        return IcdCodeSetValidationOutcome(
            original_codes=original_codes,
            normalized_codes=normalized_codes,
            deduplicated_codes=deduplicated_codes,
            duplicate_codes=duplicate_codes,
            issues=issues,
        )

    def assert_valid_codes(self, codes: list[str]) -> IcdCodeSetValidationOutcome:
        outcome = self.validate_codes(codes)
        if not outcome.passed():
            raise IcdCodeSetValidationError(outcome)
        return outcome

    def _validate_code_existence_and_billability(
        self,
        codes: list[str],
    ) -> list[IcdCodeSetValidationIssue]:
        if self._official_icd_repository is None:
            return []

        issues: list[IcdCodeSetValidationIssue] = []
        for code in codes:
            record = self._official_icd_repository.get_code(code)
            if record is None:
                issues.append(
                    IcdCodeSetValidationIssue(
                        severity="error",
                        rule_type="unknown_code",
                        source_code=code,
                        message=f"ICD-10-CM code {code} does not exist in the official file.",
                        remediation="Re-resolve the diagnosis against official ICD-10-CM candidates.",
                    )
                )
                continue
            if not record.is_billable:
                issues.append(
                    IcdCodeSetValidationIssue(
                        severity="error",
                        rule_type="non_billable_code",
                        source_code=code,
                        message=f"ICD-10-CM code {code} is a non-billable header code.",
                        remediation="Select the most specific billable child code supported by the note.",
                    )
                )
        return issues

    def _validate_excludes1_conflicts(
        self,
        codes: list[str],
    ) -> list[IcdCodeSetValidationIssue]:
        issues: list[IcdCodeSetValidationIssue] = []
        emitted_pairs: set[tuple[str, str, str]] = set()
        for source_code in codes:
            for rule_text in self._icd_rule_repository.applicable_notes(source_code, "excludes1"):
                for related_code in codes:
                    if source_code == related_code:
                        continue
                    if not _rule_text_references_code(rule_text, related_code):
                        continue
                    pair_key = tuple(sorted([source_code, related_code]) + [rule_text])
                    if pair_key in emitted_pairs:
                        continue
                    emitted_pairs.add(pair_key)
                    issues.append(
                        IcdCodeSetValidationIssue(
                            severity="error",
                            rule_type="excludes1_conflict",
                            source_code=source_code,
                            related_code=related_code,
                            message=(
                                f"ICD-10-CM Excludes1 conflict: {source_code} cannot be "
                                f"reported with {related_code}."
                            ),
                            rule_text=rule_text,
                            remediation=(
                                "Revise the note or adjudicated labels so mutually exclusive "
                                "conditions are not both coded."
                            ),
                        )
                    )
        return issues

    def _collect_instructional_warnings(
        self,
        codes: list[str],
    ) -> list[IcdCodeSetValidationIssue]:
        warnings: list[IcdCodeSetValidationIssue] = []
        for code in codes:
            for note_type in ("codeFirst", "useAdditionalCode", "codeAlso"):
                for rule_text in self._icd_rule_repository.applicable_notes(code, note_type):
                    warnings.append(
                        IcdCodeSetValidationIssue(
                            severity="warning",
                            rule_type=note_type,
                            source_code=code,
                            message=f"{code} has ICD-10-CM '{note_type}' instruction.",
                            rule_text=rule_text,
                            remediation="Review sequencing/additional-code requirements.",
                        )
                    )
        return warnings


def _rule_text_references_code(rule_text: str, code: str) -> bool:
    references = _extract_code_references(rule_text)
    normalized_code = normalize_icd_code(code).replace(".", "")
    category = normalized_code[:3]
    for reference in references:
        if reference["kind"] == "range":
            if reference["start"][:3] <= category <= reference["end"][:3]:
                return True
        if reference["kind"] == "prefix" and normalized_code.startswith(reference["prefix"]):
            return True
        if reference["kind"] == "exact":
            exact = reference["code"]
            if normalized_code == exact:
                return True
            if len(exact) == 3 and normalized_code.startswith(exact):
                return True
    return False


def _extract_code_references(rule_text: str) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    consumed_spans: list[tuple[int, int]] = []

    range_pattern = re.compile(
        r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?)\s*-\s*"
        r"([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?)\b",
        re.IGNORECASE,
    )
    for match in range_pattern.finditer(rule_text):
        references.append(
            {
                "kind": "range",
                "start": normalize_icd_code(match.group(1)).replace(".", ""),
                "end": normalize_icd_code(match.group(2)).replace(".", ""),
            }
        )
        consumed_spans.append(match.span())

    prefix_pattern = re.compile(
        r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?)\.-|\b"
        r"([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?)-",
        re.IGNORECASE,
    )
    for match in prefix_pattern.finditer(rule_text):
        if _span_is_consumed(match.span(), consumed_spans):
            continue
        raw_prefix = match.group(1) or match.group(2)
        references.append(
            {
                "kind": "prefix",
                "prefix": normalize_icd_code(raw_prefix).replace(".", ""),
            }
        )
        consumed_spans.append(match.span())

    exact_pattern = re.compile(r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?)\b", re.IGNORECASE)
    for match in exact_pattern.finditer(rule_text):
        if _span_is_consumed(match.span(), consumed_spans):
            continue
        references.append(
            {
                "kind": "exact",
                "code": normalize_icd_code(match.group(1)).replace(".", ""),
            }
        )
    return references


def _span_is_consumed(span: tuple[int, int], consumed_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(
        start >= consumed_start and end <= consumed_end
        for consumed_start, consumed_end in consumed_spans
    )
