"""
ICD-10-CM code-set validation and final adjudication models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DiagnosisMentionStatus = Literal[
    "active",
    "historical",
    "negated",
    "ruled_out",
    "family_history",
    "incidental",
]


class ClinicalDiagnosisMention(BaseModel):
    """
    One diagnosis-like concept extracted from a generated note.
    """

    diagnosis_name: str
    status: DiagnosisMentionStatus
    evidence: str = ""
    should_code: bool = False
    coding_rationale: str = ""


class IcdCodeSetValidationIssue(BaseModel):
    """
    Deterministic code-set validation issue.
    """

    severity: Literal["error", "warning"]
    rule_type: str
    source_code: str | None = None
    related_code: str | None = None
    message: str
    rule_text: str = ""
    remediation: str = ""


class IcdCodeSetValidationOutcome(BaseModel):
    """
    Result of deterministic ICD-10-CM code-set validation.
    """

    original_codes: list[str] = Field(default_factory=list)
    normalized_codes: list[str] = Field(default_factory=list)
    deduplicated_codes: list[str] = Field(default_factory=list)
    duplicate_codes: list[str] = Field(default_factory=list)
    issues: list[IcdCodeSetValidationIssue] = Field(default_factory=list)

    def passed(self) -> bool:
        return not self.errors()

    def errors(self) -> list[IcdCodeSetValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[IcdCodeSetValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


class FinalIcdCodeAdjudicationOutcome(BaseModel):
    """
    Final coder-style adjudication for one generated note.
    """

    outcome: Literal["pass", "fail"]
    seeded_icd10_codes: list[str] = Field(default_factory=list)
    adjudicated_icd10_codes: list[str] = Field(default_factory=list)
    added_icd10_codes: list[str] = Field(default_factory=list)
    removed_seeded_icd10_codes: list[str] = Field(default_factory=list)
    diagnosis_mentions: list[ClinicalDiagnosisMention] = Field(default_factory=list)
    code_set_validation_outcome: IcdCodeSetValidationOutcome
    adjudication_rationale: str = ""
    revision_targets: list[str] = Field(default_factory=list)
    unresolved_active_diagnoses: list[str] = Field(default_factory=list)
    adjudicator_prompt_id: str | None = None
    adjudicator_prompt_version: str | None = None

    def passed(self) -> bool:
        return self.outcome == "pass" and self.code_set_validation_outcome.passed()
