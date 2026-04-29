"""
Deterministic ICD-10-CM coding guideline rule packs.

These rules implement official ICD-10-CM coding guidelines that are not always
expressed as direct Excludes1/Excludes2 pairs in the tabular XML, but are
mandated by the ICD-10-CM Official Guidelines for Coding and Reporting.

Each rule function receives a list of undotted, uppercased ICD-10-CM codes
(e.g. ["J209", "J069"]) and returns zero or more IcdCodeSetValidationIssue
objects describing any detected guideline conflicts.
"""

from __future__ import annotations

from clinical_note_generation_v3.core.models.icd_adjudication import IcdCodeSetValidationIssue
from clinical_note_generation_v3.infrastructure.data_preprocessing.icd_rule_repository import (
    normalize_icd_code,
)


def run_all_guideline_rules(codes: list[str]) -> list[IcdCodeSetValidationIssue]:
    """
    Run all registered coding-guideline rule packs against a deduplicated code set.

    Accepts both dotted (J20.9) and undotted (J209) codes; internally normalizes
    to undotted uppercase for prefix matching.  Returns IcdCodeSetValidationIssue
    entries for any detected conflicts.
    """
    undotted = [normalize_icd_code(c).replace(".", "") for c in codes if c.strip()]
    issues: list[IcdCodeSetValidationIssue] = []
    for rule_fn in _GUIDELINE_RULES:
        issues.extend(rule_fn(undotted))
    return issues


# ---------------------------------------------------------------------------
# Respiratory — bronchitis subsumes nonspecific URI
# ---------------------------------------------------------------------------

def _bronchitis_subsumes_nonspecific_uri(
    codes: list[str],
) -> list[IcdCodeSetValidationIssue]:
    """
    ICD-10-CM Guideline (Section I.C.10): When acute bronchitis (J20.-) is
    documented, do not additionally code acute upper respiratory infection of
    multiple/unspecified sites (J06.-).  Bronchitis is the more specific
    diagnosis and subsumes the nonspecific URI presentation.

    This rule catches the J20.9 + J06.9 conflict that is not always expressed
    as a direct Excludes1 pair in the official tabular XML.
    """
    bronchitis = [c for c in codes if c.startswith("J20")]
    uri_nonspecific = [c for c in codes if c.startswith("J06")]
    if not (bronchitis and uri_nonspecific):
        return []

    issues: list[IcdCodeSetValidationIssue] = []
    for uri_code in uri_nonspecific:
        issues.append(
            IcdCodeSetValidationIssue(
                severity="error",
                rule_type="guideline_specificity_conflict",
                source_code=normalize_icd_code(uri_code),
                related_code=normalize_icd_code(bronchitis[0]),
                message=(
                    f"Coding guideline conflict: {normalize_icd_code(uri_code)} "
                    f"(acute URI, unspecified) cannot be coded alongside "
                    f"{normalize_icd_code(bronchitis[0])} (acute bronchitis). "
                    "When bronchitis is documented, code J20.- only; "
                    "J06.- is nonspecific and subsumed by the more specific diagnosis."
                ),
                rule_text=(
                    "ICD-10-CM Official Guidelines I.C.10: Report the most specific "
                    "respiratory infection diagnosis. J06.- (acute upper respiratory "
                    "infections, multiple/unspecified sites) must not be reported "
                    "alongside the more specific bronchitis code J20.-."
                ),
                remediation=(
                    "Remove J06.- from the code set. Code the acute bronchitis "
                    "diagnosis J20.- only."
                ),
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Asthma — same-severity conflicting encounter classification
# ---------------------------------------------------------------------------

_ASTHMA_SEVERITY_PREFIXES: frozenset[str] = frozenset({"J452", "J453", "J454", "J455"})


def _asthma_same_severity_conflicting_encounter(
    codes: list[str],
) -> list[IcdCodeSetValidationIssue]:
    """
    ICD-10-CM combination-code guideline: J45.x0 (uncomplicated), J45.x1
    (with acute exacerbation), and J45.x2 (with status asthmaticus) are
    mutually exclusive for the same asthma severity level.  A patient cannot
    simultaneously be uncomplicated and in exacerbation for the same severity.
    """
    issues: list[IcdCodeSetValidationIssue] = []
    emitted: set[tuple[str, str]] = set()
    for code_a in codes:
        if len(code_a) < 4 or code_a[:4] not in _ASTHMA_SEVERITY_PREFIXES:
            continue
        for code_b in codes:
            if code_a == code_b or len(code_b) < 4:
                continue
            if code_a[:4] != code_b[:4]:
                continue
            pair = tuple(sorted([code_a, code_b]))
            if pair in emitted:
                continue
            emitted.add(pair)
            issues.append(
                IcdCodeSetValidationIssue(
                    severity="error",
                    rule_type="guideline_combination_code_conflict",
                    source_code=normalize_icd_code(code_a),
                    related_code=normalize_icd_code(code_b),
                    message=(
                        f"Asthma coding conflict: {normalize_icd_code(code_a)} and "
                        f"{normalize_icd_code(code_b)} describe the same asthma severity "
                        "with conflicting encounter classifications (e.g., uncomplicated "
                        "vs. with exacerbation). Use only one code per severity per encounter."
                    ),
                    rule_text=(
                        "ICD-10-CM: Asthma combination codes J45.x0/x1/x2 are mutually "
                        "exclusive for the same severity level (mild-intermittent J452, "
                        "mild-persistent J453, moderate J454, severe J455)."
                    ),
                    remediation=(
                        "Select one code that best represents the encounter: "
                        "uncomplicated (x0), with exacerbation (x1), "
                        "or with status asthmaticus (x2)."
                    ),
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Endocrine — diabetes mellitus type conflict
# ---------------------------------------------------------------------------

_DIABETES_TYPE_PREFIXES: tuple[str, ...] = ("E10", "E11", "E13")
_DIABETES_TYPE_LABELS: dict[str, str] = {
    "E10": "Type 1 diabetes mellitus",
    "E11": "Type 2 diabetes mellitus",
    "E13": "Other specified diabetes mellitus",
}


def _diabetes_type_conflict(codes: list[str]) -> list[IcdCodeSetValidationIssue]:
    """
    ICD-10-CM Guideline (Section I.C.4.a): A patient is classified under one
    type of diabetes mellitus per encounter.  Coding both E10 (Type 1) and E11
    (Type 2), or combining either with E13 (Other specified), implies
    contradictory classification and is a coding error in most scenarios.
    """
    present: list[str] = [
        prefix
        for prefix in _DIABETES_TYPE_PREFIXES
        if any(c.startswith(prefix) for c in codes)
    ]
    if len(present) < 2:
        return []

    representative: dict[str, str] = {
        prefix: normalize_icd_code(next(c for c in codes if c.startswith(prefix)))
        for prefix in present
    }
    issues: list[IcdCodeSetValidationIssue] = []
    for i, type_a in enumerate(present):
        for type_b in present[i + 1:]:
            issues.append(
                IcdCodeSetValidationIssue(
                    severity="error",
                    rule_type="guideline_diabetes_type_conflict",
                    source_code=representative[type_a],
                    related_code=representative[type_b],
                    message=(
                        f"Diabetes type conflict: {representative[type_a]} "
                        f"({_DIABETES_TYPE_LABELS[type_a]}) and {representative[type_b]} "
                        f"({_DIABETES_TYPE_LABELS[type_b]}) classify different DM types. "
                        "A patient should be assigned one DM type per encounter."
                    ),
                    rule_text=(
                        "ICD-10-CM Official Guidelines I.C.4.a: Assign the appropriate "
                        "diabetes mellitus code category. Combining E10 and E11, or either "
                        "with E13, is a coding error absent extraordinary clinical justification."
                    ),
                    remediation=(
                        "Verify the diabetes type documented in the note and code only one "
                        "DM category per encounter. Add Z79.4 if the patient uses insulin."
                    ),
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Neoplasm — active malignancy vs personal history conflict
# ---------------------------------------------------------------------------

def _neoplasm_active_vs_personal_history_conflict(
    codes: list[str],
) -> list[IcdCodeSetValidationIssue]:
    """
    ICD-10-CM Guideline (Section I.C.2): Active malignancy codes (C00–C96) and
    personal-history-of-malignancy codes (Z85.-) should not appear together for
    the same primary site.  Personal history implies the malignancy was treated
    and is no longer present; active C codes imply it is still present.

    Emitted as a warning (not error) because without a site-level lookup we
    cannot always confirm the codes describe the same site.
    """
    active_cancers = [c for c in codes if len(c) >= 3 and c[:1] == "C" and c[1:3].isdigit()]
    history_codes = [c for c in codes if c.startswith("Z85")]
    if not (active_cancers and history_codes):
        return []

    active_display = ", ".join(normalize_icd_code(c) for c in active_cancers)
    history_display = ", ".join(normalize_icd_code(c) for c in history_codes)
    return [
        IcdCodeSetValidationIssue(
            severity="warning",
            rule_type="guideline_neoplasm_active_vs_history",
            source_code=normalize_icd_code(active_cancers[0]),
            related_code=normalize_icd_code(history_codes[0]),
            message=(
                f"Possible neoplasm status conflict: active malignancy ({active_display}) "
                f"is present alongside personal history of malignancy ({history_display}). "
                "If both describe the same primary site, only one classification applies."
            ),
            rule_text=(
                "ICD-10-CM Official Guidelines I.C.2: Do not code personal history of "
                "malignancy (Z85.-) when active malignancy is still present. Use active "
                "malignancy C codes for current disease; Z85.- only when no active disease "
                "remains after treatment."
            ),
            remediation=(
                "If the cancer is currently active, remove Z85.-. "
                "If in remission or resolved, use the appropriate remission/surveillance code "
                "rather than the active malignancy C code."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Registry — order matters: more critical rules first
# ---------------------------------------------------------------------------

_GUIDELINE_RULES = [
    _bronchitis_subsumes_nonspecific_uri,
    _asthma_same_severity_conflicting_encounter,
    _diabetes_type_conflict,
    _neoplasm_active_vs_personal_history_conflict,
]
