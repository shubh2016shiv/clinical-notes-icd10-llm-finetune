"""
ICD-10-CM semantic constraint extractor.

Converts one resolved ICD code (with its official long and short description)
into a structured IcdCodeSemanticConstraints object that the note generator
and evaluator both consume.

Extraction uses three tiers in order:

  Tier 1 — Deterministic regex
      High-confidence, fast extraction of laterality, encounter type, temporal
      state, severity, with/without complication flags, and the unspecified flag.
      These patterns cover the vast majority of well-formed ICD descriptions.

  Tier 2 — Chapter/family-specific rule pack
      Domain knowledge per chapter family (injury, endocrine/metabolic,
      behavioral health, respiratory/infectious, neoplasm, musculoskeletal,
      cardiovascular).  Each rule pack derives must_include and must_not_imply
      lists from the signals already detected in Tier 1.

  Tier 3 — LLM escalation (optional, injected)
      Called only when tiers 1+2 produce zero must_include items — meaning the
      description is unusual or compound and deterministic rules cannot resolve
      it.  Accepts a JSONGenerationClient injected at construction time.
      If no LLM client is provided, Tier 3 is silently skipped and a best-effort
      result is returned from tiers 1+2 alone.

Public classes
--------------
  IcdCodeSemanticConstraintExtractor
"""

from __future__ import annotations

import logging
import re

from clinical_note_generation_v3.core.models.bundle import ResolvedConditionCode
from clinical_note_generation_v3.core.models.constraints import (
    ConstraintExtractionSource,
    ExtractedSemanticSignal,
    IcdCodeSemanticConstraints,
)
from clinical_note_generation_v3.core.ports.llm_generation_port import JSONGenerationClient
from clinical_note_generation_v3.prompt_specs.registry import (
    build_constraint_extraction_prompt_spec,
)
from clinical_note_generation_v3.prompt_specs.rendering import compose_chat_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 1: Deterministic regex patterns
# ---------------------------------------------------------------------------

_LATERALITY_PATTERN = re.compile(r"\b(right|left|bilateral)\b", re.IGNORECASE)

_ENCOUNTER_TYPE_KEYWORDS: dict[str, str] = {
    "initial encounter": "initial",
    "subsequent encounter": "subsequent",
    "sequela": "sequela",
}

_TEMPORAL_STATE_PATTERNS: dict[str, re.Pattern] = {
    "acute": re.compile(r"\bacute\b", re.IGNORECASE),
    "chronic": re.compile(r"\bchronic\b", re.IGNORECASE),
    "recurrent": re.compile(r"\brecurrent\b", re.IGNORECASE),
    "in_remission": re.compile(r"\bin\s+(partial\s+|full\s+)?remission\b", re.IGNORECASE),
    "history": re.compile(r"\b(personal history|history of)\b", re.IGNORECASE),
}

_SEVERITY_PATTERNS: dict[str, re.Pattern] = {
    "mild": re.compile(r"\bmild\b", re.IGNORECASE),
    "moderate": re.compile(r"\bmoderate\b", re.IGNORECASE),
    "severe": re.compile(r"\bsevere\b", re.IGNORECASE),
    "profound": re.compile(r"\bprofound\b", re.IGNORECASE),
    "uncontrolled": re.compile(r"\buncontrolled\b", re.IGNORECASE),
}

# Captures everything after "with" up to punctuation or end of string.
# E.g., "with diabetic chronic kidney disease" → "diabetic chronic kidney disease"
_WITH_COMPLICATION_PATTERN = re.compile(r"\bwith\s+([^,;.()\n]+)", re.IGNORECASE)
# E.g., "without complications" → "complications"
_WITHOUT_COMPLICATION_PATTERN = re.compile(r"\bwithout\s+([^,;.()\n]+)", re.IGNORECASE)

_UNSPECIFIED_PATTERN = re.compile(r"\bunspecified\b", re.IGNORECASE)


def _run_deterministic_regex_extraction(description_text: str) -> list[ExtractedSemanticSignal]:
    """
    Runs all Tier-1 regex patterns over the combined ICD description text.

    Returns one ExtractedSemanticSignal per match, preserving the matched text
    span for traceability.  Confidence is set to 0.97 for all deterministic hits.
    """
    signals: list[ExtractedSemanticSignal] = []

    # Laterality — take only the first match (ICD descriptions name laterality once)
    laterality_match = _LATERALITY_PATTERN.search(description_text)
    if laterality_match:
        signals.append(
            ExtractedSemanticSignal(
                signal_type="laterality",
                extracted_value=laterality_match.group(1).lower(),
                confidence=0.97,
                extraction_source=ConstraintExtractionSource.DETERMINISTIC_REGEX,
                matched_text_span=laterality_match.group(0),
            )
        )

    # Encounter type — scan for known keyword phrases
    lowered = description_text.lower()
    for keyword, encounter_type_value in _ENCOUNTER_TYPE_KEYWORDS.items():
        if keyword in lowered:
            signals.append(
                ExtractedSemanticSignal(
                    signal_type="encounter_type",
                    extracted_value=encounter_type_value,
                    confidence=0.97,
                    extraction_source=ConstraintExtractionSource.DETERMINISTIC_REGEX,
                    matched_text_span=keyword,
                )
            )
            break  # descriptions have at most one encounter type keyword

    # Temporal states
    for state_name, pattern in _TEMPORAL_STATE_PATTERNS.items():
        match = pattern.search(description_text)
        if match:
            signals.append(
                ExtractedSemanticSignal(
                    signal_type="temporal_state",
                    extracted_value=state_name,
                    confidence=0.97,
                    extraction_source=ConstraintExtractionSource.DETERMINISTIC_REGEX,
                    matched_text_span=match.group(0),
                )
            )

    # Severity
    for severity_level, pattern in _SEVERITY_PATTERNS.items():
        match = pattern.search(description_text)
        if match:
            signals.append(
                ExtractedSemanticSignal(
                    signal_type="severity",
                    extracted_value=severity_level,
                    confidence=0.97,
                    extraction_source=ConstraintExtractionSource.DETERMINISTIC_REGEX,
                    matched_text_span=match.group(0),
                )
            )

    # With-complication flags
    for match in _WITH_COMPLICATION_PATTERN.finditer(description_text):
        captured_complication = match.group(1).strip().rstrip(" ,;")
        if captured_complication:
            signals.append(
                ExtractedSemanticSignal(
                    signal_type="with_complication",
                    extracted_value=captured_complication.lower(),
                    confidence=0.95,
                    extraction_source=ConstraintExtractionSource.DETERMINISTIC_REGEX,
                    matched_text_span=match.group(0),
                )
            )

    # Without-complication flags
    for match in _WITHOUT_COMPLICATION_PATTERN.finditer(description_text):
        captured_complication = match.group(1).strip().rstrip(" ,;")
        if captured_complication:
            signals.append(
                ExtractedSemanticSignal(
                    signal_type="without_complication",
                    extracted_value=captured_complication.lower(),
                    confidence=0.95,
                    extraction_source=ConstraintExtractionSource.DETERMINISTIC_REGEX,
                    matched_text_span=match.group(0),
                )
            )

    # Unspecified flag
    if _UNSPECIFIED_PATTERN.search(description_text):
        signals.append(
            ExtractedSemanticSignal(
                signal_type="is_unspecified",
                extracted_value="true",
                confidence=0.97,
                extraction_source=ConstraintExtractionSource.DETERMINISTIC_REGEX,
                matched_text_span="unspecified",
            )
        )

    return signals


# ---------------------------------------------------------------------------
# Chapter family classification
# ---------------------------------------------------------------------------


def _classify_chapter_family_from_icd_code(icd_code: str) -> str:
    """
    Maps the leading character(s) of an ICD-10-CM code to a chapter family name.

    The chapter family determines which Tier-2 rule pack applies to derive
    must_include and must_not_imply items.
    """
    code_upper = icd_code.upper().lstrip()
    if not code_upper:
        return "other"

    first_letter = code_upper[0]

    chapter_map: dict[str, str] = {
        "A": "respiratory_infectious",
        "B": "respiratory_infectious",
        "C": "neoplasm",
        "E": "endocrine_metabolic",
        "F": "behavioral_health",
        "G": "neurological",
        "I": "cardiovascular",
        "J": "respiratory_infectious",
        "K": "digestive",
        "M": "musculoskeletal",
        "N": "genitourinary",
        "O": "obstetric",
        "S": "injury",
        "T": "injury",
    }

    # D codes: D0-D4 are neoplasms, D5+ are blood disorders
    if first_letter == "D":
        try:
            second_digit = (
                int(code_upper[1]) if len(code_upper) > 1 and code_upper[1].isdigit() else 5
            )
            return "neoplasm" if second_digit <= 4 else "blood_disorder"
        except (IndexError, ValueError):
            return "blood_disorder"

    return chapter_map.get(first_letter, "other")


# ---------------------------------------------------------------------------
# Helper: read signals collected by Tier 1
# ---------------------------------------------------------------------------


def _first_signal_value_for_type(
    signals: list[ExtractedSemanticSignal], signal_type: str
) -> str | None:
    for signal in signals:
        if signal.signal_type == signal_type:
            return signal.extracted_value
    return None


def _all_signal_values_for_type(
    signals: list[ExtractedSemanticSignal], signal_type: str
) -> list[str]:
    return [s.extracted_value for s in signals if s.signal_type == signal_type]


def _opposite_laterality_side(laterality: str) -> str | None:
    opposites = {"left": "right", "right": "left"}
    return opposites.get(laterality)


# ---------------------------------------------------------------------------
# Tier 2: Chapter/family-specific rule packs
# ---------------------------------------------------------------------------


def _derive_must_include_and_must_not_imply_for_injury_family(
    laterality: str | None,
    encounter_type: str | None,
    with_complication_flags: list[str],
    without_complication_flags: list[str],
) -> tuple[list[str], list[str]]:
    must_include: list[str] = [
        "mechanism of injury or how the injury occurred",
        "injury site documentation with relevant physical findings",
    ]
    must_not_imply: list[str] = []

    if encounter_type == "initial":
        must_include.extend(
            [
                "acute injury presentation consistent with an initial visit",
                "initial treatment plan or disposition from this encounter",
            ]
        )
        must_not_imply.extend(
            [
                "follow-up healing progress context suggesting a return visit",
                "residual sequela unless a separate sequela code is seeded",
            ]
        )
    elif encounter_type == "subsequent":
        must_include.extend(
            [
                "healing progress assessment since the initial encounter",
                "ongoing treatment or rehabilitation status",
            ]
        )
        must_not_imply.extend(
            [
                "fresh acute initial presentation of the same injury",
            ]
        )
    elif encounter_type == "sequela":
        must_include.extend(
            [
                "residual symptoms or functional limitations from the prior injury",
                "documentation that the original injury has structurally resolved",
            ]
        )
        must_not_imply.extend(
            [
                "active fracture healing or ongoing acute injury progression",
                "initial acute injury presentation",
            ]
        )

    if laterality:
        must_include.append(
            f"{laterality}-sided physical examination findings consistent with the injury"
        )
        opposite = _opposite_laterality_side(laterality)
        if opposite:
            must_not_imply.append(f"{opposite} side as the primarily injured or symptomatic side")

    if with_complication_flags:
        for complication in with_complication_flags:
            must_include.append(f"clinical evidence for {complication}")

    if without_complication_flags:
        for complication in without_complication_flags:
            must_not_imply.append(
                f"presence of {complication} (the coded condition explicitly excludes this)"
            )

    return must_include, must_not_imply


def _derive_must_include_and_must_not_imply_for_endocrine_metabolic_family(
    temporal_states: list[str],
    severity_qualifiers: list[str],
    with_complication_flags: list[str],
    without_complication_flags: list[str],
    is_unspecified: bool,
) -> tuple[list[str], list[str]]:
    must_include: list[str] = [
        "relevant lab results or monitoring values for the metabolic condition",
        "current medication or management regimen",
    ]
    must_not_imply: list[str] = []

    if "acute" in temporal_states:
        must_include.append("acute metabolic derangement presentation with clinical urgency")
    elif "chronic" in temporal_states or not temporal_states:
        must_include.append("chronic disease management context appropriate for a follow-up visit")

    if "in_remission" in temporal_states:
        must_include.append("documentation of current remission status")
        must_not_imply.append("active symptomatic disease episode suggesting relapse")

    if severity_qualifiers:
        must_include.append(
            f"symptom or finding severity consistent with {' / '.join(severity_qualifiers)} degree"
        )

    for complication in with_complication_flags:
        must_include.append(f"clinical evidence for {complication}")

    for complication in without_complication_flags:
        must_not_imply.append(
            f"presence of {complication} (the coded condition explicitly excludes this)"
        )

    if is_unspecified:
        must_not_imply.append(
            "enough detail to specify a subtype that would require a more specific ICD code"
        )

    return must_include, must_not_imply


def _derive_must_include_and_must_not_imply_for_behavioral_health_family(
    temporal_states: list[str],
    severity_qualifiers: list[str],
    is_unspecified: bool,
) -> tuple[list[str], list[str]]:
    must_include: list[str] = [
        "presenting psychiatric symptoms consistent with the diagnosis",
        "mental status examination findings or clinical observations",
    ]
    must_not_imply: list[str] = [
        "somatic or physical chief complaint as the primary reason for visit"
    ]

    if severity_qualifiers:
        must_include.append(
            f"symptom severity presentation consistent with {' / '.join(severity_qualifiers)} level"
        )

    if "recurrent" in temporal_states:
        must_include.extend(
            [
                "documentation of prior episode history establishing the recurrent pattern",
                "current episode context clearly distinct from historical episodes",
            ]
        )

    if "in_remission" in temporal_states:
        must_include.append("current remission status with supporting clinical evidence")
        must_not_imply.append("active acute psychiatric symptom episode")

    if "chronic" in temporal_states:
        must_include.append(
            "ongoing psychiatric management context appropriate for a follow-up visit"
        )

    if is_unspecified:
        must_not_imply.append(
            "a specific psychiatric subtype that would require a more specific ICD code"
        )

    return must_include, must_not_imply


def _derive_must_include_and_must_not_imply_for_respiratory_infectious_family(
    temporal_states: list[str],
    severity_qualifiers: list[str],
    with_complication_flags: list[str],
    without_complication_flags: list[str],
) -> tuple[list[str], list[str]]:
    must_include: list[str] = ["respiratory or infectious symptom description"]
    must_not_imply: list[str] = []

    if "acute" in temporal_states:
        must_include.append("acute onset respiratory or infectious symptom presentation")
    elif "chronic" in temporal_states:
        must_include.extend(
            [
                "chronic respiratory disease management context",
                "baseline status and interval change since last visit",
            ]
        )

    if severity_qualifiers:
        must_include.append(
            f"severity of respiratory or infectious symptoms consistent with {' / '.join(severity_qualifiers)} level"
        )

    for complication in with_complication_flags:
        must_include.append(f"clinical evidence for {complication}")

    for complication in without_complication_flags:
        must_not_imply.append(
            f"presence of {complication} (the coded condition explicitly excludes this)"
        )

    return must_include, must_not_imply


def _derive_must_include_and_must_not_imply_for_neoplasm_family(
    temporal_states: list[str],
    with_complication_flags: list[str],
    without_complication_flags: list[str],
) -> tuple[list[str], list[str]]:
    must_include: list[str] = [
        "oncologic diagnosis documentation with sufficient clinical specificity"
    ]
    must_not_imply: list[str] = []

    if "in_remission" in temporal_states:
        must_include.extend(
            [
                "documented remission status with supporting evidence",
                "surveillance visit context",
            ]
        )
        must_not_imply.extend(
            [
                "active chemotherapy, radiation, or immunotherapy unless explicitly seeded",
                "progressive disease or active tumor burden",
            ]
        )
    elif "history" in temporal_states:
        must_include.append(
            "past oncologic history clearly framed as historical (not active disease)"
        )
        must_not_imply.append("ongoing active malignancy requiring treatment")
    else:
        must_include.extend(
            [
                "current active malignancy documentation",
                "current treatment modality or management plan",
            ]
        )
        must_not_imply.append("full remission or cancer-free status unless explicitly seeded")

    for complication in with_complication_flags:
        must_include.append(f"clinical evidence for {complication}")

    for complication in without_complication_flags:
        must_not_imply.append(
            f"presence of {complication} (the coded condition explicitly excludes this)"
        )

    return must_include, must_not_imply


def _derive_must_include_and_must_not_imply_for_musculoskeletal_family(
    laterality: str | None,
    temporal_states: list[str],
    severity_qualifiers: list[str],
    with_complication_flags: list[str],
    without_complication_flags: list[str],
) -> tuple[list[str], list[str]]:
    must_include: list[str] = ["musculoskeletal symptom description with functional impact"]
    must_not_imply: list[str] = []

    if laterality:
        must_include.append(f"{laterality}-sided musculoskeletal examination findings")
        opposite = _opposite_laterality_side(laterality)
        if opposite:
            must_not_imply.append(f"{opposite} side as the primarily symptomatic or affected side")

    if "acute" in temporal_states:
        must_include.append("acute musculoskeletal injury or flare presentation")
    elif "chronic" in temporal_states:
        must_include.append("chronic musculoskeletal disease management or follow-up context")

    if severity_qualifiers:
        must_include.append(
            f"functional or pain severity consistent with {' / '.join(severity_qualifiers)} level"
        )

    for complication in with_complication_flags:
        must_include.append(f"clinical evidence for {complication}")

    for complication in without_complication_flags:
        must_not_imply.append(
            f"presence of {complication} (the coded condition explicitly excludes this)"
        )

    return must_include, must_not_imply


def _derive_must_include_and_must_not_imply_for_cardiovascular_family(
    temporal_states: list[str],
    severity_qualifiers: list[str],
    with_complication_flags: list[str],
    without_complication_flags: list[str],
) -> tuple[list[str], list[str]]:
    must_include: list[str] = ["cardiovascular symptom or status documentation"]
    must_not_imply: list[str] = []

    if "acute" in temporal_states:
        must_include.append("acute cardiovascular event or presentation")
    elif "chronic" in temporal_states:
        must_include.extend(
            [
                "chronic cardiovascular disease management context",
                "current medication regimen and monitoring status",
            ]
        )

    if severity_qualifiers:
        must_include.append(
            f"cardiovascular status consistent with {' / '.join(severity_qualifiers)} level"
        )

    for complication in with_complication_flags:
        must_include.append(f"clinical evidence for {complication}")

    for complication in without_complication_flags:
        must_not_imply.append(
            f"presence of {complication} (the coded condition explicitly excludes this)"
        )

    return must_include, must_not_imply


def _derive_must_include_and_must_not_imply_for_other_family(
    laterality: str | None,
    with_complication_flags: list[str],
    without_complication_flags: list[str],
    is_unspecified: bool,
) -> tuple[list[str], list[str]]:
    must_include: list[str] = ["relevant clinical presentation consistent with the coded condition"]
    must_not_imply: list[str] = []

    if laterality:
        must_include.append(f"{laterality}-sided clinical findings")
        opposite = _opposite_laterality_side(laterality)
        if opposite:
            must_not_imply.append(f"{opposite} side as the primarily affected side")

    for complication in with_complication_flags:
        must_include.append(f"clinical evidence for {complication}")

    for complication in without_complication_flags:
        must_not_imply.append(
            f"presence of {complication} (the coded condition explicitly excludes this)"
        )

    if is_unspecified:
        must_not_imply.append("enough specificity to justify a more precise ICD-10-CM subtype code")

    return must_include, must_not_imply


def _derive_constraint_items_using_family_rule_pack(
    chapter_family: str,
    laterality: str | None,
    encounter_type: str | None,
    temporal_states: list[str],
    severity_qualifiers: list[str],
    with_complication_flags: list[str],
    without_complication_flags: list[str],
    is_unspecified: bool,
) -> tuple[list[str], list[str]]:
    """
    Dispatches to the appropriate family rule pack and returns
    (must_include_items, must_not_imply_items).
    """
    if chapter_family == "injury":
        return _derive_must_include_and_must_not_imply_for_injury_family(
            laterality=laterality,
            encounter_type=encounter_type,
            with_complication_flags=with_complication_flags,
            without_complication_flags=without_complication_flags,
        )

    if chapter_family == "endocrine_metabolic":
        return _derive_must_include_and_must_not_imply_for_endocrine_metabolic_family(
            temporal_states=temporal_states,
            severity_qualifiers=severity_qualifiers,
            with_complication_flags=with_complication_flags,
            without_complication_flags=without_complication_flags,
            is_unspecified=is_unspecified,
        )

    if chapter_family == "behavioral_health":
        return _derive_must_include_and_must_not_imply_for_behavioral_health_family(
            temporal_states=temporal_states,
            severity_qualifiers=severity_qualifiers,
            is_unspecified=is_unspecified,
        )

    if chapter_family == "respiratory_infectious":
        return _derive_must_include_and_must_not_imply_for_respiratory_infectious_family(
            temporal_states=temporal_states,
            severity_qualifiers=severity_qualifiers,
            with_complication_flags=with_complication_flags,
            without_complication_flags=without_complication_flags,
        )

    if chapter_family == "neoplasm":
        return _derive_must_include_and_must_not_imply_for_neoplasm_family(
            temporal_states=temporal_states,
            with_complication_flags=with_complication_flags,
            without_complication_flags=without_complication_flags,
        )

    if chapter_family == "musculoskeletal":
        return _derive_must_include_and_must_not_imply_for_musculoskeletal_family(
            laterality=laterality,
            temporal_states=temporal_states,
            severity_qualifiers=severity_qualifiers,
            with_complication_flags=with_complication_flags,
            without_complication_flags=without_complication_flags,
        )

    if chapter_family == "cardiovascular":
        return _derive_must_include_and_must_not_imply_for_cardiovascular_family(
            temporal_states=temporal_states,
            severity_qualifiers=severity_qualifiers,
            with_complication_flags=with_complication_flags,
            without_complication_flags=without_complication_flags,
        )

    # All other families (digestive, genitourinary, neurological, obstetric, etc.)
    return _derive_must_include_and_must_not_imply_for_other_family(
        laterality=laterality,
        with_complication_flags=with_complication_flags,
        without_complication_flags=without_complication_flags,
        is_unspecified=is_unspecified,
    )


# ---------------------------------------------------------------------------
# Tier 3: LLM escalation prompt builder
# ---------------------------------------------------------------------------


def _build_llm_escalation_prompt_for_constraint_extraction(
    *,
    icd_code: str,
    long_description: str,
    short_description: str,
    signals_already_extracted: list[ExtractedSemanticSignal],
) -> str:
    if signals_already_extracted:
        signal_summary = "\n".join(
            f"  - {s.signal_type}: {s.extracted_value}" for s in signals_already_extracted
        )
    else:
        signal_summary = "  (none)"

    return f"""You are a clinical documentation specialist and expert ICD-10-CM coder.

Your task: Given an ICD-10-CM code and its official description, derive concrete \
note-writing obligations and prohibitions for a clinical note writer who must write \
a realistic clinical note supporting this diagnosis.

ICD-10-CM Code: {icd_code}
Long description: {long_description}
Short description: {short_description}

Signals already extracted by deterministic patterns (do NOT repeat these):
{signal_summary}

Return ONLY a JSON object in exactly this format — no markdown fences, no extra text:

{{
  "must_include_in_note": [
    "<concrete clinical element the note must contain>"
  ],
  "must_not_imply_in_note": [
    "<thing the note must not state or imply>"
  ]
}}

Rules:
1. must_include_in_note: 2 to 5 specific, concrete clinical documentation elements.
   Good: "left-sided ankle tenderness and swelling on examination"
   Bad: "document the patient's condition" (too vague)
2. must_not_imply_in_note: 1 to 3 concrete prohibitions that would contradict the diagnosis.
3. Do NOT repeat signals already listed above.
4. Return only the JSON object — no markdown fences, no preamble."""


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------


class IcdCodeSemanticConstraintExtractor:
    """
    Converts one resolved ICD code into note-writing constraints using a
    three-tier extraction strategy.

    Parameters
    ----------
    llm_escalation_client
        Optional LLM client (implements JSONGenerationClient protocol) used in
        Tier 3 escalation.  When None, Tier 3 is silently skipped and the
        extractor returns whatever tiers 1+2 produced.  Inject a real client
        for production; pass None for fast offline or test scenarios.
    """

    def __init__(
        self,
        llm_escalation_client: JSONGenerationClient | None = None,
    ) -> None:
        self._llm_escalation_client = llm_escalation_client

    def extract_semantic_constraints_for_resolved_condition(
        self,
        resolved_condition: ResolvedConditionCode,
    ) -> IcdCodeSemanticConstraints:
        """
        Run all three extraction tiers and return a fully populated
        IcdCodeNoteWritingConstraints for the given resolved condition.

        Parameters
        ----------
        resolved_condition
            A single resolved ICD code entry from a SeededClinicalBundle.

        Returns
        -------
        IcdCodeSemanticConstraints
            Constraint set ready to be passed to the note generator and evaluator.
        """
        combined_icd_description_text = (
            f"{resolved_condition.icd_long_description} "
            f"{resolved_condition.icd_short_description}"
        )

        # --- Tier 1: deterministic regex ---
        deterministic_regex_signals = _run_deterministic_regex_extraction(
            combined_icd_description_text
        )
        all_extracted_signals: list[ExtractedSemanticSignal] = list(deterministic_regex_signals)

        # --- Classify chapter family ---
        chapter_family = _classify_chapter_family_from_icd_code(resolved_condition.icd_code)

        # --- Assemble signal summary for Tier 2 dispatch ---
        laterality = _first_signal_value_for_type(all_extracted_signals, "laterality")
        encounter_type = _first_signal_value_for_type(all_extracted_signals, "encounter_type")
        temporal_states = _all_signal_values_for_type(all_extracted_signals, "temporal_state")
        severity_qualifiers = _all_signal_values_for_type(all_extracted_signals, "severity")
        with_complication_flags = _all_signal_values_for_type(
            all_extracted_signals, "with_complication"
        )
        without_complication_flags = _all_signal_values_for_type(
            all_extracted_signals, "without_complication"
        )
        is_unspecified = any(
            signal.signal_type == "is_unspecified" for signal in all_extracted_signals
        )

        # --- Tier 2: family-specific rule pack ---
        must_include_items, must_not_imply_items = _derive_constraint_items_using_family_rule_pack(
            chapter_family=chapter_family,
            laterality=laterality,
            encounter_type=encounter_type,
            temporal_states=temporal_states,
            severity_qualifiers=severity_qualifiers,
            with_complication_flags=with_complication_flags,
            without_complication_flags=without_complication_flags,
            is_unspecified=is_unspecified,
        )

        # --- Tier 3: LLM escalation if tiers 1+2 produced no must_include items ---
        constraint_extractor_prompt_id: str | None = None
        constraint_extractor_prompt_version: str | None = None
        if not must_include_items and self._llm_escalation_client is not None:
            logger.info(
                "Escalating to LLM for constraint extraction: code=%s description='%s'",
                resolved_condition.icd_code,
                resolved_condition.icd_short_description[:60],
            )
            (
                llm_derived_signals,
                constraint_extractor_prompt_id,
                constraint_extractor_prompt_version,
            ) = self._run_llm_escalation_and_return_signals(
                resolved_condition=resolved_condition,
                signals_already_extracted=all_extracted_signals,
            )
            all_extracted_signals.extend(llm_derived_signals)
            must_include_items.extend(
                _all_signal_values_for_type(llm_derived_signals, "must_include")
            )
            must_not_imply_items.extend(
                _all_signal_values_for_type(llm_derived_signals, "must_not_imply")
            )

        return IcdCodeSemanticConstraints(
            icd_code=resolved_condition.icd_code,
            icd_short_description=resolved_condition.icd_short_description,
            laterality=laterality,
            encounter_type=encounter_type,
            temporal_states=temporal_states,
            severity_qualifiers=severity_qualifiers,
            with_complication_flags=with_complication_flags,
            without_complication_flags=without_complication_flags,
            is_unspecified_code=is_unspecified,
            chapter_family=chapter_family,
            must_include_in_note=must_include_items,
            must_not_imply_in_note=must_not_imply_items,
            all_extracted_signals=all_extracted_signals,
            constraint_extractor_prompt_id=constraint_extractor_prompt_id,
            constraint_extractor_prompt_version=constraint_extractor_prompt_version,
        )

    # ------------------------------------------------------------------
    # Tier 3 private implementation
    # ------------------------------------------------------------------

    def _run_llm_escalation_and_return_signals(
        self,
        *,
        resolved_condition: ResolvedConditionCode,
        signals_already_extracted: list[ExtractedSemanticSignal],
    ) -> tuple[list[ExtractedSemanticSignal], str | None, str | None]:
        """
        Calls the LLM to derive must_include and must_not_imply items for
        codes that the deterministic and family rule tiers could not handle.

        Returns a list of ExtractedSemanticSignal objects with signal_type
        'must_include' or 'must_not_imply', sourced from LLM_ESCALATION.
        Returns an empty list if the LLM call fails or returns unparseable JSON.
        """
        prompt_spec = build_constraint_extraction_prompt_spec(
            icd_code=resolved_condition.icd_code,
            long_description=resolved_condition.icd_long_description,
            short_description=resolved_condition.icd_short_description,
            signals_already_extracted=signals_already_extracted,
        )
        escalation_prompt = compose_chat_prompt(
            system_prompt=prompt_spec.system_prompt,
            user_prompt=prompt_spec.user_prompt,
        )

        try:
            response_dict = self._llm_escalation_client.generate_json(  # type: ignore[union-attr]
                escalation_prompt,
                response_schema=prompt_spec.response_schema,
            )
        except Exception:
            logger.warning(
                "LLM escalation failed for code %s — skipping Tier 3.",
                resolved_condition.icd_code,
                exc_info=True,
            )
            return [], None, None

        llm_derived_signals: list[ExtractedSemanticSignal] = []

        for must_include_item in response_dict.get("must_include_in_note", []):
            if isinstance(must_include_item, str) and must_include_item.strip():
                llm_derived_signals.append(
                    ExtractedSemanticSignal(
                        signal_type="must_include",
                        extracted_value=must_include_item.strip(),
                        confidence=0.78,
                        extraction_source=ConstraintExtractionSource.LLM_ESCALATION,
                        matched_text_span=None,
                    )
                )

        for must_not_imply_item in response_dict.get("must_not_imply_in_note", []):
            if isinstance(must_not_imply_item, str) and must_not_imply_item.strip():
                llm_derived_signals.append(
                    ExtractedSemanticSignal(
                        signal_type="must_not_imply",
                        extracted_value=must_not_imply_item.strip(),
                        confidence=0.78,
                        extraction_source=ConstraintExtractionSource.LLM_ESCALATION,
                        matched_text_span=None,
                    )
                )

        return (
            llm_derived_signals,
            prompt_spec.prompt_id,
            prompt_spec.prompt_version,
        )

    # ------------------------------------------------------------------
    # Backward-compatible wrapper during naming transition
    # ------------------------------------------------------------------

    def extract_constraints_for_resolved_condition(
        self,
        resolved_condition: ResolvedConditionCode,
    ) -> IcdCodeSemanticConstraints:
        return self.extract_semantic_constraints_for_resolved_condition(resolved_condition)
