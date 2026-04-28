"""
Deterministic pre-check runner for generated clinical notes.

These checks are intentionally cheap and fully code-based so obviously weak
notes can be filtered before LLM-based evaluation begins.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from clinical_note_generation_v3.core.models.constraints import (
    ClinicalBundleSemanticConstraints,
)
from clinical_note_generation_v3.core.models.evaluation import (
    DeterministicPreCheckOutcome,
)
from clinical_note_generation_v3.core.models.note import GeneratedClinicalNote

_GENERIC_ICD_CODE_PATTERN = re.compile(
    r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b",
    re.IGNORECASE,
)


class DeterministicPreCheckRunner:
    """
    Runs deterministic quality gates before support verification and rubric judging.
    """

    def __init__(
        self,
        *,
        minimum_note_character_count: int = 500,
        near_duplicate_similarity_threshold: float = 0.92,
        maximum_repeated_line_count: int = 2,
    ) -> None:
        self._minimum_note_character_count = minimum_note_character_count
        self._near_duplicate_similarity_threshold = near_duplicate_similarity_threshold
        self._maximum_repeated_line_count = maximum_repeated_line_count

    def run_prechecks(
        self,
        *,
        generated_clinical_note: GeneratedClinicalNote,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
        recent_accepted_note_texts: list[str] | None = None,
        candidate_note_embedding: list[float] | None = None,
        recent_accepted_note_embeddings: list[list[float]] | None = None,
    ) -> DeterministicPreCheckOutcome:
        """
        Run all deterministic pre-checks and return pass or hard_fail.

        Near-duplicate detection runs in two stages:
          1. Cosine similarity on pre-computed embeddings (primary semantic gate).
             Catches semantic rewrites and synonym swaps.  Requires the caller to
             supply candidate_note_embedding and recent_accepted_note_embeddings.
          2. Token-set Jaccard similarity (secondary lexical gate).
             Always runs; catches near-verbatim copies regardless of embedding
             availability, but misses semantic rewrites.

        When embedding vectors are not supplied, only the Jaccard gate runs.
        Callers should provide embeddings whenever an embedding client is available.
        """
        note_text = generated_clinical_note.note_text
        failure_reasons: list[str] = []

        minimum_length_failure_reason = self._check_minimum_note_length(note_text)
        if minimum_length_failure_reason:
            failure_reasons.append(minimum_length_failure_reason)

        icd_code_leakage_failure_reason = self._check_for_icd_code_leakage(note_text)
        if icd_code_leakage_failure_reason:
            failure_reasons.append(icd_code_leakage_failure_reason)

        required_section_failure_reason = self._check_required_note_sections(note_text)
        if required_section_failure_reason:
            failure_reasons.append(required_section_failure_reason)

        repeated_boilerplate_failure_reason = self._check_for_repeated_boilerplate(note_text)
        if repeated_boilerplate_failure_reason:
            failure_reasons.append(repeated_boilerplate_failure_reason)

        copied_description_failure_reason = self._check_for_overly_literal_icd_description_copying(
            note_text=note_text,
            bundle_semantic_constraints=bundle_semantic_constraints,
        )
        if copied_description_failure_reason:
            failure_reasons.append(copied_description_failure_reason)

        if candidate_note_embedding is not None and recent_accepted_note_embeddings:
            cosine_near_duplicate_reason = self._check_for_near_duplicate_by_cosine_similarity(
                candidate_note_embedding=candidate_note_embedding,
                recent_accepted_note_embeddings=recent_accepted_note_embeddings,
            )
            if cosine_near_duplicate_reason:
                failure_reasons.append(cosine_near_duplicate_reason)

        jaccard_near_duplicate_reason = self._check_for_near_duplicate_by_jaccard(
            note_text=note_text,
            recent_accepted_note_texts=recent_accepted_note_texts or [],
        )
        if jaccard_near_duplicate_reason:
            failure_reasons.append(jaccard_near_duplicate_reason)

        if failure_reasons:
            return DeterministicPreCheckOutcome(
                outcome="hard_fail",
                failure_reasons=failure_reasons,
            )

        return DeterministicPreCheckOutcome(
            outcome="pass",
            failure_reasons=[],
        )

    def _check_minimum_note_length(self, note_text: str) -> str | None:
        if len(note_text.strip()) < self._minimum_note_character_count:
            return (
                f"Clinical note is too short: {len(note_text.strip())} characters; "
                f"minimum required is {self._minimum_note_character_count}."
            )
        return None

    def _check_for_icd_code_leakage(self, note_text: str) -> str | None:
        leaked_icd_code_match = _GENERIC_ICD_CODE_PATTERN.search(note_text)
        if leaked_icd_code_match:
            return (
                f"Clinical note contains a literal ICD code string: "
                f"{leaked_icd_code_match.group(0)}."
            )
        return None

    def _check_required_note_sections(self, note_text: str) -> str | None:
        normalized_note_text = note_text.lower()

        has_assessment_section = "assessment" in normalized_note_text
        has_plan_section = "plan" in normalized_note_text
        has_history_section = (
            "hpi" in normalized_note_text
            or "history of present illness" in normalized_note_text
            or "history" in normalized_note_text
        )

        missing_sections: list[str] = []
        if not has_history_section:
            missing_sections.append("history/HPI")
        if not has_assessment_section:
            missing_sections.append("assessment")
        if not has_plan_section:
            missing_sections.append("plan")

        if missing_sections:
            return (
                "Clinical note is missing required structural sections: "
                + ", ".join(missing_sections)
                + "."
            )
        return None

    def _check_for_repeated_boilerplate(self, note_text: str) -> str | None:
        normalized_lines = [line.strip().lower() for line in note_text.splitlines() if line.strip()]
        repeated_line_counts = Counter(normalized_lines)

        overly_repeated_lines = [
            repeated_line
            for repeated_line, repeated_count in repeated_line_counts.items()
            if repeated_count > self._maximum_repeated_line_count and len(repeated_line) > 20
        ]

        if overly_repeated_lines:
            return (
                "Clinical note overuses repeated boilerplate lines, including: "
                + "; ".join(overly_repeated_lines[:3])
                + "."
            )
        return None

    _ASSESSMENT_HEADERS = frozenset(
        {
            "assessment:",
            "diagnosis:",
            "diagnoses:",
            "impression:",
            "a/p:",
            "assessment and plan:",
            "assessment/plan:",
        }
    )
    _NON_ASSESSMENT_HEADERS = (
        "plan:",
        "hpi:",
        "history of present illness:",
        "subjective:",
        "chief complaint:",
        "review of systems:",
        "physical exam:",
        "physical examination:",
        "medications:",
        "past medical history:",
    )

    def _check_for_overly_literal_icd_description_copying(
        self,
        *,
        note_text: str,
        bundle_semantic_constraints: ClinicalBundleSemanticConstraints,
    ) -> str | None:
        note_lower = note_text.lower()

        for (
            code_semantic_constraints
        ) in bundle_semantic_constraints.per_code_note_writing_constraints:
            desc_lower = code_semantic_constraints.icd_short_description.lower().strip()
            if len(desc_lower) < 25:
                continue
            match_pos = note_lower.find(desc_lower)
            if match_pos == -1:
                continue
            if self._icd_description_is_in_assessment_section(note_lower, match_pos):
                continue
            return (
                "Clinical note appears to copy an official ICD description too literally: "
                f"'{code_semantic_constraints.icd_short_description}'."
            )

        return None

    def _icd_description_is_in_assessment_section(self, note_lower: str, match_pos: int) -> bool:
        preceding = note_lower[:match_pos]
        last_assessment_pos = max(
            (preceding.rfind(h) for h in self._ASSESSMENT_HEADERS),
            default=-1,
        )
        if last_assessment_pos == -1:
            return False
        last_other_pos = max(
            (preceding.rfind(h) for h in self._NON_ASSESSMENT_HEADERS),
            default=-1,
        )
        return last_assessment_pos > last_other_pos

    def _check_for_near_duplicate_by_cosine_similarity(
        self,
        *,
        candidate_note_embedding: list[float],
        recent_accepted_note_embeddings: list[list[float]],
    ) -> str | None:
        """
        Primary semantic near-duplicate gate.

        Computes cosine similarity between the candidate note's embedding and
        every stored accepted-note embedding.  No network calls — the caller
        pre-computes embeddings and passes them in.  This catches semantic
        rewrites and synonym swaps that Jaccard misses.
        """
        highest_cosine_similarity = max(
            self._compute_cosine_similarity(candidate_note_embedding, accepted_embedding)
            for accepted_embedding in recent_accepted_note_embeddings
        )
        if highest_cosine_similarity >= self._near_duplicate_similarity_threshold:
            return (
                "Clinical note is semantically too similar to a recently accepted note "
                f"(cosine_similarity={highest_cosine_similarity:.3f}, "
                f"threshold={self._near_duplicate_similarity_threshold:.3f})."
            )
        return None

    def _check_for_near_duplicate_by_jaccard(
        self,
        *,
        note_text: str,
        recent_accepted_note_texts: list[str],
    ) -> str | None:
        """
        Secondary lexical near-duplicate gate.

        Token-set Jaccard similarity catches near-verbatim copies but is
        insensitive to section reshuffling and synonym substitution.  Always
        runs as a secondary safety net; use cosine similarity as the primary
        gate whenever embeddings are available.
        """
        if not recent_accepted_note_texts:
            return None

        candidate_note_tokens = self._normalize_note_text_to_token_set(note_text)
        if not candidate_note_tokens:
            return None

        highest_jaccard_similarity = 0.0
        for recent_accepted_note_text in recent_accepted_note_texts:
            comparison_note_tokens = self._normalize_note_text_to_token_set(
                recent_accepted_note_text
            )
            if not comparison_note_tokens:
                continue
            similarity = self._compute_jaccard_similarity(
                candidate_note_tokens,
                comparison_note_tokens,
            )
            highest_jaccard_similarity = max(highest_jaccard_similarity, similarity)

        if highest_jaccard_similarity >= self._near_duplicate_similarity_threshold:
            return (
                "Clinical note is lexically too similar to a recently accepted note "
                f"(jaccard_similarity={highest_jaccard_similarity:.3f}, "
                f"threshold={self._near_duplicate_similarity_threshold:.3f})."
            )
        return None

    def _normalize_note_text_to_token_set(self, note_text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", note_text.lower()) if len(token) > 2}

    @staticmethod
    def _compute_cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        magnitude_a = math.sqrt(sum(a * a for a in vector_a))
        magnitude_b = math.sqrt(sum(b * b for b in vector_b))
        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)

    def _compute_jaccard_similarity(
        self,
        first_token_set: set[str],
        second_token_set: set[str],
    ) -> float:
        union_tokens = first_token_set | second_token_set
        if not union_tokens:
            return 0.0
        intersection_tokens = first_token_set & second_token_set
        return len(intersection_tokens) / len(union_tokens)
