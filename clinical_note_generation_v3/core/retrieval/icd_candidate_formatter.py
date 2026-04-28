"""
Candidate formatting helpers for ICD selector prompts.

LAYER: core/retrieval
ARCHITECTURE:
  list[CandidateCode]
      -> assign_candidate_ids()      (stable one-based id assignment)
      -> format_candidates_as_markdown_kv()  (Markdown-KV text for prompt)
      -> selector prompt payload

  LLMs compare options more reliably when candidates are presented as explicit
  records with stable integer ids rather than as a raw JSON array. The Markdown-KV
  format makes each candidate scannable without requiring JSON parsing logic in
  the model's attention window.

DATA FLOW:
  list[CandidateCode] (retrieved) -> list[CandidateCode] (with ids) -> str (prompt payload)

DEPENDENCIES:
  - core/models/icd_codes.py (CandidateCode)
"""

from clinical_note_generation_v3.core.models.icd_codes import CandidateCode


def assign_candidate_ids(candidates: list[CandidateCode]) -> list[CandidateCode]:
    """
    Return candidates with stable one-based candidate_id values assigned.

    The ids are used by the selector to reference codes and by the validator
    to confirm that selected candidate_ids match the claimed codes.

    Args:
        candidates: Retrieved ICD candidate list (candidate_id may be None).

    Returns:
        New list of CandidateCode objects with candidate_id set to 1-based index.

    Raises:
        None.

    Example:
        >>> assign_candidate_ids([CandidateCode(code="E11.9", description="T2DM")])[0].candidate_id
        1
    """
    return [
        candidate.model_copy(update={"candidate_id": index})
        for index, candidate in enumerate(candidates, start=1)
    ]


def format_candidates_as_markdown_kv(candidates: list[CandidateCode]) -> str:
    """
    Format a candidate list as Markdown key-value records for use in LLM prompts.

    Each candidate becomes a Markdown-level-3 section with fields on separate
    lines, making it easy for the LLM to scan individual candidates. The
    candidate_id field is the primary selection key referenced in selector output.

    Args:
        candidates: Candidate ICD codes with candidate_id already populated.

    Returns:
        Multiline Markdown-KV string suitable for prompt injection.

    Raises:
        None.

    Example:
        >>> result = format_candidates_as_markdown_kv([CandidateCode(candidate_id=1, code="E11.9", description="T2DM")])
        >>> "Candidate 1" in result
        True
    """
    lines: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate.candidate_id or index
        lines.extend(
            [
                f"### Candidate {candidate_id}",
                f"candidate_id: {candidate_id}",
                f"code: {candidate.code}",
                f"description: {candidate.description}",
                f"source: {candidate.source}",
            ]
        )
        if candidate.score is not None:
            lines.append(f"retrieval_score: {candidate.score:.6f}")
        lines.append("")
    return "\n".join(lines).strip()
