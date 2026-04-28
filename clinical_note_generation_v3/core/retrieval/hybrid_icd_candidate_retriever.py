"""
Hybrid retriever combining FAISS vector and BM25 lexical ICD candidate search.

LAYER: core/retrieval
ARCHITECTURE:
  ICDCodeRepositoryPort  (billable records for BM25 index construction)
  ICDVectorIndexPort     (optional FAISS vector search)
      -> BM25ICDCandidateIndex (built at construction from billable_records)
      -> HybridICDCandidateRetriever.retrieve()
      -> deduplicated list[CandidateCode]

  Per-call retrieval strategy:
    1. BM25 per individual focus term (lexical, high precision for known terms)
    2. FAISS query over full note + all focus terms (semantic, high recall)
    3. BM25 query over full clinical note (fallback lexical sweep)
    4. Deduplicate by normalized code, preserve insertion order

DATA FLOW:
  (clinical_note, focus_terms, limit)
      -> BM25 per term + FAISS combined query
      -> _deduplicate_candidate_list()
      -> top-limit CandidateCode objects

DEPENDENCIES:
  - core/models/icd_codes.py (CandidateCode)
  - core/ports/icd_repository_port.py (ICDCodeRepositoryPort)
  - core/ports/icd_vector_index_port.py (ICDVectorIndexPort)
  - core/retrieval/bm25_icd_candidate_index.py (BM25ICDCandidateIndex)
"""

import math

from clinical_note_generation_v3.core.models.icd_codes import CandidateCode
from clinical_note_generation_v3.core.ports.icd_repository_port import ICDCodeRepositoryPort
from clinical_note_generation_v3.core.ports.icd_vector_index_port import ICDVectorIndexPort
from clinical_note_generation_v3.core.retrieval.bm25_icd_candidate_index import (
    BM25ICDCandidateIndex,
)


class HybridICDCandidateRetriever:
    """
    Combine FAISS vector candidates with BM25 lexical candidates for ICD retrieval.

    Builds a BM25 index at construction time over billable records. When a FAISS
    vector index is provided, semantic and lexical results are merged and
    deduplicated. Without FAISS, operates in BM25-only mode.

    Args:
        repository: ICD repository implementing ICDCodeRepositoryPort, used to
                    access billable_records for BM25 index construction.
        vector_index: Optional FAISS vector index implementing ICDVectorIndexPort.
                      When None, the retriever uses BM25 only.

    Returns:
        Retriever ready to answer hybrid queries.

    Raises:
        None.

    Example:
        >>> isinstance([], list)
        True
    """

    def __init__(
        self,
        *,
        repository: ICDCodeRepositoryPort,
        vector_index: ICDVectorIndexPort | None = None,
    ) -> None:
        self._repository = repository
        self._vector_index = vector_index
        self._bm25_index = BM25ICDCandidateIndex(repository.billable_records)

    def retrieve(
        self,
        *,
        clinical_note: str,
        focus_terms: list[str],
        limit: int,
    ) -> list[CandidateCode]:
        """
        Retrieve deduplicated ICD candidates using hybrid BM25 + FAISS strategy.

        Args:
            clinical_note: Full synthetic clinical note text.
            focus_terms: Expected condition terms or LLM-extracted normalized concepts.
            limit: Maximum number of candidates to return.

        Returns:
            Deduplicated, ranked CandidateCode list (at most `limit` items).

        Raises:
            Exception: Propagates vector index errors when FAISS is configured.

        Example:
            >>> isinstance("note", str)
            True
        """
        per_focus_term_limit = _compute_per_focus_term_limit(
            total_limit=limit, focus_term_count=len(focus_terms)
        )
        candidates: list[CandidateCode] = []

        for focus_term in focus_terms:
            candidates.extend(self._bm25_index.query(focus_term, limit=per_focus_term_limit))

        if self._vector_index and self._vector_index.count > 0:
            expanded_focus_terms = list(dict.fromkeys([*focus_terms, clinical_note]))
            vector_query_text = " ".join([clinical_note, *expanded_focus_terms])
            candidates.extend(self._vector_index.query(vector_query_text, limit=limit))

        candidates.extend(self._bm25_index.query(clinical_note, limit=max(limit, 25)))

        return _deduplicate_candidate_list(candidates)[:limit]


def _compute_per_focus_term_limit(*, total_limit: int, focus_term_count: int) -> int:
    if focus_term_count <= 0:
        return max(total_limit, 25)
    return max(8, math.ceil(total_limit / focus_term_count))


def _deduplicate_candidate_list(candidates: list[CandidateCode]) -> list[CandidateCode]:
    seen_normalized_codes: set[str] = set()
    deduplicated: list[CandidateCode] = []
    for candidate in candidates:
        normalized_code = candidate.code.replace(".", "").upper()
        if normalized_code in seen_normalized_codes:
            continue
        seen_normalized_codes.add(normalized_code)
        deduplicated.append(candidate)
    return deduplicated
