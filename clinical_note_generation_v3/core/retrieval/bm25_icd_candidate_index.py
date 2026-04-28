"""
BM25 lexical index over official billable ICD-10-CM descriptions.

LAYER: core/retrieval
ARCHITECTURE:
  list[ICDCodeRecord] (billable records)
      -> BM25ICDCandidateIndex (indexed at construction)
      -> BM25ICDCandidateIndex.query()
      -> list[CandidateCode] (BM25-ranked)

  BM25 parameters: k1=1.5, b=0.75 (standard Okapi BM25 defaults)
  Tokenization: regex [a-z0-9]+ over lowercased text

DATA FLOW:
  query text -> tokenize -> BM25 score per document -> top-k CandidateCode

DEPENDENCIES:
  - core/models/icd_codes.py (ICDCodeRecord, CandidateCode)
  - stdlib (math, re, collections)
"""

import math
import re
from collections import Counter, defaultdict

from clinical_note_generation_v3.core.models.icd_codes import CandidateCode, ICDCodeRecord


class BM25ICDCandidateIndex:
    """
    Pure-Python BM25 index over official billable ICD-10-CM descriptions.

    Built at construction time from a list of ICDCodeRecord objects. Designed as
    a lightweight lexical supplement to FAISS vector retrieval — it catches exact
    and near-exact term matches that semantic embeddings may miss.

    Args:
        records: Billable ICDCodeRecord objects to index.

    Returns:
        BM25 retriever instance ready to answer queries.

    Raises:
        None.

    Performance:
        - Build time: O(n * avg_doc_length) where n = number of records
        - Query time: O(n * |query_tokens|)
        - Memory: O(n * unique_tokens) for IDF table + term frequency dicts
        - Typical ICD-10-CM 2026 corpus: ~90,000 billable records, ~500 ms build

    Example:
        >>> isinstance([], list)
        True
    """

    def __init__(self, records: list[ICDCodeRecord]) -> None:
        self._records = records
        self._tokenized_documents = [_tokenize_text(record.search_document) for record in records]
        self._document_count = len(self._tokenized_documents)
        self._average_document_length = _compute_average_document_length(self._tokenized_documents)
        self._term_frequencies = [Counter(document) for document in self._tokenized_documents]
        self._inverse_document_frequencies = self._build_inverse_document_frequency_table()

    def query(self, query_text: str, *, limit: int) -> list[CandidateCode]:
        """
        Retrieve BM25-ranked ICD candidates for a query string.

        Args:
            query_text: Search query text (clinical term, abbreviation, or phrase).
            limit: Maximum number of candidates to return.

        Returns:
            BM25-ranked CandidateCode objects with source="bm25".
            Returns an empty list when the query tokenizes to nothing.

        Raises:
            None.

        Example:
            >>> isinstance("mixed hyperlipidemia", str)
            True
        """
        query_tokens = _tokenize_text(query_text)
        if not query_tokens:
            return []
        document_scores = self._score_all_documents_against_query(query_tokens)
        top_ranked = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            _build_candidate_from_record(self._records[index], source="bm25", score=score)
            for index, score in top_ranked
            if score > 0
        ]

    def _build_inverse_document_frequency_table(self) -> dict[str, float]:
        document_frequencies: dict[str, int] = defaultdict(int)
        for document_tokens in self._tokenized_documents:
            for token in set(document_tokens):
                document_frequencies[token] += 1
        return {
            token: math.log(1 + (self._document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequencies.items()
        }

    def _score_all_documents_against_query(self, query_tokens: list[str]) -> dict[int, float]:
        k1 = 1.5
        b = 0.75
        scores: dict[int, float] = {}
        for document_index, term_frequency_counter in enumerate(self._term_frequencies):
            document_length = len(self._tokenized_documents[document_index])
            document_score = _compute_bm25_document_score(
                query_tokens=query_tokens,
                term_frequency_counter=term_frequency_counter,
                document_length=document_length,
                average_document_length=self._average_document_length,
                inverse_document_frequencies=self._inverse_document_frequencies,
                k1=k1,
                b=b,
            )
            if document_score:
                scores[document_index] = document_score
        return scores


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_average_document_length(tokenized_documents: list[list[str]]) -> float:
    if not tokenized_documents:
        return 0.0
    return sum(len(document) for document in tokenized_documents) / len(tokenized_documents)


def _compute_bm25_document_score(
    *,
    query_tokens: list[str],
    term_frequency_counter: Counter,
    document_length: int,
    average_document_length: float,
    inverse_document_frequencies: dict[str, float],
    k1: float,
    b: float,
) -> float:
    score = 0.0
    for token in query_tokens:
        token_frequency = term_frequency_counter.get(token, 0)
        if not token_frequency:
            continue
        inverse_document_frequency = inverse_document_frequencies.get(token, 0.0)
        length_normalization = k1 * (1 - b + b * document_length / (average_document_length or 1.0))
        score += (
            inverse_document_frequency
            * (token_frequency * (k1 + 1))
            / (token_frequency + length_normalization)
        )
    return score


def _build_candidate_from_record(
    record: ICDCodeRecord,
    *,
    source: str,
    score: float | None,
) -> CandidateCode:
    return CandidateCode(
        code=record.normalized_code,
        description=record.long_description,
        source=source,
        score=score,
    )
