"""
core/retrieval — ICD candidate retrieval using BM25 and hybrid BM25+FAISS strategies.

  BM25ICDCandidateIndex       — pure-Python Okapi BM25 index over ICD descriptions
  HybridICDCandidateRetriever — merges FAISS vector results with BM25 lexical results
  assign_candidate_ids        — assigns stable 1-based IDs to a candidate list
  format_candidates_as_markdown_kv — formats candidates for LLM prompt injection
"""

from .bm25_icd_candidate_index import BM25ICDCandidateIndex
from .hybrid_icd_candidate_retriever import HybridICDCandidateRetriever
from .icd_candidate_formatter import assign_candidate_ids, format_candidates_as_markdown_kv

__all__ = [
    "BM25ICDCandidateIndex",
    "HybridICDCandidateRetriever",
    "assign_candidate_ids",
    "format_candidates_as_markdown_kv",
]
