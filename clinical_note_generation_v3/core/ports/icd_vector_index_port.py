"""
ICD vector index port (interface) for the v3 pipeline.

LAYER: core/ports
ARCHITECTURE:
  core/retrieval depends on this protocol.
  The concrete implementation lives in infrastructure/vector_store/.

  ICDVectorIndexPort (Protocol)
      <- FAISSICDCandidateIndex (infrastructure/vector_store/)

DATA FLOW:
  query text -> ICDVectorIndexPort -> list[CandidateCode]

DEPENDENCIES:
  - core/models/icd_codes.py (CandidateCode)
  - typing (Protocol, runtime_checkable)
"""

from typing import Protocol, runtime_checkable

from clinical_note_generation_v3.core.models.icd_codes import CandidateCode


@runtime_checkable
class ICDVectorIndexPort(Protocol):
    """
    Structural protocol for ICD vector search indexes.

    Defines the minimal query interface required by HybridICDCandidateRetriever
    in core/retrieval/ so that the retriever does not depend directly on the
    FAISS infrastructure class. This enables test injection of mock indexes and
    future replacement of FAISS with alternative vector stores.
    """

    @property
    def count(self) -> int:
        """Return the number of indexed records (0 if the index is empty)."""
        ...

    def query(self, query_text: str, *, limit: int) -> list[CandidateCode]:
        """
        Retrieve ICD candidates by vector similarity.

        Args:
            query_text: Query text to embed and search against the index.
            limit: Maximum number of candidates to return.

        Returns:
            Ranked CandidateCode objects.

        Raises:
            RuntimeError: If the index is empty and has not been built.
        """
        ...
