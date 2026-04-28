"""
core/ports — abstract Protocol interfaces that decouple core logic from infrastructure.

  JSONGenerationClient  — LLM provider that returns parsed JSON dicts
  ICDCodeRepositoryPort — ICD record lookup and billable-record access
  ICDVectorIndexPort    — vector similarity search over ICD embeddings
  EmbeddingClient       — text embedding for vector indexing and querying
"""

from .llm_generation_port import JSONGenerationClient
from .icd_repository_port import ICDCodeRepositoryPort
from .icd_vector_index_port import ICDVectorIndexPort
from .embedding_port import EmbeddingClient

__all__ = [
    "JSONGenerationClient",
    "ICDCodeRepositoryPort",
    "ICDVectorIndexPort",
    "EmbeddingClient",
]
