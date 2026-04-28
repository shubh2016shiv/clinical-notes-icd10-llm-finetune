"""
Embedding provider implementations for the v3 pipeline.
"""

from .embedding_client_factory import (
    EmbeddingCandidate,
    SelectedEmbeddingClient,
    create_default_openai_embedding_client,
    create_embedding_client,
    default_embedding_candidates,
    select_first_working_embedding_client,
)
from .gemini_embedding_client import GeminiEmbeddingClient
from .openai_embedding_client import OpenAIEmbeddingClient

__all__ = [
    "EmbeddingCandidate",
    "GeminiEmbeddingClient",
    "OpenAIEmbeddingClient",
    "SelectedEmbeddingClient",
    "create_default_openai_embedding_client",
    "create_embedding_client",
    "default_embedding_candidates",
    "select_first_working_embedding_client",
]
