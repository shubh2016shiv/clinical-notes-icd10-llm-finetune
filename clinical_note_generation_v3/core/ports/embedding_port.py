"""
Embedding provider port (interface) for the v3 pipeline.

LAYER: core/ports
ARCHITECTURE:
  core/retrieval and infrastructure/vector_store depend on this protocol.
  Concrete implementations live in infrastructure/embedding_provider/.

  EmbeddingClient (Protocol)
      <- OpenAIEmbeddingClient       (infrastructure/embedding_provider/)
      <- GeminiEmbeddingClient       (infrastructure/embedding_provider/)
      <- DeterministicHashEmbeddingClient (infrastructure/embedding_provider/)

DATA FLOW:
  text(s) -> EmbeddingClient -> list[float] vector(s)

DEPENDENCIES:
  - typing (Protocol, runtime_checkable)
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """
    Structural protocol for all embedding provider clients.

    Any class that implements embed_documents and embed_query with the correct
    signatures satisfies this protocol without explicit inheritance. Used as the
    type annotation throughout core/ and infrastructure/vector_store/ to enforce
    the Dependency Inversion Principle.

    Note:
        Annotate constructor parameters with this protocol, not with concrete
        client classes, so that tests can inject DeterministicHashEmbeddingClient
        without calling live APIs.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of documents for indexing.

        Args:
            texts: Document strings to embed.

        Returns:
            List of float vectors in the same order as texts.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single search query for retrieval.

        Args:
            text: Query string.

        Returns:
            Single float vector.
        """
        ...
