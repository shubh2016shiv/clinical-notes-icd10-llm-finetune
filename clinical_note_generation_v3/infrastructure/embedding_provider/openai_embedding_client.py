"""
OpenAI embedding client for FAISS-backed ICD retrieval in v3.
"""

from __future__ import annotations

import os


class OpenAIEmbeddingClient:
    """
    OpenAI embedding client that satisfies the v3 EmbeddingClient protocol.
    """

    provider_name = "openai"

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")

        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return [list(item.embedding) for item in response.data]
