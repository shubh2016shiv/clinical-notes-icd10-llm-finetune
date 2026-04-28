"""
Gemini embedding client for FAISS-backed ICD retrieval in v3.
"""

from __future__ import annotations

import os


class GeminiEmbeddingClient:
    """
    Gemini embedding client that satisfies the v3 EmbeddingClient protocol.
    """

    provider_name = "gemini"

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings.")

        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        self._genai = genai

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text, task_type="retrieval_document") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text, task_type="retrieval_query")

    def _embed_text(self, text: str, *, task_type: str) -> list[float]:
        response = self._genai.embed_content(
            model=self.model_name,
            content=text,
            task_type=task_type,
        )
        embedding = response.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Gemini embedding response did not include an embedding vector.")
        return [float(value) for value in embedding]
