"""
Embedding client selection for v3 FAISS indexing and retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from clinical_note_generation_v3.config.settings import V3PipelineSettings
from clinical_note_generation_v3.core.ports.embedding_port import EmbeddingClient
from clinical_note_generation_v3.infrastructure.embedding_provider.gemini_embedding_client import (
    GeminiEmbeddingClient,
)
from clinical_note_generation_v3.infrastructure.embedding_provider.openai_embedding_client import (
    OpenAIEmbeddingClient,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingCandidate:
    provider_name: str
    model_name: str

    @property
    def display_name(self) -> str:
        return f"{self.provider_name}/{self.model_name}"


@dataclass(frozen=True)
class SelectedEmbeddingClient:
    client: EmbeddingClient
    provider_name: str
    model_name: str
    candidate_chain: list[str]

    @property
    def display_name(self) -> str:
        return f"{self.provider_name}/{self.model_name}"


def default_embedding_candidates(settings: V3PipelineSettings) -> list[EmbeddingCandidate]:
    return [
        EmbeddingCandidate("openai", settings.openai_embedding_model),
    ]


def create_embedding_client(
    *,
    provider_name: str,
    model_name: str,
    settings: V3PipelineSettings,
) -> EmbeddingClient:
    provider_key = provider_name.strip().lower()
    if provider_key == "gemini":
        return GeminiEmbeddingClient(model_name=model_name, api_key=settings.gemini_api_key)
    if provider_key == "openai":
        return OpenAIEmbeddingClient(model_name=model_name, api_key=settings.openai_api_key)
    raise RuntimeError(f"Unsupported embedding provider '{provider_name}'.")


def create_default_openai_embedding_client(settings: V3PipelineSettings) -> SelectedEmbeddingClient:
    client = OpenAIEmbeddingClient(
        model_name=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    return SelectedEmbeddingClient(
        client=client,
        provider_name="openai",
        model_name=settings.openai_embedding_model,
        candidate_chain=[f"openai/{settings.openai_embedding_model}"],
    )


def select_first_working_embedding_client(
    *,
    settings: V3PipelineSettings,
    candidates: list[EmbeddingCandidate] | None = None,
    probe_text: str = "ICD-10-CM diagnosis embedding provider probe",
    client_factory: Callable[..., EmbeddingClient] = create_embedding_client,
) -> SelectedEmbeddingClient:
    resolved_candidates = candidates or default_embedding_candidates(settings)
    candidate_chain = [candidate.display_name for candidate in resolved_candidates]
    failures: list[str] = []

    for candidate in resolved_candidates:
        logger.info("Testing embedding model for FAISS indexing: %s", candidate.display_name)
        try:
            client = client_factory(
                provider_name=candidate.provider_name,
                model_name=candidate.model_name,
                settings=settings,
            )
            probe_vector = client.embed_query(probe_text)
            if not probe_vector:
                raise RuntimeError("probe returned an empty vector")
            logger.info(
                "Selected embedding model for FAISS indexing: %s dimension=%d",
                candidate.display_name,
                len(probe_vector),
            )
            return SelectedEmbeddingClient(
                client=client,
                provider_name=candidate.provider_name,
                model_name=candidate.model_name,
                candidate_chain=candidate_chain,
            )
        except Exception as error:
            failure_message = f"{candidate.display_name}: {type(error).__name__}: {error}"
            failures.append(failure_message)
            logger.warning("Embedding model unavailable for FAISS indexing: %s", failure_message)

    raise RuntimeError(
        "No embedding model could be used for FAISS indexing. Attempts: " + " | ".join(failures)
    )
