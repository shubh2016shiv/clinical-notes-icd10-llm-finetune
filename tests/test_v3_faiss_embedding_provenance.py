from __future__ import annotations

import pytest

from clinical_note_generation_v3.config.settings import V3PipelineSettings
from clinical_note_generation_v3.core.models.icd_codes import ICDCodeRecord
from clinical_note_generation_v3.infrastructure.embedding_provider.embedding_client_factory import (
    default_embedding_candidates,
)
from clinical_note_generation_v3.infrastructure.vector_store.faiss_icd_candidate_index import (
    FAISSICDCandidateIndex,
)


class FakeEmbeddingClient:
    def __init__(self, provider_name: str, model_name: str, fail: bool = False) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError(f"{self.provider_name}/{self.model_name} unavailable")
        base = float((sum(ord(ch) for ch in text) % 10) + 1)
        return [base, base + 1.0, base + 2.0]


class FakeRepository:
    @property
    def billable_records(self) -> list[ICDCodeRecord]:
        return [
            ICDCodeRecord(
                code="E119",
                is_billable=True,
                short_description="Type 2 diabetes mellitus",
                long_description="Type 2 diabetes mellitus without complications",
            ),
            ICDCodeRecord(
                code="I10",
                is_billable=True,
                short_description="Essential hypertension",
                long_description="Essential primary hypertension",
            ),
        ]


def _settings() -> V3PipelineSettings:
    return V3PipelineSettings(
        gemini_api_key="gemini-key",
        openai_api_key="openai-key",
        deepseek_api_key="deepseek-key",
    )


def test_default_embedding_candidates_are_openai_only() -> None:
    candidates = default_embedding_candidates(_settings())

    assert [candidate.provider_name for candidate in candidates] == ["openai"]
    assert candidates[0].model_name == "text-embedding-3-small"


def test_faiss_manifest_blocks_mismatched_retrieval_embedding_model(tmp_path) -> None:
    index = FAISSICDCandidateIndex(
        repository=FakeRepository(),
        embedding_client=FakeEmbeddingClient("openai", "text-embedding-3-small"),
        persist_directory=tmp_path,
        prefer_gpu=False,
    )
    index.build(
        embedding_provider_name="openai",
        embedding_model_name="text-embedding-3-small",
        candidate_chain=["openai/text-embedding-3-small"],
        official_icd_order_filename="icd10cm-order-April-1-2026.txt",
        build_command="test-build",
    )

    manifest = FAISSICDCandidateIndex.read_manifest(tmp_path)
    assert manifest["embedding_provider"] == "openai"
    assert manifest["embedding_model"] == "text-embedding-3-small"

    with pytest.raises(RuntimeError, match="Index was built with openai/text-embedding-3-small"):
        FAISSICDCandidateIndex(
            repository=FakeRepository(),
            embedding_client=FakeEmbeddingClient("openai", "text-embedding-3-large"),
            persist_directory=tmp_path,
            prefer_gpu=False,
        )


def test_missing_manifest_fails_mandatory_hybrid_contract(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Hybrid ICD retrieval requires a valid FAISS index"):
        FAISSICDCandidateIndex.read_manifest(tmp_path)
