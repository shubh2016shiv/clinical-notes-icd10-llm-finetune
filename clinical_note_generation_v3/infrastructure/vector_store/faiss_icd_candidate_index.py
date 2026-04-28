"""
FAISS-backed vector index for official billable ICD-10-CM codes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from clinical_note_generation_v3.core.models.icd_codes import CandidateCode, ICDCodeRecord
from clinical_note_generation_v3.core.ports.embedding_port import EmbeddingClient
from clinical_note_generation_v3.core.ports.icd_repository_port import ICDCodeRepositoryPort


class FAISSICDCandidateIndex:
    """
    Local FAISS (or NumPy) vector index for official billable ICD-10-CM codes.

    Automatically loads a previously persisted index on construction.
    """

    _INDEX_FILENAME = "icd10cm_2026_april_1.faiss"
    _NUMPY_INDEX_FILENAME = "icd10cm_2026_april_1.vectors.npz"
    _METADATA_FILENAME = "icd10cm_2026_april_1.metadata.jsonl"
    _MANIFEST_FILENAME = "icd10cm_2026_april_1.manifest.json"
    _MANIFEST_SCHEMA_VERSION = 1
    MANDATORY_HYBRID_REMEDIATION = (
        "Hybrid ICD retrieval requires a valid FAISS index built with the recorded "
        "embedding model. Rebuild with generate_FAISS_ICD10_index.py --reset-existing."
    )

    def __init__(
        self,
        *,
        repository: ICDCodeRepositoryPort,
        embedding_client: EmbeddingClient,
        persist_directory: Path,
        prefer_gpu: bool = True,
    ) -> None:
        self._faiss_module = _try_import_faiss()
        self._repository = repository
        self._embedding_client = embedding_client
        self._persist_directory = persist_directory
        self._index_path = persist_directory / self._INDEX_FILENAME
        self._numpy_index_path = persist_directory / self._NUMPY_INDEX_FILENAME
        self._metadata_path = persist_directory / self._METADATA_FILENAME
        self._manifest_path = persist_directory / self._MANIFEST_FILENAME
        self._prefer_gpu = prefer_gpu
        self._faiss_index = None
        self._numpy_matrix = None
        self._metadata: list[dict] = []
        self._manifest: dict | None = None
        self._load_persisted_index_if_available()

    @property
    def count(self) -> int:
        return len(self._metadata)

    @property
    def backend_name(self) -> str:
        if self._faiss_module is None:
            return "numpy-cpu-fallback"
        if not self._prefer_gpu:
            return "faiss-cpu"
        try:
            gpu_count = int(self._faiss_module.get_num_gpus())
        except Exception:
            gpu_count = 0
        return "faiss-gpu" if gpu_count > 0 else "faiss-cpu"

    @property
    def manifest(self) -> dict | None:
        return self._manifest

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @classmethod
    def manifest_path_for_directory(cls, persist_directory: Path) -> Path:
        return persist_directory / cls._MANIFEST_FILENAME

    @classmethod
    def read_manifest(cls, persist_directory: Path) -> dict:
        manifest_path = cls.manifest_path_for_directory(persist_directory)
        if not manifest_path.exists():
            raise RuntimeError(cls.MANDATORY_HYBRID_REMEDIATION)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def build(
        self,
        *,
        batch_size: int = 64,
        limit_records: int | None = None,
        reset_existing: bool = False,
        embedding_provider_name: str | None = None,
        embedding_model_name: str | None = None,
        candidate_chain: list[str] | None = None,
        corpus_id: str = "icd10cm_2026_april_1",
        official_icd_order_filename: str | None = None,
        build_command: str | None = None,
    ) -> int:
        if self.count > 0 and not reset_existing:
            return 0

        records = self._repository.billable_records
        if limit_records is not None:
            records = records[:limit_records]

        all_vectors: list[list[float]] = []
        all_metadata: list[dict] = []
        for record_batch in _iterate_record_batches(records, batch_size):
            batch_documents = [record.search_document for record in record_batch]
            all_vectors.extend(self._embedding_client.embed_documents(batch_documents))
            all_metadata.extend(_build_metadata_from_records(record_batch))

        if not all_vectors:
            self._faiss_index = None
            self._metadata = []
            return 0

        if self._faiss_module is None:
            self._numpy_matrix = _build_normalized_matrix(all_vectors)
            cpu_index = None
        else:
            cpu_index = self._build_faiss_cpu_index(all_vectors)
            self._faiss_index = self._place_index_on_gpu_if_available(cpu_index)

        self._metadata = all_metadata
        vector_dimension = len(all_vectors[0])
        manifest = self._build_manifest(
            corpus_id=corpus_id,
            official_icd_order_filename=official_icd_order_filename,
            record_count=len(all_metadata),
            vector_dimension=vector_dimension,
            embedding_provider_name=embedding_provider_name,
            embedding_model_name=embedding_model_name,
            candidate_chain=candidate_chain or [],
            build_command=build_command,
        )
        self._persist_index_to_disk(cpu_index, all_metadata, manifest)
        self._manifest = manifest
        return len(all_metadata)

    def query(self, query_text: str, *, limit: int = 60) -> list[CandidateCode]:
        if self._faiss_index is None and self._numpy_matrix is None:
            raise RuntimeError("FAISS ICD index is empty. Run build() before querying.")
        if not self._metadata:
            raise RuntimeError("ICD vector index metadata is empty. Rebuild the index.")

        query_vector = _build_normalized_matrix([self._embedding_client.embed_query(query_text)])
        effective_limit = min(limit, len(self._metadata))

        if self._faiss_module is None:
            similarity_scores, result_indexes = _numpy_cosine_search(
                self._numpy_matrix, query_vector, effective_limit
            )
        else:
            similarity_scores, result_indexes = self._faiss_index.search(
                query_vector, effective_limit
            )

        return _build_candidates_from_search_results(
            indexes=result_indexes[0],
            scores=similarity_scores[0],
            metadata=self._metadata,
        )

    def _load_persisted_index_if_available(self) -> None:
        if not self._manifest_path.exists():
            return
        self._manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if not self._metadata_path.exists():
            raise RuntimeError(self.MANDATORY_HYBRID_REMEDIATION)
        if not self._index_path.exists() and not self._numpy_index_path.exists():
            raise RuntimeError(self.MANDATORY_HYBRID_REMEDIATION)
        if self._manifest.get("schema_version") != self._MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(self.MANDATORY_HYBRID_REMEDIATION)

        manifest_provider = str(self._manifest.get("embedding_provider", ""))
        manifest_model = str(self._manifest.get("embedding_model", ""))
        client_provider = str(getattr(self._embedding_client, "provider_name", ""))
        client_model = str(getattr(self._embedding_client, "model_name", ""))
        if manifest_provider != client_provider or manifest_model != client_model:
            raise RuntimeError(
                f"{self.MANDATORY_HYBRID_REMEDIATION} "
                f"Index was built with {manifest_provider}/{manifest_model}; "
                f"retrieval client is {client_provider}/{client_model}."
            )

        if not self._metadata_path.exists():
            return
        self._metadata = [
            json.loads(line)
            for line in self._metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if self._faiss_module is not None and self._index_path.exists():
            cpu_index = self._faiss_module.read_index(str(self._index_path))
            self._faiss_index = self._place_index_on_gpu_if_available(cpu_index)
            return
        if self._numpy_index_path.exists():
            import numpy as np

            self._numpy_matrix = np.load(self._numpy_index_path)["vectors"]
            return

        raise RuntimeError(self.MANDATORY_HYBRID_REMEDIATION)

    def _build_faiss_cpu_index(self, vectors: list[list[float]]):
        matrix = _as_float32_array(vectors)
        self._faiss_module.normalize_L2(matrix)
        index = self._faiss_module.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index

    def _place_index_on_gpu_if_available(self, cpu_index):
        if not self._prefer_gpu:
            return cpu_index
        try:
            if int(self._faiss_module.get_num_gpus()) <= 0:
                return cpu_index
            gpu_resources = self._faiss_module.StandardGpuResources()
            return self._faiss_module.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
        except Exception:
            return cpu_index

    def _persist_index_to_disk(self, cpu_index, metadata: list[dict], manifest: dict) -> None:
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        if self._faiss_module is None:
            import numpy as np

            np.savez_compressed(self._numpy_index_path, vectors=self._numpy_matrix)
        else:
            self._faiss_module.write_index(cpu_index, str(self._index_path))
        with self._metadata_path.open("w", encoding="utf-8") as metadata_file:
            for metadata_row in metadata:
                metadata_file.write(json.dumps(metadata_row, ensure_ascii=False))
                metadata_file.write("\n")
        self._manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _build_manifest(
        self,
        *,
        corpus_id: str,
        official_icd_order_filename: str | None,
        record_count: int,
        vector_dimension: int,
        embedding_provider_name: str | None,
        embedding_model_name: str | None,
        candidate_chain: list[str],
        build_command: str | None,
    ) -> dict:
        client_provider = embedding_provider_name or str(
            getattr(self._embedding_client, "provider_name", "")
        )
        client_model = embedding_model_name or str(
            getattr(self._embedding_client, "model_name", "")
        )
        if not client_provider or not client_model:
            raise RuntimeError("FAISS manifest requires embedding_provider and embedding_model.")
        return {
            "schema_version": self._MANIFEST_SCHEMA_VERSION,
            "corpus_id": corpus_id,
            "official_icd_order_filename": official_icd_order_filename,
            "record_count": record_count,
            "vector_dimension": vector_dimension,
            "backend_name": self.backend_name,
            "embedding_provider": client_provider,
            "embedding_model": client_model,
            "candidate_chain": candidate_chain,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "build_command": build_command,
        }


def _try_import_faiss():
    try:
        import faiss

        return faiss
    except ImportError:
        return None


def _iterate_record_batches(
    records: list[ICDCodeRecord],
    batch_size: int,
) -> Iterable[list[ICDCodeRecord]]:
    for start_index in range(0, len(records), batch_size):
        yield records[start_index : start_index + batch_size]


def _build_metadata_from_records(records: list[ICDCodeRecord]) -> list[dict]:
    return [
        {
            "code": record.normalized_code,
            "undotted_code": record.code,
            "description": record.long_description,
            "short_description": record.short_description,
            "chapter_prefix": record.chapter_prefix,
            "is_billable": record.is_billable,
        }
        for record in records
    ]


def _as_float32_array(vectors: list[list[float]]):
    import numpy as np

    return np.asarray(vectors, dtype="float32")


def _build_normalized_matrix(vectors: list[list[float]]):
    import numpy as np

    matrix = _as_float32_array(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _numpy_cosine_search(index_matrix, query_vector, limit: int):
    import numpy as np

    similarity_scores = index_matrix @ query_vector[0]
    top_indexes = np.argsort(-similarity_scores)[:limit]
    return similarity_scores[top_indexes][None, :], top_indexes[None, :]


def _build_candidates_from_search_results(
    *,
    indexes,
    scores,
    metadata: list[dict],
) -> list[CandidateCode]:
    candidates: list[CandidateCode] = []
    for index, score in zip(indexes, scores):
        if int(index) < 0:
            continue
        metadata_row = metadata[int(index)]
        candidates.append(
            CandidateCode(
                code=str(metadata_row["code"]),
                description=str(metadata_row["description"]),
                source="faiss",
                score=float(score),
            )
        )
    return candidates
