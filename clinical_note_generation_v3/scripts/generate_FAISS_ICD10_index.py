#!/usr/bin/env python3
"""
Build the v3 FAISS ICD-10-CM candidate index with embedding provenance.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_note_generation_v3.config.settings import V3PipelineSettings  # noqa: E402
from clinical_note_generation_v3.infrastructure.data_preprocessing.official_icd_loader import (  # noqa: E402
    OfficialICDCodeRepository,
)
from clinical_note_generation_v3.infrastructure.embedding_provider.embedding_client_factory import (  # noqa: E402
    create_default_openai_embedding_client,
    create_embedding_client,
)
from clinical_note_generation_v3.infrastructure.vector_store.faiss_icd_candidate_index import (  # noqa: E402
    FAISSICDCandidateIndex,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the v3 FAISS ICD-10-CM vector index with embedding provenance."
    )
    parser.add_argument(
        "--reset-existing",
        action="store_true",
        help="Remove the configured v3 FAISS directory before rebuilding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Embedding batch size for index build "
            "(default: CLINICAL_V3_FAISS_INDEX_BATCH_SIZE from settings)."
        ),
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Limit billable ICD records for smoke tests.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Override FAISS persist directory.",
    )
    parser.add_argument(
        "--prefer-gpu",
        dest="prefer_gpu",
        action="store_true",
        default=None,
        help="Prefer GPU FAISS when available.",
    )
    parser.add_argument(
        "--no-prefer-gpu",
        dest="prefer_gpu",
        action="store_false",
        help="Use CPU FAISS even when GPU FAISS is available.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = V3PipelineSettings()
    persist_directory = (args.persist_dir or settings.faiss_persist_directory).resolve()
    prefer_gpu = settings.prefer_faiss_gpu if args.prefer_gpu is None else args.prefer_gpu
    batch_size = args.batch_size if args.batch_size is not None else settings.faiss_index_batch_size

    print("Clinical Note Generation v3 FAISS ICD-10 Index Build")
    print("=" * 60)
    print(f"Official ICD order file: {settings.official_icd_order_path}")
    print(f"Persist directory: {persist_directory}")
    print(f"Batch size: {batch_size}")
    print(f"Limit records: {args.limit_records if args.limit_records is not None else 'all'}")
    print(f"Prefer GPU: {prefer_gpu}")
    print("=" * 60)

    if args.reset_existing:
        reset_target = _resolve_reset_target(
            persist_directory=persist_directory,
            explicit_persist_dir=args.persist_dir is not None,
            settings=settings,
        )
        if reset_target.exists():
            logger.info("Removing existing FAISS directory: %s", reset_target)
            shutil.rmtree(reset_target)
        else:
            logger.info("No existing FAISS directory to remove: %s", reset_target)

    existing_manifest = None
    if not args.reset_existing:
        try:
            existing_manifest = FAISSICDCandidateIndex.read_manifest(persist_directory)
        except RuntimeError:
            existing_manifest = None

    if existing_manifest:
        if str(existing_manifest.get("embedding_provider", "")).lower() != "openai":
            raise RuntimeError(
                "Existing FAISS manifest was built with "
                f"{existing_manifest.get('embedding_provider')}/"
                f"{existing_manifest.get('embedding_model')}. "
                "Default v3 indexing now requires OpenAI embeddings. "
                "Rebuild with --reset-existing."
            )
        logger.info(
            "Existing FAISS manifest found; using recorded embedding model %s/%s",
            existing_manifest["embedding_provider"],
            existing_manifest["embedding_model"],
        )
        embedding_client = create_embedding_client(
            provider_name=str(existing_manifest["embedding_provider"]),
            model_name=str(existing_manifest["embedding_model"]),
            settings=settings,
        )
        embedding_provider_name = str(existing_manifest["embedding_provider"])
        embedding_model_name = str(existing_manifest["embedding_model"])
        candidate_chain = list(existing_manifest.get("candidate_chain", []))
    else:
        logger.info(
            "Using OpenAI embedding model for FAISS indexing: openai/%s",
            settings.openai_embedding_model,
        )
        selected_embedding = create_default_openai_embedding_client(settings)
        probe_vector = selected_embedding.client.embed_query(
            "ICD-10-CM diagnosis embedding provider probe"
        )
        logger.info(
            "Selected embedding model for FAISS indexing: %s dimension=%d",
            selected_embedding.display_name,
            len(probe_vector),
        )
        embedding_client = selected_embedding.client
        embedding_provider_name = selected_embedding.provider_name
        embedding_model_name = selected_embedding.model_name
        candidate_chain = selected_embedding.candidate_chain

    repository = OfficialICDCodeRepository(settings.official_icd_order_path)
    vector_index = FAISSICDCandidateIndex(
        repository=repository,
        embedding_client=embedding_client,
        persist_directory=persist_directory,
        prefer_gpu=prefer_gpu,
    )
    built_count = vector_index.build(
        batch_size=batch_size,
        limit_records=args.limit_records,
        reset_existing=args.reset_existing,
        embedding_provider_name=embedding_provider_name,
        embedding_model_name=embedding_model_name,
        candidate_chain=candidate_chain,
        official_icd_order_filename=settings.icd_order_filename,
        build_command=" ".join(sys.argv),
    )

    if built_count == 0 and vector_index.count > 0:
        print("\nExisting FAISS index already present; no rebuild performed.")
    else:
        print("\nFAISS index build complete.")
    print(f"Records indexed: {vector_index.count}")
    print(f"Backend: {vector_index.backend_name}")
    print(f"Embedding model: {embedding_provider_name}/{embedding_model_name}")
    print(f"Manifest: {vector_index.manifest_path}")
    return 0


def _resolve_reset_target(
    *,
    persist_directory: Path,
    explicit_persist_dir: bool,
    settings: V3PipelineSettings,
) -> Path:
    if explicit_persist_dir:
        return persist_directory

    default_faiss_root = (PROJECT_ROOT / "clinical_note_generation_v3" / ".faiss").resolve()
    configured_faiss_root = settings.faiss_persist_directory.resolve().parent
    if not _is_relative_to(configured_faiss_root, default_faiss_root):
        raise RuntimeError(
            "Refusing to reset FAISS because the configured directory is outside "
            f"{default_faiss_root}. Use --persist-dir to reset an explicit custom path."
        )
    return configured_faiss_root


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
