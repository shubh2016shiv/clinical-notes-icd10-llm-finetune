"""
core — the innermost layer of the v3 pipeline.

Contains:
  core/models/     — Pydantic data contracts (no external dependencies)
  core/ports/      — Abstract Protocol interfaces for dependency inversion
                     (populated in Phase 2 from infrastructure_BACKUP)
  core/services/   — Business logic services
                     (populated in Phase 4: constraint extractor,
                      support verifier, deterministic checks)
  core/retrieval/  — BM25 + FAISS hybrid retrieval
                     (copied from core_BACKUP/retrieval in Phase 3)

Nothing in core/ may import from application/, infrastructure/, or scripts/.
"""
