"""
PostgreSQL JSONB persistence for accepted clinical notes.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from typing import Any

from clinical_note_generation_v3.config.settings import V3PipelineSettings
from clinical_note_generation_v3.core.models.evaluation import AcceptedClinicalNoteResult

try:
    import psycopg
    from psycopg import sql
except ImportError:  # pragma: no cover
    psycopg = None
    sql = None


class AcceptedClinicalNotePersistenceError(RuntimeError):
    """
    Raised when an accepted note cannot be persisted after retries.
    """


def build_persistable_accepted_note_document(
    accepted_clinical_note_result: AcceptedClinicalNoteResult,
) -> dict[str, Any]:
    """
    Build the JSON document persisted for one accepted note.

    Only adjudicated_icd10_codes are used as trustworthy final ICD labels.
    """

    combined_score = accepted_clinical_note_result.final_critique.combined_score
    accepted_note = accepted_clinical_note_result.accepted_note
    adjudication = accepted_clinical_note_result.adjudication_provenance

    return {
        "correlation_id": (
            accepted_clinical_note_result.correlation_id or accepted_note.correlation_id
        ),
        "timestamp": datetime.now(UTC).isoformat(),
        "template_id": accepted_clinical_note_result.seeded_bundle.template_id,
        "archetype": accepted_clinical_note_result.seeded_bundle.archetype,
        "encounter_context": accepted_clinical_note_result.seeded_bundle.encounter_context,
        "final_icd10_codes": list(accepted_clinical_note_result.adjudicated_icd10_codes),
        "generated_clinical_note": accepted_note.note_text,
        "combined_score": combined_score,
        "required_revision": accepted_clinical_note_result.required_revision,
        "generation_prompt_id": accepted_note.generation_prompt_id,
        "generation_prompt_version": accepted_note.generation_prompt_version,
        "generation_model": accepted_note.generation_model_info.model_name,
        "generation_provider": accepted_note.generation_model_info.provider_name,
        "fake_patient": {
            "name": accepted_note.fake_patient_name,
            "mrn": accepted_note.fake_patient_mrn,
            "date_of_birth": accepted_note.fake_patient_date_of_birth,
        },
        "adjudication": {
            "outcome": adjudication.outcome if adjudication is not None else None,
            "added_icd10_codes": (
                list(adjudication.added_icd10_codes) if adjudication is not None else []
            ),
            "removed_seeded_icd10_codes": (
                list(adjudication.removed_seeded_icd10_codes) if adjudication is not None else []
            ),
            "adjudication_rationale": (
                adjudication.adjudication_rationale if adjudication is not None else ""
            ),
            "adjudicator_prompt_id": (
                adjudication.adjudicator_prompt_id if adjudication is not None else None
            ),
            "adjudicator_prompt_version": (
                adjudication.adjudicator_prompt_version if adjudication is not None else None
            ),
        },
        "pipeline_observability": {
            "correlation_id": (
                accepted_clinical_note_result.correlation_id or accepted_note.correlation_id
            ),
            "pipeline_trace": list(accepted_clinical_note_result.pipeline_trace),
        },
    }


class PostgreSqlAcceptedClinicalNotesPersistence:
    """
    Persists accepted-note documents to PostgreSQL as JSONB.
    """

    def __init__(
        self,
        *,
        dsn: str,
        table_name: str,
        max_retries: int = 3,
        connection_factory: Any | None = None,
    ) -> None:
        self._dsn = dsn
        self._table_name = table_name
        self._max_retries = max_retries
        self._connection_factory = connection_factory
        self._connection: Any | None = None
        self._schema_initialized = False
        self._connection_lock = threading.Lock()

    def persist_accepted_note(
        self,
        accepted_clinical_note_result: AcceptedClinicalNoteResult,
    ) -> dict[str, Any]:
        document = build_persistable_accepted_note_document(accepted_clinical_note_result)
        last_error: Exception | None = None

        for attempt_number in range(1, self._max_retries + 1):
            try:
                connection = self._get_connection()
                self._ensure_schema(connection)
                self._insert_document(connection=connection, document=document)
                return document
            except Exception as error:
                last_error = error
                self._reset_connection()
                if attempt_number < self._max_retries:
                    time.sleep(min(0.25 * attempt_number, 1.0))

        raise AcceptedClinicalNotePersistenceError(
            "Failed to persist accepted clinical note to PostgreSQL "
            f"after {self._max_retries} attempts."
        ) from last_error

    def _get_connection(self) -> Any:
        with self._connection_lock:
            if self._connection is None:
                self._connection = self._create_connection()
            return self._connection

    def _create_connection(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory(self._dsn)
        if psycopg is None:
            raise AcceptedClinicalNotePersistenceError(
                "psycopg is required for PostgreSQL persistence but is not installed."
            )
        connection = psycopg.connect(self._dsn)
        connection.autocommit = True
        return connection

    def _reset_connection(self) -> None:
        with self._connection_lock:
            if self._connection is not None and hasattr(self._connection, "close"):
                try:
                    self._connection.close()
                except Exception:
                    pass
            self._connection = None
            self._schema_initialized = False

    def _ensure_schema(self, connection: Any) -> None:
        if self._schema_initialized:
            return

        if sql is None:
            create_table_statement = f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    correlation_id TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    document JSONB NOT NULL
                )
            """
            create_index_statement = (
                f"CREATE INDEX IF NOT EXISTS {self._table_name}_document_gin_idx "
                f"ON {self._table_name} USING GIN (document)"
            )
        else:
            create_table_statement = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    correlation_id TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    document JSONB NOT NULL
                )
                """
            ).format(table_name=sql.Identifier(self._table_name))
            create_index_statement = sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table_name} USING GIN (document)
                """
            ).format(
                index_name=sql.Identifier(f"{self._table_name}_document_gin_idx"),
                table_name=sql.Identifier(self._table_name),
            )

        with connection.cursor() as cursor:
            cursor.execute(create_table_statement)
            cursor.execute(create_index_statement)
        self._schema_initialized = True

    def _insert_document(self, *, connection: Any, document: dict[str, Any]) -> None:
        if sql is None:
            insert_statement = (
                f"INSERT INTO {self._table_name} (correlation_id, document) "
                "VALUES (%s, %s::jsonb)"
            )
        else:
            insert_statement = sql.SQL(
                """
                INSERT INTO {table_name} (correlation_id, document)
                VALUES (%s, %s::jsonb)
                """
            ).format(table_name=sql.Identifier(self._table_name))

        with connection.cursor() as cursor:
            cursor.execute(
                insert_statement,
                (document["correlation_id"], json.dumps(document, ensure_ascii=True)),
            )


_singleton_lock = threading.Lock()
_singleton_instance: PostgreSqlAcceptedClinicalNotesPersistence | None = None
_singleton_signature: tuple[str, str, int, Any | None] | None = None


def create_postgresql_accepted_notes_persistence(
    settings: V3PipelineSettings,
    *,
    connection_factory: Any | None = None,
) -> PostgreSqlAcceptedClinicalNotesPersistence | None:
    """
    Return the shared PostgreSQL persistence singleton when enabled.
    """

    if not settings.postgresql_persistence_enabled or not settings.postgresql_dsn:
        return None

    signature = (
        settings.postgresql_dsn,
        settings.postgresql_table_name,
        settings.postgresql_persistence_max_retries,
        connection_factory,
    )

    global _singleton_instance
    global _singleton_signature
    with _singleton_lock:
        if _singleton_instance is None or _singleton_signature != signature:
            _singleton_instance = PostgreSqlAcceptedClinicalNotesPersistence(
                dsn=settings.postgresql_dsn,
                table_name=settings.postgresql_table_name,
                max_retries=settings.postgresql_persistence_max_retries,
                connection_factory=connection_factory,
            )
            _singleton_signature = signature
        return _singleton_instance


def reset_postgresql_accepted_notes_persistence_singleton() -> None:
    """
    Reset the module singleton, primarily for tests.
    """

    global _singleton_instance
    global _singleton_signature
    with _singleton_lock:
        if _singleton_instance is not None:
            _singleton_instance._reset_connection()
        _singleton_instance = None
        _singleton_signature = None
