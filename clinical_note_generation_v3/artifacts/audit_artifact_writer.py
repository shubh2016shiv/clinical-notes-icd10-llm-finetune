"""
Writer for detailed audit artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from clinical_note_generation_v3.core.models.evaluation import (
    AcceptedClinicalNoteResult,
    RejectedClinicalNoteResult,
)


class AuditArtifactWriter:
    """
    Writes detailed audit rows for accepted and rejected notes.
    """

    def write_clinical_note_audit_rows(
        self,
        *,
        clinical_note_results: list[AcceptedClinicalNoteResult | RejectedClinicalNoteResult],
        output_file_path: Path,
        append: bool = True,
    ) -> None:
        if not clinical_note_results:
            return

        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if append else "w"

        with output_file_path.open(file_mode, encoding="utf-8") as output_file:
            for clinical_note_result in clinical_note_results:
                result_status = (
                    "accepted"
                    if isinstance(clinical_note_result, AcceptedClinicalNoteResult)
                    else "rejected"
                )
                audit_row = {
                    "result_status": result_status,
                    **clinical_note_result.model_dump(mode="json"),
                }
                output_file.write(json.dumps(audit_row, ensure_ascii=True) + "\n")

    def write_audit_rows(
        self,
        *,
        clinical_note_results: list[AcceptedClinicalNoteResult | RejectedClinicalNoteResult],
        output_file_path: Path,
        append: bool = True,
    ) -> None:
        self.write_clinical_note_audit_rows(
            clinical_note_results=clinical_note_results,
            output_file_path=output_file_path,
            append=append,
        )
