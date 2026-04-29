"""
Writer for minimal accepted-note training artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from clinical_note_generation_v3.core.models.evaluation import AcceptedClinicalNoteResult


class TrainingArtifactWriter:
    """
    Writes minimal training rows for accepted notes.
    """

    def write_accepted_note_training_rows(
        self,
        *,
        accepted_clinical_note_results: list[AcceptedClinicalNoteResult],
        output_file_path: Path,
        append: bool = True,
    ) -> None:
        if not accepted_clinical_note_results:
            return

        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if append else "w"

        with output_file_path.open(file_mode, encoding="utf-8") as output_file:
            for accepted_clinical_note_result in accepted_clinical_note_results:
                adjudicated_codes = list(accepted_clinical_note_result.adjudicated_icd10_codes)
                if not adjudicated_codes:
                    raise RuntimeError(
                        "Accepted note is missing adjudicated ICD-10-CM codes; "
                        "refusing to write training labels."
                    )
                training_row = {
                    "clinical_note": accepted_clinical_note_result.accepted_note.note_text,
                    "icd10_codes": adjudicated_codes,
                    "seeded_icd10_codes": list(accepted_clinical_note_result.seeded_icd10_codes),
                }
                output_file.write(json.dumps(training_row, ensure_ascii=True) + "\n")

    def write_training_rows(
        self,
        *,
        accepted_clinical_note_results: list[AcceptedClinicalNoteResult],
        output_file_path: Path,
        append: bool = True,
    ) -> None:
        self.write_accepted_note_training_rows(
            accepted_clinical_note_results=accepted_clinical_note_results,
            output_file_path=output_file_path,
            append=append,
        )
