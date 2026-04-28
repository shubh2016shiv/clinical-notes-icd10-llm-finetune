"""
Writer for aggregate pipeline batch metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from clinical_note_generation_v3.core.models.evaluation import PipelineBatchRunMetrics


class BatchMetricsWriter:
    """
    Writes aggregate batch metrics as JSON.
    """

    def write_pipeline_batch_metrics(
        self,
        *,
        pipeline_batch_run_metrics: PipelineBatchRunMetrics,
        output_file_path: Path,
    ) -> None:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with output_file_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                pipeline_batch_run_metrics.model_dump(mode="json"),
                output_file,
                ensure_ascii=True,
                indent=2,
            )

    def write_batch_metrics(
        self,
        *,
        pipeline_batch_run_metrics: PipelineBatchRunMetrics,
        output_file_path: Path,
    ) -> None:
        self.write_pipeline_batch_metrics(
            pipeline_batch_run_metrics=pipeline_batch_run_metrics,
            output_file_path=output_file_path,
        )
