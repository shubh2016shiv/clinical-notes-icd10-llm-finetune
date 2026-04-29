#!/usr/bin/env python3
"""
Generate clinical notes using the v3 pipeline.

Usage:
  python -m clinical_note_generation_v3.scripts.generate_clinical_notes [--count COUNT] [--output-dir DIR]

Examples:
  # Generate 5 clinical notes
  python -m clinical_note_generation_v3.scripts.generate_clinical_notes --count 5

  # Generate 10 notes with custom output directory
  python -m clinical_note_generation_v3.scripts.generate_clinical_notes --count 10 --output-dir ./my_outputs
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_note_generation_v3.application.pipeline import (  # noqa: E402
    create_default_clinical_note_quality_pipeline,
)
from clinical_note_generation_v3.config.settings import V3PipelineSettings  # noqa: E402
from clinical_note_generation_v3.core.log import configure_structured_logging  # noqa: E402
from clinical_note_generation_v3.core.models.evaluation import (  # noqa: E402
    AcceptedClinicalNoteResult,
    RejectedClinicalNoteResult,
)


def configure_v3_cli_logging(*, log_verbosity: bool) -> None:
    """
    Keep intentional pipeline logs visible while suppressing noisy HTTP transport logs.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    configure_structured_logging(enabled=log_verbosity, log_level=logging.INFO)
    for logger_name in [
        "httpx",
        "httpcore",
        "openai",
        "openai._base_client",
        "openai._client",
        "urllib3",
        "google",
        "google.generativeai",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def main() -> int:
    """Generate clinical notes using the v3 pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate clinical notes using the v3 seeded clinical note quality pipeline."
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=5,
        help="Number of clinical notes to generate (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for generated notes (default: settings generated clinical notes directory)",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip writing artifact files (training, audit, metrics)",
    )
    args = parser.parse_args()

    # Load settings
    settings = V3PipelineSettings()
    configure_v3_cli_logging(log_verbosity=settings.log_verbosity)

    # Determine output directory
    output_dir = args.output_dir or settings.sample_data_directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Clinical Note Generation v3")
    print("=" * 50)
    print(f"Generating {args.count} clinical notes...")
    print(f"Output directory: {output_dir}")
    print(f"Primary LLM model: deepseek/{settings.deepseek_model}")
    print(f"Fallback LLM model: openai/{settings.openai_generation_model}")
    print(f"ICD rule cache: {settings.icd_rule_cache_path}")
    print(f"Bundle templates: {settings.bundle_template_directory}")
    print("=" * 50)

    # Create pipeline
    pipeline = create_default_clinical_note_quality_pipeline()

    # Run batch generation
    start_time = datetime.now()
    results, metrics = pipeline.run_batch_generation_pipeline(
        requested_example_count=args.count,
        write_artifacts=not args.no_artifacts,
        show_progress=True,
    )
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\nGeneration Complete!")
    print("-" * 50)
    print(f"Total attempted: {metrics.total_attempted}")
    print(f"Accepted: {metrics.total_accepted}")
    print(f"Rejected: {metrics.total_rejected}")
    print(f"Acceptance rate: {metrics.acceptance_rate:.1%}")
    print(f"Revision rate: {metrics.revision_rate:.1%}")
    print(f"Revision success rate: {metrics.revision_success_rate:.1%}")
    print(f"Average combined score: {metrics.average_combined_score_for_accepted_notes:.3f}")
    print(
        f"Average general quality score: {metrics.average_general_quality_score_for_accepted_notes:.3f}"
    )
    if metrics.average_icd_alignment_score_for_accepted_notes is not None:
        print(
            f"Average ICD alignment score: {metrics.average_icd_alignment_score_for_accepted_notes:.3f}"
        )
    else:
        print("Average ICD alignment score: n/a")
    print(f"Deterministic hard-fail rate: {metrics.deterministic_precheck_hard_fail_rate:.1%}")
    print(f"Support verifier fail rate: {metrics.support_verifier_fail_rate:.1%}")
    print(f"ICD leakage rate: {metrics.icd_code_leakage_rate:.1%}")
    print(f"ICD adjudication fail rate: {metrics.icd_adjudication_fail_rate:.1%}")
    print(f"ICD code-set validation fail rate: {metrics.code_set_validation_fail_rate:.1%}")
    top_icd_rule_failures = ", ".join(metrics.top_icd_rule_failure_types[:5])
    print(f"Top ICD rule failure types: {top_icd_rule_failures or 'none'}")
    top_failing_criteria = ", ".join(metrics.most_frequent_failing_rubric_criteria[:5])
    print(f"Top failing rubric criteria: {top_failing_criteria or 'none'}")
    print(f"Duration: {duration:.1f}s")
    print("-" * 50)

    # Print details for each note
    for i, result in enumerate(results, 1):
        if isinstance(result, AcceptedClinicalNoteResult):
            print(f"\nNote {i} [ACCEPTED]")
            print(f"  Archetype: {result.seeded_bundle.archetype}")
            print(f"  Template ID: {result.seeded_bundle.template_id}")
            print(f"  Required revision: {result.required_revision}")
            if result.final_critique.combined_score is not None:
                print(f"  Combined score: {result.final_critique.combined_score:.3f}")
            print(f"  ICD-10 codes: {', '.join(result.adjudicated_icd10_codes)}")
            print(f"  Seeded ICD-10 codes: {', '.join(result.seeded_icd10_codes)}")
            print(f"  Note text preview: {result.accepted_note.note_text[:100]}...")
        else:
            print(f"\nNote {i} [REJECTED]")
            print(f"  Primary rejection reason: {result.primary_rejection_reason}")
            print(f"  All rejection reasons: {', '.join(result.all_rejection_reasons)}")

    # Save results to JSON if requested
    if not args.no_artifacts:
        results_file = (
            output_dir / f"clinical_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        results_data = {
            "generation_params": {
                "count": args.count,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
            },
            "metrics": {
                "total_attempted": metrics.total_attempted,
                "total_accepted": metrics.total_accepted,
                "total_rejected": metrics.total_rejected,
                "acceptance_rate": metrics.acceptance_rate,
                "revision_rate": metrics.revision_rate,
                "icd_adjudication_fail_rate": metrics.icd_adjudication_fail_rate,
                "code_set_validation_fail_rate": metrics.code_set_validation_fail_rate,
                "top_icd_rule_failure_types": metrics.top_icd_rule_failure_types,
            },
            "notes": [],
        }

        for result in results:
            if isinstance(result, AcceptedClinicalNoteResult):
                note_data = {
                    "correlation_id": result.correlation_id,
                    "status": "accepted",
                    "archetype": result.seeded_bundle.archetype,
                    "template_id": result.seeded_bundle.template_id,
                    "required_revision": result.required_revision,
                    "combined_score": result.final_critique.combined_score,
                    "icd10_codes": result.adjudicated_icd10_codes,
                    "seeded_icd10_codes": result.seeded_icd10_codes,
                    "adjudication_status": (
                        result.adjudication_provenance.outcome
                        if result.adjudication_provenance
                        else "not_run"
                    ),
                    "adjudication_rationale": (
                        result.adjudication_provenance.adjudication_rationale
                        if result.adjudication_provenance
                        else ""
                    ),
                    "code_set_validation": (
                        result.code_set_validation_outcome.model_dump(mode="json")
                        if result.code_set_validation_outcome
                        else None
                    ),
                    "note_text": result.accepted_note.note_text,
                    "patient_name": result.accepted_note.fake_patient_name,
                    "patient_mrn": result.accepted_note.fake_patient_mrn,
                    "patient_dob": result.accepted_note.fake_patient_date_of_birth,
                    "pipeline_trace": result.pipeline_trace,
                }
            elif isinstance(result, RejectedClinicalNoteResult):
                note_data = {
                    "correlation_id": result.correlation_id,
                    "status": "rejected",
                    "primary_rejection_reason": result.primary_rejection_reason,
                    "all_rejection_reasons": result.all_rejection_reasons,
                    "icd10_codes": result.adjudicated_icd10_codes,
                    "seeded_icd10_codes": result.seeded_icd10_codes,
                    "adjudication_status": (
                        result.adjudication_provenance.outcome
                        if result.adjudication_provenance
                        else "not_run"
                    ),
                    "code_set_validation": (
                        result.code_set_validation_outcome.model_dump(mode="json")
                        if result.code_set_validation_outcome
                        else None
                    ),
                    "pipeline_trace": result.pipeline_trace,
                }
            else:
                continue
            results_data["notes"].append(note_data)

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    # Print artifact locations
    if not args.no_artifacts:
        print("\nArtifacts written to:")
        print(f"  Training rows: {settings.training_artifact_output_path}")
        print(f"  Audit rows: {settings.audit_artifact_output_path}")
        print(f"  Batch metrics: {settings.batch_metrics_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
