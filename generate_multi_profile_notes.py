"""
Generate 50 Clinical Notes with 5 Different Profiles

This script generates clinical notes across multiple patient profiles to create
a diverse dataset for fine-tuning. Each profile generates medically appropriate
ICD-10 codes and clinical narratives.

Profiles:
    1. chronic_care: Patients with chronic conditions (diabetes, hypertension, etc.)
    2. acute_injury: Patients with acute injuries and trauma
    3. mental_health: Patients with mental health conditions
    4. pediatric: Pediatric patients with age-appropriate conditions
    5. infectious_disease: Patients with infectious diseases

Usage:
    python generate_multi_profile_notes.py

Author: Shubham Singh
Date: December 2025
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Add clinical_note_generation to path
sys.path.insert(0, str(Path(__file__).parent))

from clinical_note_generation.core.enums import ClinicalNoteType  # noqa: E402
from clinical_note_generation.pipeline import ClinicalNotePipeline  # noqa: E402


def main():
    """Generate 50 notes across 5 profiles (10 notes each)."""

    # Load .env from project root
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from: {env_file}")
    else:
        print(f"Warning: .env file not found at {env_file}")

    # Configure logger for clean output
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("\n" + "=" * 80)
    print("CLINICAL NOTE GENERATION - MULTI-PROFILE BATCH")
    print("=" * 80 + "\n")

    # =========================================================================
    # STAGE 1: INITIALIZE PIPELINE
    # =========================================================================
    print("STAGE 1: Initializing pipeline from environment...")
    try:
        pipeline = ClinicalNotePipeline.from_environment()
        print("[OK] Pipeline initialized successfully")
        print(f"  - LLM Provider: {pipeline.config.llm_provider}")
        print(f"  - Model: {pipeline.config.gemini_model}")
        print(f"  - Dataset: {pipeline.config.icd10_dataset_path}")
        print(f"  - Total ICD-10 codes: {pipeline.repository.total_codes:,}")
        print()
    except Exception as e:
        print(f"[FAIL] Failed to initialize pipeline: {e}")
        return 1

    # =========================================================================
    # STAGE 2: DEFINE PROFILES
    # =========================================================================
    profiles = [
        ("chronic_care", "Chronic Care Patients"),
        ("acute_injury", "Acute Injury Patients"),
        ("mental_health", "Mental Health Patients"),
        ("pediatric", "Pediatric Patients"),
        ("infectious_disease", "Infectious Disease Patients"),
    ]

    notes_per_profile = 10
    total_notes_target = len(profiles) * notes_per_profile

    print("STAGE 2: Generation Plan")
    print(f"  - Total profiles: {len(profiles)}")
    print(f"  - Notes per profile: {notes_per_profile}")
    print(f"  - Total notes target: {total_notes_target}")
    print()

    # =========================================================================
    # STAGE 3: GENERATE NOTES FOR EACH PROFILE
    # =========================================================================
    print("STAGE 3: Generating notes...\n")

    all_notes = []
    profile_stats = {}

    for idx, (profile_id, profile_name) in enumerate(profiles, 1):
        print(f"[{idx}/{len(profiles)}] Generating {notes_per_profile} notes for: {profile_name}")
        print("-" * 80)

        try:
            # Generate notes for this profile
            notes = pipeline.generate_notes(
                count=notes_per_profile,
                profile=profile_id,
                note_type=ClinicalNoteType.PROGRESS_NOTE,
                validate=True,
            )

            # Track statistics
            valid_count = sum(1 for note in notes if note.is_valid)
            profile_stats[profile_id] = {
                "generated": len(notes),
                "valid": valid_count,
                "validation_rate": (valid_count / len(notes) * 100) if notes else 0,
            }

            all_notes.extend(notes)

            print(
                f"[OK] Generated {len(notes)} notes | Valid: {valid_count}/{len(notes)} ({profile_stats[profile_id]['validation_rate']:.1f}%)"
            )

            # Show sample codes from first note
            if notes:
                sample_codes = [f"{code.code}" for code in notes[0].assigned_codes[:3]]
                print(f"  Sample codes: {', '.join(sample_codes)}")

            print()

        except Exception as e:
            print(f"[FAIL] Failed to generate notes for {profile_name}: {e}")
            logger.exception(f"Error generating {profile_id} notes")
            print()
            continue

    # =========================================================================
    # STAGE 4: SAVE ALL NOTES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 4: Saving generated notes...")
    print("-" * 80)

    if all_notes:
        try:
            output_path = pipeline.save_notes(
                notes=all_notes, filename_prefix="multi_profile_clinical_notes"
            )
            print(f"[OK] Saved {len(all_notes)} notes to: {output_path}")
        except Exception as e:
            print(f"[FAIL] Failed to save notes: {e}")
            logger.exception("Error saving notes")
    else:
        print("[FAIL] No notes generated, nothing to save")

    # =========================================================================
    # STAGE 5: SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 5: Generation Summary")
    print("=" * 80 + "\n")

    print("Overall Statistics:")
    print(f"  - Total notes generated: {len(all_notes)}/{total_notes_target}")
    print(f"  - Overall validation rate: {pipeline.validation_rate:.1f}%")
    print()

    print("Per-Profile Breakdown:")
    print("-" * 80)
    print(f"{'Profile':<25} {'Generated':<12} {'Valid':<12} {'Rate':<10}")
    print("-" * 80)

    for profile_id, profile_name in profiles:
        if profile_id in profile_stats:
            stats = profile_stats[profile_id]
            print(
                f"{profile_name:<25} {stats['generated']:<12} {stats['valid']:<12} {stats['validation_rate']:.1f}%"
            )
        else:
            print(f"{profile_name:<25} {'FAILED':<12} {'-':<12} {'-':<10}")

    print("-" * 80)
    print()

    # =========================================================================
    # STAGE 6: VALIDATION INSIGHTS
    # =========================================================================
    if all_notes:
        print("Validation Insights:")
        print("-" * 80)

        # Count validation issues
        notes_with_issues = [n for n in all_notes if not n.is_valid and n.validation_result]
        if notes_with_issues:
            issue_types = {}
            for note in notes_with_issues:
                # Combine critical issues and warnings from validation_result
                all_issues = note.validation_result.critical_issues + note.validation_result.warnings
                for issue in all_issues:
                    issue_types[issue] = issue_types.get(issue, 0) + 1

            print("Common validation issues:")
            for issue, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {issue}: {count} occurrences")
        else:
            print("  [OK] All notes passed validation!")

        print()

    # =========================================================================
    # FINAL STATUS
    # =========================================================================
    print("=" * 80)
    if len(all_notes) >= total_notes_target * 0.8:  # 80% success threshold
        print("[OK] GENERATION SUCCESSFUL")
        print("=" * 80 + "\n")
        return 0
    else:
        print("[WARN] GENERATION COMPLETED WITH WARNINGS")
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
