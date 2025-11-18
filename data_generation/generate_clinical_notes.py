"""
Clinical Notes Generator CLI

Command-line interface for generating realistic, de-identified clinical notes
using Google's Gemini API with ICD-10 code validation.

This script provides a CLI for the modular clinical notes generation system,
following the project's patterns for argument parsing and logging.

Usage:
    python data_generation/generate_clinical_notes.py --num-notes 10
    python data_generation/generate_clinical_notes.py --num-notes 50 --min-diagnoses 3 --max-diagnoses 6

Author: RhythmX AI Team
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from data_generation.core.configuration import ClinicalNotesGeneratorConfiguration
from data_generation.core.enums import ClinicalNoteType
from data_generation.generators.clinical_notes_generator import ClinicalNotesGenerator


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.
    
    Step 1: Create ArgumentParser with description
    Step 2: Add num-notes argument
    Step 3: Add min-diagnoses argument
    Step 4: Add max-diagnoses argument
    Step 5: Add note-types argument
    Step 6: Add dataset-path argument
    Step 7: Add output-dir argument
    Step 8: Add log-file argument
    Step 9: Return configured parser
    
    Returns:
        ArgumentParser: Configured argument parser
        
    Example:
        >>> parser = create_argument_parser()
        >>> args = parser.parse_args(['--num-notes', '10'])
    """
    # Step 1: Create ArgumentParser with description
    parser = argparse.ArgumentParser(
        description="Generate realistic clinical notes with ICD-10 code validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate 10 clinical notes (default settings):
    python data_generation/generate_clinical_notes.py --num-notes 10

  Generate 50 notes with 3-6 diagnoses each:
    python data_generation/generate_clinical_notes.py --num-notes 50 --min-diagnoses 3 --max-diagnoses 6

  Generate notes with specific types only:
    python data_generation/generate_clinical_notes.py --num-notes 20 --note-types "PROGRESS_NOTE,DISCHARGE_SUMMARY"

  Generate with OpenAI GPT-4o:
    python data_generation/generate_clinical_notes.py --provider openai --num-notes 10

  Generate with custom output directory:
    python data_generation/generate_clinical_notes.py --num-notes 10 --output-dir ./my_notes

Available Note Types:
  - EMERGENCY_DEPARTMENT
  - INPATIENT_ADMISSION
  - PROGRESS_NOTE
  - DISCHARGE_SUMMARY
  - CONSULTATION_NOTE
  - OPERATIVE_REPORT
  - HISTORY_AND_PHYSICAL
  - OUTPATIENT_VISIT

Requirements:
  - Create a .env file in data_generation/ with your API key(s)
  - For Gemini: GEMINI_API_KEY from https://makersuite.google.com/app/apikey
  - For OpenAI: OPENAI_API_KEY from https://platform.openai.com/api-keys
  - See .env.example for template
        """
    )
    
    # Step 2: Add num-notes argument
    parser.add_argument(
        "--num-notes",
        type=int,
        default=10,
        help="Number of clinical notes to generate (default: 10)"
    )
    
    # Step 3: Add min-diagnoses argument
    parser.add_argument(
        "--min-diagnoses",
        type=int,
        default=2,
        help="Minimum number of diagnoses per note (default: 2)"
    )
    
    # Step 4: Add max-diagnoses argument
    parser.add_argument(
        "--max-diagnoses",
        type=int,
        default=5,
        help="Maximum number of diagnoses per note (default: 5)"
    )
    
    # Step 5: Add note-types argument
    parser.add_argument(
        "--note-types",
        type=str,
        default="ALL",
        help=(
            "Comma-separated list of note types to generate (default: ALL). "
            "Example: PROGRESS_NOTE,DISCHARGE_SUMMARY"
        )
    )

    # Step 5.1: Add provider argument
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "openai"],
        default="gemini",
        help="API provider to use (default: gemini). Options: gemini, openai"
    )

    # Step 6: Add dataset-path argument
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data_generation/icd10_dataset.json",
        help="Path to ICD-10 dataset JSON file (default: data_generation/icd10_dataset.json)"
    )
    
    # Step 7: Add output-dir argument
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_generation/generated_clinical_notes",
        help="Output directory for generated notes (default: data_generation/generated_clinical_notes)"
    )
    
    # Step 8: Add log-file argument
    parser.add_argument(
        "--log-file",
        type=str,
        default="data_generation/clinical_notes_generation.log",
        help="Log file path (default: data_generation/clinical_notes_generation.log)"
    )
    
    # Step 9: Return configured parser
    return parser


def parse_note_types(note_types_string: str) -> List[ClinicalNoteType]:
    """
    Parse note types string into list of ClinicalNoteType enums.
    
    Step 1: Handle "ALL" case
    Step 2: Split comma-separated string
    Step 3: Convert strings to enums
    Step 4: Validate each type
    Step 5: Return list of enums
    
    Args:
        note_types_string: Comma-separated string of note types or "ALL"
        
    Returns:
        List of ClinicalNoteType enums
        
    Raises:
        ValueError: If an invalid note type is specified
        
    Example:
        >>> types = parse_note_types("PROGRESS_NOTE,DISCHARGE_SUMMARY")
        >>> print(len(types))
        2
    """
    # Step 1: Handle "ALL" case
    if note_types_string.upper() == "ALL":
        return list(ClinicalNoteType)
    
    # Step 2: Split comma-separated string
    type_strings = [t.strip().upper() for t in note_types_string.split(',')]
    
    # Step 3: Convert strings to enums
    parsed_types = []
    
    for type_string in type_strings:
        # Step 4: Validate each type
        try:
            note_type = ClinicalNoteType[type_string]
            parsed_types.append(note_type)
        except KeyError:
            available_types = [t.name for t in ClinicalNoteType]
            error_message = (
                f"Invalid note type: '{type_string}'. "
                f"Available types: {', '.join(available_types)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
    
    # Step 5: Return list of enums
    return parsed_types


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.
    
    Step 1: Validate num-notes is positive
    Step 2: Validate min-diagnoses is positive
    Step 3: Validate max-diagnoses is >= min-diagnoses
    Step 4: Validate dataset file exists
    Step 5: Log validation success
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        ValueError: If any argument is invalid
        
    Example:
        >>> args = parser.parse_args(['--num-notes', '10'])
        >>> validate_arguments(args)
    """
    # Step 1: Validate num-notes is positive
    if args.num_notes <= 0:
        raise ValueError(f"num-notes must be positive, got: {args.num_notes}")
    
    # Step 2: Validate min-diagnoses is positive
    if args.min_diagnoses <= 0:
        raise ValueError(f"min-diagnoses must be positive, got: {args.min_diagnoses}")
    
    # Step 3: Validate max-diagnoses is >= min-diagnoses
    if args.max_diagnoses < args.min_diagnoses:
        raise ValueError(
            f"max-diagnoses ({args.max_diagnoses}) must be >= "
            f"min-diagnoses ({args.min_diagnoses})"
        )
    
    # Step 4: Validate dataset file exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"ICD-10 dataset file not found: {args.dataset_path}. "
            f"Please ensure the dataset file exists at the specified path."
        )
    
    # Step 5: Log validation success
    logger.debug("Command-line arguments validated successfully")


def main() -> None:
    """
    Main function to run the clinical notes generation CLI.
    
    Step 1: Parse command-line arguments
    Step 2: Validate arguments
    Step 3: Parse note types
    Step 4: Load configuration from environment
    Step 5: Override configuration with CLI arguments
    Step 6: Initialize generator
    Step 7: Generate clinical notes
    Step 8: Save generated notes
    Step 9: Display completion summary
    
    Raises:
        SystemExit: If critical error occurs
    """
    # Step 1: Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("CLINICAL NOTES GENERATOR WITH ICD-10 VALIDATION")
    print("=" * 80)
    print()
    
    try:
        # Step 2: Validate arguments
        validate_arguments(args)
        
        # Step 3: Parse note types
        note_types = parse_note_types(args.note_types)
        
        logger.info(f"Generating {args.num_notes} clinical notes")
        logger.info(f"API Provider: {args.provider}")
        logger.info(f"Note types: {[t.name for t in note_types]}")
        logger.info(f"Diagnoses per note: {args.min_diagnoses}-{args.max_diagnoses}")
        logger.info(f"Dataset: {args.dataset_path}")
        logger.info(f"Output directory: {args.output_dir}")

        # Step 4: Load configuration from environment
        logger.info("\nLoading configuration from environment...")

        # Set API provider environment variable based on CLI argument
        import os
        os.environ["API_PROVIDER"] = args.provider.upper()

        configuration = ClinicalNotesGeneratorConfiguration.load_from_environment()

        # Step 5: Override configuration with CLI arguments
        configuration.icd10_dataset_file_path = args.dataset_path
        configuration.output_directory = args.output_dir
        configuration.log_file_path = args.log_file
        
        # Step 6: Initialize generator
        logger.info("\nInitializing clinical notes generator...")
        generator = ClinicalNotesGenerator(configuration)
        
        # Step 7: Generate clinical notes
        logger.info("\nStarting clinical notes generation...")
        generated_notes = generator.generate_multiple_clinical_notes(
            total_notes_count=args.num_notes,
            note_types=note_types,
            min_diagnoses=args.min_diagnoses,
            max_diagnoses=args.max_diagnoses
        )
        
        # Step 8: Save generated notes
        logger.info("\nSaving generated notes...")
        output_path = generator.save_generated_notes(generated_notes)
        
        # Step 9: Display completion summary
        print()
        print("=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"✓ Generated {len(generated_notes)} clinical notes")
        
        if output_path:
            print(f"✓ Output saved to: {output_path}")
        else:
            print("⚠ No output file created (no notes were successfully generated)")
            print("⚠ Check the logs above for error details (likely API quota limit)")
        
        print(f"✓ Log file: {args.log_file}")
        
        # Display validation statistics only if notes were generated
        if generated_notes:
            validation_counts = {}
            for note in generated_notes:
                status = note.validation_status
                validation_counts[status] = validation_counts.get(status, 0) + 1
            
            print("\nValidation Status Distribution:")
            for status, count in validation_counts.items():
                percentage = (count / len(generated_notes) * 100) if generated_notes else 0
                print(f"  {status}: {count} ({percentage:.1f}%)")
            
            print("=" * 80)
            print("\n✓ Ready for use in ICD-10 fine-tuning!")
        else:
            print("=" * 80)
            print("\n⚠ No notes were generated. Common causes:")
            print("  - API quota limit exceeded (check your Gemini API plan)")
            print("  - Network connectivity issues")
            print("  - Invalid API key")
            print("\nPlease check the error messages above and try again.")
        
        print()
        
    except ValueError as error:
        logger.error(f"Invalid argument: {error}")
        sys.exit(1)
    
    except FileNotFoundError as error:
        logger.error(f"File not found: {error}")
        sys.exit(1)
    
    except Exception as error:
        logger.error(f"Unexpected error: {error}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()

