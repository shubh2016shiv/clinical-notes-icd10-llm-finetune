"""
Clinical Notes Generator Orchestrator

This module provides the main orchestrator class that coordinates clinical note
generation, validation, and saving operations.

Author: RhythmX AI Team
Date: November 2025
"""

import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from ..clients.gemini_api_client import GeminiAPIClient
from ..clients.openai_api_client import OpenAIAPIClient
from ..core.configuration import ClinicalNotesGeneratorConfiguration
from ..core.data_models import ClinicalNote
from ..core.enums import APIProvider, ClinicalNoteType
from ..managers.icd10_dataset_manager import ICD10DatasetManager
from ..utils.file_operations import save_notes_to_json, generate_summary_report


class ClinicalNotesGenerator:
    """
    Main orchestrator class for clinical note generation and validation.
    
    This class coordinates all components of the clinical notes generation system
    including the ICD-10 dataset manager, Gemini API client, and file operations.
    It manages the complete workflow from note generation through validation to
    file export.
    
    Attributes:
        configuration: Configuration object with all settings
        icd10_dataset_manager: Manager for ICD-10 code dataset
        api_client: Client for API interactions (Gemini or OpenAI)
    
    Example:
        >>> config = ClinicalNotesGeneratorConfiguration.load_from_environment()
        >>> generator = ClinicalNotesGenerator(config)
        >>> notes = generator.generate_multiple_clinical_notes(total_notes_count=10)
    """
    
    def __init__(self, configuration: ClinicalNotesGeneratorConfiguration):
        """
        Initialize the clinical notes generator orchestrator.
        
        Step 1: Store configuration
        Step 2: Validate configuration
        Step 3: Initialize ICD-10 dataset manager
        Step 4: Initialize Gemini API client
        Step 5: Log orchestrator initialization
        
        Args:
            configuration: Configuration object with all settings
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If ICD-10 dataset file not found
            
        Example:
            >>> config = ClinicalNotesGeneratorConfiguration.load_from_environment()
            >>> generator = ClinicalNotesGenerator(config)
        """
        # Step 1: Store configuration
        self.configuration = configuration
        
        # Step 2: Validate configuration
        self.configuration.validate()
        
        logger.info("=" * 80)
        logger.info("CLINICAL NOTES GENERATOR ORCHESTRATOR INITIALIZING")
        logger.info("=" * 80)
        
        # Step 3: Initialize ICD-10 dataset manager
        logger.info("Initializing ICD-10 dataset manager...")
        self.icd10_dataset_manager = ICD10DatasetManager(
            dataset_file_path=configuration.icd10_dataset_file_path
        )
        
        # Step 4: Initialize API client based on provider
        if configuration.api_provider == APIProvider.GEMINI:
            logger.info("Initializing Gemini API client...")
            self.api_client = GeminiAPIClient(
                api_key=configuration.gemini_api_key,
                model_name=configuration.gemini_model_name,
                temperature=configuration.generation_temperature,
                top_p=configuration.generation_top_p,
                top_k=configuration.generation_top_k,
                max_output_tokens=configuration.max_output_tokens
            )
        elif configuration.api_provider == APIProvider.OPENAI:
            logger.info("Initializing OpenAI API client...")
            self.api_client = OpenAIAPIClient(
                api_key=configuration.openai_api_key,
                model_name=configuration.openai_model_name,
                temperature=configuration.generation_temperature,
                top_p=configuration.generation_top_p,
                max_tokens=configuration.max_output_tokens
            )
        
        # Step 5: Log orchestrator initialization
        logger.info("=" * 80)
        logger.success("CLINICAL NOTES GENERATOR ORCHESTRATOR INITIALIZED")
        logger.info("=" * 80)
    
    def generate_single_clinical_note(
        self,
        note_type: ClinicalNoteType,
        number_of_diagnoses: int = 3
    ) -> Optional[ClinicalNote]:
        """
        Generate a single clinical note with validation and retry logic.
        
        Step 1: Log note generation start
        Step 2: Select random billable ICD-10 codes
        Step 3: Generate clinical note using API client
        Step 4: Validate the generated note with retry logic
        Step 4.1: Perform initial validation
        Step 4.2: If confidence < threshold, retry up to max_retries times
        Step 4.3: Keep the best result (highest confidence >= threshold)
        Step 4.4: If none meet threshold, return None (note skipped)
        Step 5: Determine validation status
        Step 6: Create ClinicalNote object
        Step 7: Log generation completion
        Step 8: Return clinical note or None if skipped
        
        Args:
            note_type: Type of clinical note to generate
            number_of_diagnoses: Number of ICD-10 codes to assign
            
        Returns:
            ClinicalNote object with validation results if confidence >= threshold,
            None if all validation attempts failed to meet threshold
            
        Raises:
            ValueError: If no billable codes available
            Exception: If generation fails
            
        Example:
            >>> note = generator.generate_single_clinical_note(
            ...     note_type=ClinicalNoteType.PROGRESS_NOTE,
            ...     number_of_diagnoses=3
            ... )
        """
        # Step 1: Log note generation start
        logger.info("-" * 80)
        logger.info(f"Generating {note_type.value}...")
        
        # Step 2: Select random billable ICD-10 codes
        selected_icd10_codes = self.icd10_dataset_manager.get_random_billable_codes(
            count=number_of_diagnoses
        )
        
        if not selected_icd10_codes:
            error_message = "No billable ICD-10 codes available in dataset"
            logger.error(error_message)
            raise ValueError(error_message)
        
        logger.info(
            f"Selected ICD-10 codes: "
            f"{[code.icd10_code for code in selected_icd10_codes]}"
        )
        
        # Step 3: Generate clinical note using API client
        logger.info("Generating clinical note content...")
        clinical_note_text = self.api_client.generate_clinical_note(
            note_type=note_type,
            target_icd10_codes=selected_icd10_codes
        )
        
        logger.info(
            f"Generated note content ({len(clinical_note_text)} characters)"
        )
        
        # Step 4: Validate the generated note with retry logic
        logger.info("Validating ICD-10 code assignments...")
        assigned_code_strings = [code.icd10_code for code in selected_icd10_codes]
        
        # Step 4.1: Perform initial validation
        best_validation_response = None
        best_confidence = 0.0
        threshold = self.configuration.validation_confidence_threshold
        max_retries = self.configuration.validation_max_retries
        total_attempts = max_retries + 1  # Initial attempt + retries
        
        for attempt in range(total_attempts):
            if attempt > 0:
                logger.info(
                    f"Retry attempt {attempt}/{max_retries} "
                    f"(confidence {best_confidence:.1f}% < threshold {threshold}%)"
                )
            
            validation_response = self.api_client.validate_icd10_assignment(
                clinical_note=clinical_note_text,
                assigned_icd10_codes=assigned_code_strings,
                icd10_reference_data=selected_icd10_codes
            )
            
            confidence = validation_response.get('confidence_score', 0.0)
            
            # Step 4.2 & 4.3: Keep track of best result
            if confidence > best_confidence:
                best_confidence = confidence
                best_validation_response = validation_response
            
            logger.info(
                f"Attempt {attempt + 1}/{total_attempts} - "
                f"Confidence Score: {confidence:.1f}%"
            )
            
            # If we meet threshold, we can stop early
            if confidence >= threshold:
                logger.success(
                    f"Validation passed threshold ({confidence:.1f}% >= {threshold}%)"
                )
                break
        
        # Step 4.4: Check if we have a result that meets threshold
        if best_confidence < threshold:
            logger.warning(
                f"All validation attempts failed to meet threshold. "
                f"Best confidence: {best_confidence:.1f}% < {threshold}%. "
                f"Skipping this note."
            )
            logger.info("-" * 80)
            return None
        
        # Step 5: Determine validation status using best result
        validation_response = best_validation_response
        invalid_count = len(validation_response.get('invalid_codes', []))
        confidence = validation_response.get('confidence_score', 0)
        
        if invalid_count == 0 and confidence >= 80:
            validation_status = "VALID"
        elif confidence >= 60:
            validation_status = "ACCEPTABLE"
        else:
            validation_status = "REVIEW_REQUIRED"
        
        logger.info(f"Validation Status: {validation_status}")
        logger.info(f"Final Confidence Score: {confidence:.1f}%")
        
        # Step 6: Create ClinicalNote object
        note_id = (
            f"NOTE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{random.randint(1000, 9999)}"
        )
        
        clinical_note = ClinicalNote(
            note_id=note_id,
            note_type=note_type.value,
            clinical_content=clinical_note_text,
            assigned_icd10_codes=assigned_code_strings,
            icd10_code_details=[asdict(code) for code in selected_icd10_codes],
            generation_timestamp=datetime.now().isoformat(),
            validation_status=validation_status,
            validation_details=validation_response
        )
        
        # Step 7: Log generation completion
        logger.success(f"Successfully generated note: {note_id}")
        logger.info("-" * 80)
        
        # Step 8: Return clinical note
        return clinical_note
    
    def generate_multiple_clinical_notes(
        self,
        total_notes_count: int = 10,
        note_types: Optional[List[ClinicalNoteType]] = None,
        min_diagnoses: int = 2,
        max_diagnoses: int = 5
    ) -> List[ClinicalNote]:
        """
        Generate multiple clinical notes with various types and diagnoses.
        
        Step 1: Initialize note types if not provided
        Step 2: Log batch generation start
        Step 3: Generate each note in loop
        Step 3.1: Select random note type
        Step 3.2: Select random number of diagnoses
        Step 3.3: Generate note
        Step 3.4: Handle errors gracefully
        Step 4: Log batch generation summary
        Step 5: Return generated notes
        
        Args:
            total_notes_count: Number of clinical notes to generate
            note_types: List of note types to use (if None, uses all types)
            min_diagnoses: Minimum number of diagnoses per note
            max_diagnoses: Maximum number of diagnoses per note
            
        Returns:
            List of generated ClinicalNote objects
            
        Example:
            >>> notes = generator.generate_multiple_clinical_notes(
            ...     total_notes_count=10,
            ...     min_diagnoses=2,
            ...     max_diagnoses=5
            ... )
        """
        # Step 1: Initialize note types if not provided
        if note_types is None:
            note_types = list(ClinicalNoteType)
        
        # Step 2: Log batch generation start
        logger.info("=" * 80)
        logger.info(f"GENERATING {total_notes_count} CLINICAL NOTES")
        logger.info("=" * 80)
        logger.info(f"Note Types: {len(note_types)} types available")
        logger.info(f"Diagnoses per note: {min_diagnoses}-{max_diagnoses}")
        logger.info("=" * 80)
        
        generated_notes = []
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Step 3: Generate each note in loop
        for note_index in range(total_notes_count):
            logger.info(f"\n[{note_index + 1}/{total_notes_count}] ", )
            
            try:
                # Step 3.1: Select random note type
                selected_note_type = random.choice(note_types)
                
                # Step 3.2: Select random number of diagnoses
                diagnoses_count = random.randint(min_diagnoses, max_diagnoses)
                
                # Step 3.3: Generate note (may return None if validation fails)
                clinical_note = self.generate_single_clinical_note(
                    note_type=selected_note_type,
                    number_of_diagnoses=diagnoses_count
                )
                
                # Step 3.4: Handle skipped notes (None return)
                if clinical_note is None:
                    skipped_count += 1
                    logger.warning(
                        f"Note [{note_index + 1}] skipped due to low validation confidence"
                    )
                    continue
                
                generated_notes.append(clinical_note)
                successful_count += 1
                
            except Exception as error:
                # Step 3.5: Handle errors gracefully
                failed_count += 1
                logger.error(f"ERROR generating note [{note_index + 1}]: {error}")
                continue
        
        # Step 4: Log batch generation summary
        logger.info("=" * 80)
        logger.info("BATCH GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.success(f"Successfully generated: {successful_count}/{total_notes_count} notes")
        if skipped_count > 0:
            logger.warning(
                f"Skipped (low confidence): {skipped_count}/{total_notes_count} notes"
            )
        if failed_count > 0:
            logger.error(f"Failed to generate: {failed_count}/{total_notes_count} notes")
        logger.info("=" * 80)
        
        # Step 5: Return generated notes
        return generated_notes
    
    def save_generated_notes(
        self,
        clinical_notes: List[ClinicalNote],
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Save generated clinical notes to JSON file with summary report.
        
        Step 1: Clean up previous files in output directory
        Step 2: Generate output filename if not provided
        Step 3: Build full output path
        Step 4: Save notes to JSON file
        Step 5: Generate and save summary report
        Step 6: Log save completion
        Step 7: Return output path
        
        Args:
            clinical_notes: List of ClinicalNote objects to save
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to the saved JSON file, or None if no notes were provided
            
        Raises:
            IOError: If files cannot be written
            
        Example:
            >>> path = generator.save_generated_notes(notes)
            >>> print(path)
            'data_generation/generated_clinical_notes/clinical_notes_20250101_120000.json'
        """
        # Step 1: Clean up previous files in output directory
        output_directory = self.configuration.output_directory
        output_dir_path = Path(output_directory)
        
        # Create directory if it doesn't exist
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Delete all existing files in the output directory
        deleted_count = 0
        if output_dir_path.exists():
            for existing_file in output_dir_path.iterdir():
                if existing_file.is_file():
                    try:
                        existing_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted previous file: {existing_file.name}")
                    except Exception as error:
                        logger.warning(
                            f"Could not delete previous file {existing_file.name}: {error}"
                        )
        
        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} previous file(s) from output directory"
            )
        
        # Step 2: Generate output filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"clinical_notes_{timestamp}.json"
        
        # Step 3: Build full output path
        output_path = f"{output_directory}/{output_filename}"
        
        # Check if there are any notes to save
        if not clinical_notes:
            logger.warning("=" * 80)
            logger.warning("NO CLINICAL NOTES TO SAVE")
            logger.warning("=" * 80)
            logger.warning(
                "No clinical notes were successfully generated. "
                "This may be due to API quota limits, network errors, or other issues. "
                "Please check the logs above for error details."
            )
            logger.warning("=" * 80)
            return None
        
        logger.info("=" * 80)
        logger.info("SAVING GENERATED CLINICAL NOTES")
        logger.info("=" * 80)
        
        # Step 3: Save notes to JSON file
        saved_path = save_notes_to_json(
            clinical_notes=clinical_notes,
            output_file_path=output_path
        )
        
        # Step 4: Generate and save summary report
        summary_filename = output_filename.replace('.json', '_summary.txt')
        summary_path = f"{output_directory}/{summary_filename}"
        
        generate_summary_report(
            clinical_notes=clinical_notes,
            output_file_path=summary_path
        )
        
        # Step 5: Log save completion
        logger.info("=" * 80)
        logger.success("CLINICAL NOTES SAVED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"JSON File: {saved_path}")
        logger.info(f"Summary Report: {summary_path}")
        logger.info("=" * 80)
        
        # Step 6: Return output path
        return saved_path

