"""
File Operations Utilities

This module provides utility functions for saving clinical notes to files and
generating summary reports for the clinical note generation process.

Author: RhythmX AI Team
Date: November 2025
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from loguru import logger

from ..core.data_models import ClinicalNote


def save_notes_to_json(
    clinical_notes: List[ClinicalNote],
    output_file_path: str,
    verbose_output: bool = True
) -> str:
    """
    Save clinical notes to a JSON file.
    
    Step 1: Create output directory if it doesn't exist
    Step 2: Convert ClinicalNote objects to dictionaries
    Step 3: Write JSON file with proper formatting
    Step 4: Log save success with file size
    Step 5: Return the file path
    
    Args:
        clinical_notes: List of ClinicalNote objects to save
        output_file_path: Path where JSON file will be saved
        verbose_output: If True, include all fields. If False, exclude non-essential fields
                       (generation_timestamp, validation_status, confidence_score, and
                       most icd10_code_details fields except icd10_code and icd10_name)
        
    Returns:
        Absolute path to the saved file
        
    Raises:
        IOError: If file cannot be written
        
    Example:
        >>> notes = [note1, note2, note3]
        >>> path = save_notes_to_json(notes, "output/notes.json")
        >>> print(path)
        '/path/to/output/notes.json'
    """
    # Step 1: Create output directory if it doesn't exist
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(clinical_notes)} clinical notes to: {output_file_path}")
    
    try:
        # Step 2: Convert ClinicalNote objects to dictionaries
        notes_data = []
        for note in clinical_notes:
            # Build base note dictionary
            note_dict = {
                "note_id": note.note_id,
                "note_type": note.note_type,
                "clinical_content": note.clinical_content,
                "assigned_icd10_codes": note.assigned_icd10_codes,
            }
            
            if verbose_output:
                # Include all fields when verbose
                note_dict["icd10_code_details"] = note.icd10_code_details
                note_dict["generation_timestamp"] = note.generation_timestamp
                note_dict["validation_status"] = note.validation_status
                note_dict["validation_details"] = note.validation_details
            else:
                # Filter icd10_code_details to only include icd10_code and icd10_name
                filtered_code_details = []
                for code_detail in note.icd10_code_details:
                    filtered_code_details.append({
                        "icd10_code": code_detail.get("icd10_code"),
                        "icd10_name": code_detail.get("icd10_name")
                    })
                note_dict["icd10_code_details"] = filtered_code_details
                
                # Filter validation_details to exclude confidence_score
                validation_details = dict(note.validation_details)
                validation_details.pop("confidence_score", None)
                note_dict["validation_details"] = validation_details
                # Note: generation_timestamp and validation_status are excluded
            
            notes_data.append(note_dict)
        
        # Step 3: Write JSON file with proper formatting
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(notes_data, file, indent=2, ensure_ascii=False)
        
        # Step 4: Log save success with file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Successfully saved clinical notes to: {output_file_path} "
            f"({file_size_mb:.2f} MB)"
        )
        
        # Step 5: Return the file path
        return str(output_path.absolute())
        
    except Exception as error:
        error_message = f"Error saving clinical notes to file: {error}"
        logger.error(error_message)
        raise IOError(error_message)


def generate_summary_report(
    clinical_notes: List[ClinicalNote],
    output_file_path: str
) -> str:
    """
    Generate and save a human-readable summary report of generated clinical notes.
    
    Step 1: Create output directory if it doesn't exist
    Step 2: Calculate validation statistics
    Step 3: Build report header
    Step 4: Build validation statistics section
    Step 5: Build individual note details section
    Step 6: Write report to file
    Step 7: Log report save success
    Step 8: Return file path
    
    Args:
        clinical_notes: List of ClinicalNote objects to summarize
        output_file_path: Path where summary report will be saved
        
    Returns:
        Absolute path to the saved report file
        
    Raises:
        IOError: If file cannot be written
        
    Example:
        >>> notes = [note1, note2, note3]
        >>> path = generate_summary_report(notes, "output/summary.txt")
        >>> print(path)
        '/path/to/output/summary.txt'
    """
    # Step 1: Create output directory if it doesn't exist
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating summary report for {len(clinical_notes)} notes")
    
    try:
        # Step 2: Calculate validation statistics
        validation_counts = {}
        total_confidence = 0
        
        for note in clinical_notes:
            status = note.validation_status
            validation_counts[status] = validation_counts.get(status, 0) + 1
            total_confidence += note.validation_details.get('confidence_score', 0)
        
        average_confidence = (
            total_confidence / len(clinical_notes)
            if clinical_notes else 0
        )
        
        # Step 3: Build report header
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CLINICAL NOTES GENERATION SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(
            f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"Total Notes Generated: {len(clinical_notes)}")
        report_lines.append(f"Average Confidence Score: {average_confidence:.1f}%")
        report_lines.append("")
        
        # Step 4: Build validation statistics section
        report_lines.append("VALIDATION STATUS DISTRIBUTION:")
        report_lines.append("-" * 80)
        
        for status in ["VALID", "ACCEPTABLE", "REVIEW_REQUIRED"]:
            count = validation_counts.get(status, 0)
            percentage = (count / len(clinical_notes) * 100) if clinical_notes else 0
            report_lines.append(f"  {status}: {count} ({percentage:.1f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("INDIVIDUAL NOTE DETAILS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Step 5: Build individual note details section
        for index, note in enumerate(clinical_notes, 1):
            report_lines.append(f"Note #{index}: {note.note_id}")
            report_lines.append(f"  Type: {note.note_type}")
            report_lines.append(f"  ICD-10 Codes: {', '.join(note.assigned_icd10_codes)}")
            report_lines.append(f"  Validation Status: {note.validation_status}")
            report_lines.append(
                f"  Confidence Score: "
                f"{note.validation_details.get('confidence_score', 0)}%"
            )
            
            # Add validation notes if available
            validation_notes = note.validation_details.get('validation_notes', 'N/A')
            report_lines.append(f"  Validation Notes: {validation_notes}")
            
            # Add invalid codes if any (handle both string and dict formats)
            invalid_codes = note.validation_details.get('invalid_codes', [])
            if invalid_codes:
                # Convert to strings if they're dictionaries
                invalid_codes_str = []
                for code in invalid_codes:
                    if isinstance(code, dict):
                        # Extract code or reason from dict
                        invalid_codes_str.append(str(code.get('code', code.get('reason', str(code)))))
                    else:
                        invalid_codes_str.append(str(code))
                report_lines.append(f"  Invalid Codes: {', '.join(invalid_codes_str)}")
            
            # Add missing codes if any (handle both string and dict formats)
            missing_codes = note.validation_details.get('missing_codes', [])
            if missing_codes:
                # Convert to strings if they're dictionaries
                missing_codes_str = []
                for code in missing_codes:
                    if isinstance(code, dict):
                        # Extract code or reason from dict
                        missing_codes_str.append(str(code.get('code', code.get('reason', str(code)))))
                    else:
                        missing_codes_str.append(str(code))
                report_lines.append(f"  Missing Codes: {', '.join(missing_codes_str)}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Step 6: Write report to file
        report_content = "\n".join(report_lines)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(report_content)
        
        # Step 7: Log report save success
        logger.info(f"Successfully saved summary report to: {output_file_path}")
        
        # Step 8: Return file path
        return str(output_path.absolute())
        
    except Exception as error:
        error_message = f"Error generating summary report: {error}"
        logger.error(error_message)
        raise IOError(error_message)

