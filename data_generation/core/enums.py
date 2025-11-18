"""
Enumerations for Clinical Notes Generation

This module contains all enumeration types used in clinical notes generation,
including different types of clinical documentation.

Author: RhythmX AI Team
Date: November 2025
"""

from enum import Enum


class APIProvider(Enum):
    """
    Enumeration of supported API providers for clinical note generation.

    Attributes:
        GEMINI: Google's Gemini API
        OPENAI: OpenAI's GPT models
    """
    GEMINI = "gemini"
    OPENAI = "openai"


class ClinicalNoteType(Enum):
    """
    Enumeration of different clinical note types used in medical documentation.
    
    Each note type represents a specific type of medical documentation with
    distinct purposes, content requirements, and use cases in clinical practice.
    
    Attributes:
        EMERGENCY_DEPARTMENT: Documentation of emergency room visits
        INPATIENT_ADMISSION: Initial admission notes for hospitalized patients
        PROGRESS_NOTE: Daily updates on patient status during hospital stay
        DISCHARGE_SUMMARY: Comprehensive summary at hospital discharge
        CONSULTATION_NOTE: Specialist consultation documentation
        OPERATIVE_REPORT: Surgical procedure documentation
        HISTORY_AND_PHYSICAL: Complete patient history and physical exam
        OUTPATIENT_VISIT: Routine outpatient clinic visit documentation
    
    Example:
        >>> note_type = ClinicalNoteType.EMERGENCY_DEPARTMENT
        >>> print(note_type.value)
        'Emergency Department Visit'
    """
    
    EMERGENCY_DEPARTMENT = "Emergency Department Visit"
    INPATIENT_ADMISSION = "Inpatient Admission Note"
    PROGRESS_NOTE = "Progress Note"
    DISCHARGE_SUMMARY = "Discharge Summary"
    CONSULTATION_NOTE = "Consultation Note"
    OPERATIVE_REPORT = "Operative Report"
    HISTORY_AND_PHYSICAL = "History and Physical Examination"
    OUTPATIENT_VISIT = "Outpatient Visit Note"

