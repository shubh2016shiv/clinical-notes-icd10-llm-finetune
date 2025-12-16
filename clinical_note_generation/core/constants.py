"""
Constants for Clinical Note Generation

This module defines constant values used throughout the clinical note
generation pipeline. Constants are:
    1. Centralized for easy modification
    2. Type-hinted for IDE support
    3. Documented with usage context

Constant Categories:
    GENERATION_PROFILES   → Patient profile configurations
    CODE_CATEGORIES       → ICD-10 code category mappings
    VALIDATION_PATTERNS   → Regex patterns for validation

Author: Shubham Singh
Date: December 2025
"""

from typing import Dict, List, Any


# =============================================================================
# STAGE 1: GENERATION PROFILES
# =============================================================================
# Define patient profile configurations for themed code selection.
# Each profile specifies primary conditions and likely comorbidities.

GENERATION_PROFILES: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # 1.1 Common Clinical Profiles
    # -------------------------------------------------------------------------
    "default": {
        "description": "Mixed profile with varied code selection across all categories.",
        "primary_search_terms": [],  # No specific focus
        "comorbidity_ratio": 0.0,
        "comorbidity_terms": [],
    },
    "chronic_care": {
        "description": "Focus on common chronic conditions with related comorbidities.",
        "primary_search_terms": [
            "diabetes",
            "hypertension",
            "copd",
            "heart disease",
            "asthma",
            "arthritis",
            "obesity",
            "chronic kidney",
        ],
        "comorbidity_ratio": 0.75,
        "comorbidity_terms": ["cardiology", "renal", "neurology", "endocrine", "obesity"],
    },
    "acute_injury": {
        "description": "Focus on acute injuries, fractures, and trauma.",
        "primary_search_terms": [
            "fracture",
            "injury",
            "trauma",
            "wound",
            "burn",
            "dislocation",
            "sprain",
            "contusion",
            "laceration",
        ],
        "comorbidity_ratio": 0.25,
        "comorbidity_terms": ["pain", "complication", "infection"],
    },
    "mental_health": {
        "description": "Focus on psychiatric and mental health diagnoses.",
        "primary_search_terms": [
            "depression",
            "anxiety",
            "bipolar",
            "schizophrenia",
            "ptsd",
            "substance abuse",
            "mood disorder",
            "panic",
        ],
        "comorbidity_ratio": 0.60,
        "comorbidity_terms": ["sleep disorder", "chronic pain", "substance"],
    },
    "infectious_disease": {
        "description": "Focus on infectious diseases and complications.",
        "primary_search_terms": [
            "pneumonia",
            "influenza",
            "infection",
            "sepsis",
            "covid",
            "tuberculosis",
            "cellulitis",
            "urinary tract",
        ],
        "comorbidity_ratio": 0.50,
        "comorbidity_terms": ["respiratory", "fever", "immunocompromised"],
    },
    # -------------------------------------------------------------------------
    # 1.2 Specialty Profiles
    # -------------------------------------------------------------------------
    "cardiology": {
        "description": "Focus on cardiovascular diseases.",
        "primary_search_terms": [
            "heart failure",
            "coronary",
            "arrhythmia",
            "hypertension",
            "myocardial infarction",
            "atrial fibrillation",
            "chest pain",
        ],
        "comorbidity_ratio": 0.80,
        "comorbidity_terms": ["diabetes", "renal", "stroke", "peripheral vascular"],
    },
    "oncology": {
        "description": "Focus on cancer diagnoses and treatment.",
        "primary_search_terms": [
            "cancer",
            "neoplasm",
            "carcinoma",
            "lymphoma",
            "leukemia",
            "metastasis",
            "malignancy",
        ],
        "comorbidity_ratio": 0.70,
        "comorbidity_terms": ["pain", "anemia", "cachexia", "complication"],
    },
    "pediatric": {
        "description": "Focus on pediatric-specific conditions.",
        "primary_search_terms": [
            "asthma",
            "otitis media",
            "strep throat",
            "gastroenteritis",
            "bronchiolitis",
            "fever",
            "viral infection",
        ],
        "comorbidity_ratio": 0.30,
        "comorbidity_terms": ["developmental disorder", "allergy", "fever"],
    },
    "geriatric": {
        "description": "Focus on age-related conditions.",
        "primary_search_terms": [
            "dementia",
            "osteoporosis",
            "fall",
            "urinary incontinence",
            "delirium",
            "frailty",
            "alzheimer",
        ],
        "comorbidity_ratio": 0.85,
        "comorbidity_terms": ["hypertension", "diabetes", "heart disease", "arthritis", "renal"],
    },
}


# =============================================================================
# STAGE 2: ICD-10 CODE CATEGORY MAPPING
# =============================================================================
# Maps ICD-10 first character to diagnosis category.

ICD10_CHAPTER_MAP: Dict[str, str] = {
    "A": "Infectious and Parasitic Diseases",
    "B": "Infectious and Parasitic Diseases",
    "C": "Neoplasms",
    "D": "Neoplasms and Blood Diseases",
    "E": "Endocrine, Nutritional and Metabolic Diseases",
    "F": "Mental and Behavioral Disorders",
    "G": "Diseases of the Nervous System",
    "H": "Diseases of the Eye and Ear",
    "I": "Diseases of the Circulatory System",
    "J": "Diseases of the Respiratory System",
    "K": "Diseases of the Digestive System",
    "L": "Diseases of the Skin",
    "M": "Diseases of the Musculoskeletal System",
    "N": "Diseases of the Genitourinary System",
    "O": "Pregnancy, Childbirth and Puerperium",
    "P": "Conditions Originating in Perinatal Period",
    "Q": "Congenital Malformations",
    "R": "Symptoms, Signs and Abnormal Findings",
    "S": "Injury, Poisoning (Anatomical)",
    "T": "Injury, Poisoning (Type)",
    "V": "External Causes - Transport",
    "W": "External Causes - Falls/Exposure",
    "X": "External Causes - Other",
    "Y": "External Causes - Medical/Surgical",
    "Z": "Factors Influencing Health Status",
}


# =============================================================================
# STAGE 3: VALIDATION PATTERNS
# =============================================================================
# Regex patterns for clinical note validation.

NEGATION_PATTERNS: List[str] = [
    r"no\s+",
    r"not\s+",
    r"denies\s+",
    r"without\s+",
    r"negative\s+for\s+",
    r"ruled\s+out",
    r"unlikely\s+",
    r"absence\s+of\s+",
    r"no\s+evidence\s+of\s+",
    r"no\s+sign\s+of\s+",
]

UNCERTAINTY_PATTERNS: List[str] = [
    r"possible\s+",
    r"probable\s+",
    r"suspected\s+",
    r"rule\s+out\s+",
    r"differential\s+includes\s+",
    r"cannot\s+exclude\s+",
    r"may\s+have\s+",
    r"could\s+be\s+",
    r"questionable\s+",
]

# Patterns that should NOT appear in clinical note text (data leakage)
ICD10_CODE_PATTERN = r"[A-Z]\d{2}\.?\d{0,4}[A-Z]?"


# =============================================================================
# STAGE 4: LLM PROMPT TEMPLATES
# =============================================================================
# Template strings for LLM prompts (keys for lookup, not full templates).

PROMPT_TEMPLATE_KEYS = {
    "generation": "clinical_note_generation",
    "validation": "icd10_validation",
    "code_ranking": "code_ranking",
}


# =============================================================================
# STAGE 5: QUALITY THRESHOLDS
# =============================================================================
# Thresholds for quality metrics and validation.

QUALITY_THRESHOLDS = {
    "min_note_length": 100,  # characters
    "max_note_length": 10000,  # characters
    "min_confidence": 70.0,  # 0-100
    "critical_issue_threshold": 1,  # max critical issues to accept
    "warning_threshold": 3,  # max warnings before concern
}


# =============================================================================
# STAGE 6: LOGGING CONFIGURATION
# =============================================================================
# Standard log format for consistent logging across modules.

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
