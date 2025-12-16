"""
Clinical Notes Data Loader
===========================

WHAT THIS MODULE DOES:
Loads raw clinical notes from JSON files and performs initial schema validation.
This is the entry point for the data pipeline.

WHY WE NEED THIS:
1. **Separation of Concerns**: Data loading logic is isolated from preprocessing
   and formatting. This makes it easy to support different input formats in the
   future (CSV, database, etc.) by just swapping this module.

2. **Early Validation**: We validate the data schema immediately after loading
   to catch problems early (missing fields, wrong types) before expensive
   preprocessing begins.

3. **Type Safety**: Returns strongly-typed data structures that subsequent layers
   can rely on.

HOW IT WORKS:
1. Read JSON file from disk
2. Validate schema (each note has required fields: text, ICD-10 codes, metadata)
3. Convert to Pydantic models for type-safe access
4. Return structured data ready for preprocessing
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# DATA MODELS
# ============================================================================
# These Pydantic models define the expected structure of clinical notes.
# Using Pydantic gives us automatic validation and type safety.

class ClinicalNote(BaseModel):
    """
    Represents a single clinical note with associated ICD-10 codes.
    
    EDUCATIONAL NOTE - Data Structure Design:
    We use Pydantic models instead of plain dictionaries because:
    1. Type safety: Fields are validated automatically
    2. Documentation: Field descriptions document the data schema
    3. IDE support: Auto-completion and type checking
    4. Validation: Custom validators catch malformed data early
    """
    
    note_id: str = Field(..., description="Unique identifier for this clinical note")
    
    clinical_text: str = Field(..., description="The raw clinical note text (narrative)")
    
    icd10_codes: List[str] = Field(..., description="List of ICD-10 codes assigned to this note")
    
    profile: str = Field(default="unknown", description="Clinical profile/category (e.g., 'chronic_care', 'acute_injury')")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (patient demographics, encounter info)")
    
    @field_validator("clinical_text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """
        Ensure clinical text is not empty or whitespace-only.
        
        WHY THIS VALIDATION:
        Empty notes cannot be fine-tuned on (nothing to learn from).
        Catching this early prevents wasted computation later.
        """
        if not v or not v.strip():
            raise ValueError("clinical_text cannot be empty")
        return v
    
    @field_validator("icd10_codes")
    @classmethod
    def validate_codes_not_empty(cls, v: List[str]) -> List[str]:
        """
        Ensure at least one ICD-10 code is present.
        
        WHY THIS VALIDATION:
        Notes without codes have no target for supervised learning.
        This is a fundamental requirement for classification fine-tuning.
        """
        if not v or len(v) == 0:
            raise ValueError("At least one ICD-10 code is required")
        return v


class ClinicalDataset(BaseModel):
    """
    Collection of clinical notes with dataset-level metadata.
    
    This represents the entire dataset loaded from a file.
    """
    
    notes: List[ClinicalNote] = Field(..., description="List of clinical notes")
    
    dataset_name: str = Field(default="clinical_notes", description="Name of this dataset")
    
    total_notes: int = Field(default=0, description="Total number of notes in dataset")
    
    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to compute derived fields.
        
        EDUCATIONAL NOTE - Pydantic Lifecycle:
        model_post_init runs after validation but before the object is returned.
        We use it to compute fields that depend on other fields (e.g., total_notes).
        """
        if self.total_notes == 0:
            self.total_notes = len(self.notes)


# ============================================================================
# LOADER FUNCTIONS
# ============================================================================

def load_clinical_notes_json(file_path: Path) -> ClinicalDataset:
    """
    Load clinical notes from a JSON file.
    
    Args:
        file_path: Path to JSON file containing clinical notes
    
    Returns:
        ClinicalDataset object with validated notes
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If data doesn't match expected schema
    
    EDUCATIONAL NOTE - Error Handling Strategy:
    We let exceptions propagate rather than catching and logging them.
    This is intentional: data loading errors should STOP the pipeline
    immediately, not silently fail. The caller can decide how to handle errors.
    
    JSON Format Expected:
    Either:
    1. Array of note objects: [{note_id, clinical_text, icd10_codes}, ...]
    2. Object with 'notes' key: {notes: [...], metadata: {...}}
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Clinical notes file not found: {file_path}. "
            "Please ensure the data file exists before running the pipeline."
        )
    
    # Read raw JSON
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(raw_data, list):
        # Format 1: Direct array of notes
        notes_data = raw_data
        dataset_name = file_path.stem
    elif isinstance(raw_data, dict) and "notes" in raw_data:
        # Format 2: Object with 'notes' key
        notes_data = raw_data["notes"]
        dataset_name = raw_data.get("dataset_name", file_path.stem)
    else:
        raise ValueError(
            f"Unexpected JSON structure in {file_path}. "
            "Expected either a list of notes or an object with 'notes' key."
        )
    
    # Parse and validate each note
    # EDUCATIONAL NOTE - Why validate individually?
    # If we validate all at once and one note fails, we lose all data.
    # By validating individually, we can report which specific note failed
    # and potentially skip it while keeping the rest.
    notes = []
    errors = []
    
    for idx, note_data in enumerate(notes_data):
        try:
            # Normalize field names (handle different conventions)
            normalized = _normalize_note_fields(note_data, idx)
            note = ClinicalNote(**normalized)
            notes.append(note)
        except Exception as e:
            errors.append(f"Note {idx}: {str(e)}")
    
    # Report validation errors if any
    if errors:
        error_summary = "\n".join(errors[:10])  # Show first 10 errors
        if len(errors) > 10:
            error_summary += f"\n... and {len(errors) - 10} more errors"
        
        raise ValueError(
            f"Failed to validate {len(errors)} out of {len(notes_data)} notes:\n"
            f"{error_summary}"
        )
    
    return ClinicalDataset(
        notes=notes,
        dataset_name=dataset_name,
        total_notes=len(notes)
    )


def _normalize_note_fields(note_data: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize field names to match ClinicalNote schema.
    
    This handles different naming conventions in source data:
    - 'text' vs 'clinical_text' vs 'note_text'
    - 'codes' vs 'icd10_codes' vs 'icd_codes'
    - 'id' vs 'note_id'
    
    Args:
        note_data: Raw note dictionary from JSON
        index: Note index (used to generate ID if missing)
    
    Returns:
        Normalized dictionary matching ClinicalNote schema
    
    EDUCATIONAL NOTE - Defensive Programming:
    Real-world data is messy. By handling multiple field name variations,
    we make the system robust to different data sources without requiring
    manual preprocessing.
    """
    normalized = {}
    
    # Note ID
    normalized["note_id"] = (
        note_data.get("note_id") or
        note_data.get("id") or
        f"note_{index:06d}"
    )
    
    # Clinical text
    normalized["clinical_text"] = (
        note_data.get("clinical_text") or
        note_data.get("text") or
        note_data.get("note_text") or
        note_data.get("narrative") or
        ""
    )
    
    # ICD-10 codes
    normalized["icd10_codes"] = (
        note_data.get("icd10_codes") or
        note_data.get("assigned_icd10_codes") or  # Handle generated data format
        note_data.get("codes") or
        note_data.get("icd_codes") or
        []
    )
    
    # Profile
    normalized["profile"] = note_data.get("profile", "unknown")
    
    # Metadata (everything else)
    excluded_keys = {"note_id", "id", "clinical_text", "text", "note_text",
                     "narrative", "icd10_codes", "assigned_icd10_codes", "codes", "icd_codes", "profile"}
    normalized["metadata"] = {
        k: v for k, v in note_data.items() if k not in excluded_keys
    }
    
    return normalized


def load_and_summarize(file_path: Path) -> None:
    """
    Load dataset and print summary statistics.
    
    This is a utility function for quick exploratory data analysis.
    
    EDUCATIONAL NOTE - EDA Before Fine-Tuning:
    Always inspect your data before fine-tuning. Key questions:
    1. How many examples do we have? (< 100: probably not enough)
    2. How are codes distributed? (Highly imbalanced: need class weights)
    3. How long are notes? (Too long: increase max_seq_length or truncate)
    4. Are profiles balanced? (Helps ensure model sees diverse cases)
    """
    dataset = load_clinical_notes_json(file_path)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.dataset_name}")
    print(f"{'='*60}")
    print(f"Total notes: {dataset.total_notes}")
    
    # Profile distribution
    from collections import Counter
    profile_counts = Counter(note.profile for note in dataset.notes)
    print("\nProfile distribution:")
    for profile, count in profile_counts.most_common():
        print(f"  {profile}: {count} notes ({count/dataset.total_notes*100:.1f}%)")
    
    # Code statistics
    all_codes = [code for note in dataset.notes for code in note.icd10_codes]
    unique_codes = set(all_codes)
    print("\nICD-10 code statistics:")
    print(f"  Total code assignments: {len(all_codes)}")
    print(f"  Unique codes: {len(unique_codes)}")
    print(f"  Avg codes per note: {len(all_codes)/dataset.total_notes:.2f}")
    
    #Text length statistics
    text_lengths = [len(note.clinical_text) for note in dataset.notes]
    print("\nText length statistics (characters):")
    print(f"  Min: {min(text_lengths)}")
    print(f"  Max: {max(text_lengths)}")
    print(f"  Mean: {sum(text_lengths)/len(text_lengths):.0f}")
    print(f"  Median: {sorted(text_lengths)[len(text_lengths)//2]}")
    
    print(f"{'='*60}\n")
