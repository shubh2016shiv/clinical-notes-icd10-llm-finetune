"""
Data Preparation Script
=======================

WHAT THIS SCRIPT DOES:
End-to-end data preparation pipeline: Load → Format → Validate → Split → Save.
Run with: python -m icd10_fine_tune.scripts.prepare_data

EDUCATIONAL NOTE - Data Pipeline Order:
1. LOAD: Parse raw JSON into ClinicalNote objects
2. FORMAT: Convert to ChatML (system/user/assistant messages)
3. VALIDATE: Check ICD-10 codes, token lengths, JSON structure
4. SPLIT: Train/validation stratified by profile
5. SAVE: Write to JSONL for HuggingFace datasets
"""

import sys
from typing import List

from icd10_fine_tune.config.settings import settings
from icd10_fine_tune.data.loader import load_clinical_notes_json, load_and_summarize
from icd10_fine_tune.data.formatter import (
    batch_convert_to_chatml,
    save_chatml_dataset,
    ChatMLSample
)
from icd10_fine_tune.data.format_validator import validate_dataset, filter_valid_samples


def split_dataset(
    samples: List[ChatMLSample],
    val_ratio: float = 0.2,
    seed: int = 42
) -> tuple:
    """
    Split samples into train and validation sets.
    
    EDUCATIONAL NOTE - Stratified Splitting:
    For clinical data with profiles (chronic_care, acute_injury, etc.),
    stratified splitting ensures each split has proportional representation.
    This prevents validation set from being all one profile type.
    """
    import random
    from collections import defaultdict
    
    random.seed(seed)
    
    # Group by profile for stratification
    by_profile = defaultdict(list)
    for sample in samples:
        profile = sample.metadata.get("profile", "unknown")
        by_profile[profile].append(sample)
    
    train_samples = []
    val_samples = []
    
    for profile, profile_samples in by_profile.items():
        random.shuffle(profile_samples)
        split_idx = int(len(profile_samples) * (1 - val_ratio))
        train_samples.extend(profile_samples[:split_idx])
        val_samples.extend(profile_samples[split_idx:])
    
    # Shuffle final lists
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    return train_samples, val_samples


def main():
    """Main data preparation entry point."""
    print("\n" + "="*60)
    print("ICD-10 Fine-Tuning Data Preparation")
    print("="*60 + "\n")
    
    # ====================================================================
    # STEP 1: LOAD RAW DATA
    # ====================================================================
    print("Step 1: Loading raw clinical notes...")
    
    raw_path = settings.raw_data_path
    if not raw_path.exists():
        print(f"ERROR: Raw data not found at {raw_path}")
        print("  Please ensure clinical notes JSON exists.")
        return 1
    
    # Show data summary
    load_and_summarize(raw_path)
    
    # Load full dataset
    dataset = load_clinical_notes_json(raw_path)
    print(f"[OK] Loaded {dataset.total_notes} clinical notes\n")
    
    # ====================================================================
    # STEP 2: FORMAT AS CHATML
    # ====================================================================
    print("Step 2: Converting to ChatML format...")
    
    samples = batch_convert_to_chatml(dataset.notes)
    print(f"[OK] Converted {len(samples)} samples to ChatML\n")
    
    # ====================================================================
    # STEP 3: VALIDATE
    # ====================================================================
    print("Step 3: Validating samples...")
    
    validate_dataset(
        samples,
        max_seq_length=settings.max_seq_length,
        strict=True,
        verbose=True
    )
    
    # Filter invalid samples
    valid_samples = filter_valid_samples(samples, settings.max_seq_length)
    print(f"[OK] {len(valid_samples)} samples passed validation\n")
    
    # ====================================================================
    # STEP 4: SPLIT TRAIN/VALIDATION
    # ====================================================================
    print("Step 4: Splitting into train/validation...")
    
    train_samples, val_samples = split_dataset(
        valid_samples,
        val_ratio=settings.validation_split,
        seed=settings.random_seed
    )
    
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print("[OK] Split complete\n")
    
    # ====================================================================
    # STEP 5: SAVE
    # ====================================================================
    print("Step 5: Saving processed datasets...")
    
    output_dir = settings.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    
    save_chatml_dataset(train_samples, train_path)
    save_chatml_dataset(val_samples, val_path)
    
    # ====================================================================
    # DONE
    # ====================================================================
    print("\n" + "="*60)
    print("Data Preparation Complete!")
    print("="*60)
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    print("\n  Next step: python -m icd10_fine_tune.scripts.train")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
