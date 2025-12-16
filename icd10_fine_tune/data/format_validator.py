"""
ChatML Format Validator
=======================

WHAT THIS MODULE DOES:
Validates ChatML-formatted training samples to ensure they meet all requirements
for successful fine-tuning. This is a critical quality gate before training begins.

WHY WE NEED THIS:
1. **Prevent Training Failures**: Invalid data causes cryptic errors hours into training.
   Better to catch issues upfront.

2. **Data Quality**: Ensures ICD-10 codes are valid, text length is appropriate,
   and JSON is well-formed.

3. **Token Budget Enforcement**: With 6GB VRAM and max_seq_length=512, samples
   that are too long will cause OOM errors. We need to catch these early.

HOW IT WORKS:
1. Validate ChatML structure (3 messages with correct roles)
2. Validate assistant response is parseable JSON
3. Validate ICD-10 codes match expected format
4. Count tokens and reject samples exceeding max_seq_length
5. Report detailed validation errors for debugging

EDUCATIONAL NOTE - Pre-Training Validation:
Many fine-tuning failures stem from bad data that passed unnoticed.
Spending extra time on validation saves hours of debugging later.
"""

import json
import re
from typing import List, Dict, Any, Tuple

from icd10_fine_tune.data.formatter import ChatMLSample


# ICD-10 CODE VALIDATION
# ============================================================================
# ICD-10 codes follow a specific format: Letter + 2 digits + optional dot + more digits
# Examples: E11.9, I10, Z79.4, J45.901
#
# EDUCATIONAL NOTE - ICD-10 Format:
# - First character: Letter (A-Z), represents category
# - Next 2 characters: Numbers (00-99), represents specific condition
# - Optional: Dot followed by 1-4 more digits for specificity
# - Valid: E11.9, I10, J45.901, E119 (EHR systems often omit dots)
# - Invalid: E1, 123.4 (no letter)
#
# NOTE: We allow codes without dots (e.g., "E119") because many EHR systems
# store codes this way. The dot is formatting, not part of the code itself.

ICD10_PATTERN = re.compile(r"^[A-Z][0-9]{2}\.?[0-9]{0,4}$")

def validate_icd10_code(code: str) -> bool:
    """
    Validate that a string matches ICD-10 format.
    
    Args:
        code: ICD-10 code string
    
    Returns:
        True if valid, False otherwise
    
    EDUCATIONAL NOTE - Regex vs Lookup Table:
    We use regex for format validation but you could also:
    1. Check against official ICD-10 code list (more accurate)
    2. Use a medical terminology API (slower but authoritative)
    
    Regex is fast and catches obvious errors (typos, wrong format).
    For production, consider adding a code existence check.
    """
    if not isinstance(code, str):
        return False
    return ICD10_PATTERN.match(code.strip()) is not None


def validate_icd10_codes(codes: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate a list of ICD-10 codes.
    
    Args:
        codes: List of ICD-10 code strings
    
    Returns:
        Tuple of (all_valid, list of invalid codes)
    """
    invalid_codes = [code for code in codes if not validate_icd10_code(code)]
    return (len(invalid_codes) == 0, invalid_codes)


# ============================================================================
# CHATML STRUCTURE VALIDATION
# ============================================================================

def validate_chatml_structure(sample: ChatMLSample) -> Tuple[bool, str]:
    """
    Validate that a ChatML sample has the correct structure.
    
    Args:
        sample: ChatMLSample to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    
    EDUCATIONAL NOTE - What Makes Valid ChatML:
    1. Exactly 3 messages (system, user, assistant)
    2. Messages in correct order (system -> user -> assistant)
    3. Each message has non-empty content
    4. Roles are valid strings
    
    Why strict validation?
    - Trainers expect this exact format
    - Wrong order confuses the model about context
    - Empty messages waste tokens
    """
    messages = sample.messages
    
    # Check message count
    if len(messages) != 3:
        return False, f"Expected 3 messages, got {len(messages)}"
    
    # Check roles in correct order
    expected_roles = ["system", "user", "assistant"]
    actual_roles = [msg.role for msg in messages]
    
    if actual_roles != expected_roles:
        return False, f"Expected roles {expected_roles}, got {actual_roles}"
    
    # Check all messages have content
    for i, msg in enumerate(messages):
        if not msg.content or not msg.content.strip():
            return False, f"Message {i} ({msg.role}) has empty content"
    
    return True, ""


def validate_assistant_json(sample: ChatMLSample) -> Tuple[bool, str]:
    """
    Validate that assistant message contains valid JSON with ICD-10 codes.
    
    Args:
        sample: ChatMLSample to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    
    EDUCATIONAL NOTE - Why Validate JSON:
    The model learns to generate JSON by seeing it in training data.
    If training data has malformed JSON, the model will learn to generate
    malformed JSON. Garbage in, garbage out.
    """
    assistant_msg = sample.messages[2]  # Assistant is always 3rd message
    
    try:
        # Parse JSON
        data = json.loads(assistant_msg.content)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    
    # Check for icd10_codes key
    if "icd10_codes" not in data:
        return False, "JSON missing 'icd10_codes' key"
    
    # Check codes is a list
    codes = data["icd10_codes"]
    if not isinstance(codes, list):
        return False, f"'icd10_codes' must be a list, got {type(codes).__name__}"
    
    # Check at least one code
    if len(codes) == 0:
        return False, "Empty icd10_codes list"
    
    # Validate each code
    is_valid, invalid_codes = validate_icd10_codes(codes)
    if not is_valid:
        return False, f"Invalid ICD-10 codes: {invalid_codes}"
    
    return True, ""


def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count for a text string.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token (4 is conservative for English)
    
    Returns:
        Estimated token count
    
    EDUCATIONAL NOTE - Token Estimation:
    Accurate token counting requires the actual tokenizer, which is slow.
    For validation, we use a fast heuristic: ~4 characters per token.
    
    This is conservative (slightly overestimates tokens) which is safer
    than underestimating and hitting OOM during training.
    
    For production, you could use:
    - transformers.AutoTokenizer for exact counts
    - tiktoken for fast approximation (OpenAI's tokenizer library)
    """
    return int(len(text) / chars_per_token) + 1  # +1 for safety


def validate_token_length(
    sample: ChatMLSample,
    max_seq_length: int,
    tokenizer_multiplier: float = 1.2
) -> Tuple[bool, str]:
    """
    Validate that sample doesn't exceed max sequence length.
    
    Args:
        sample: ChatMLSample to validate
        max_seq_length: Maximum allowed sequence length
        tokenizer_multiplier: Safety multiplier for tokenizer overhead
    
    Returns:
        Tuple of (is_valid, error_message)
    
    EDUCATIONAL NOTE - Why Token Length Matters:
    1. Longer sequences need more VRAM (quadratic in transformer attention)
    2. Exceeding max_seq_length causes silent truncation or OOM
    3. Very long sequences slow down training significantly
    
    On 6GB VRAM:
    - max_seq_length=512: Safe
    - max_seq_length=1024: Risky (might OOM)
    - max_seq_length=2048: Will definitely OOM
    
    The tokenizer adds special tokens ([SEP], [CLS], etc.) and separators,
    so we multiply by 1.2 for safety margin.
    """
    # Concatenate all message content
    full_text = "\n".join(msg.content for msg in sample.messages)
    
    # Estimate tokens
    estimated_tokens = estimate_token_count(full_text)
    
    # Apply safety multiplier for tokenizer overhead
    estimated_with_overhead = int(estimated_tokens * tokenizer_multiplier)
    
    if estimated_with_overhead > max_seq_length:
        return False, (
            f"Estimated {estimated_with_overhead} tokens exceeds "
            f"max_seq_length={max_seq_length}"
        )
    
    return True, ""


# ============================================================================
# COMPREHENSIVE VALIDATION
# ============================================================================

def validate_sample(
    sample: ChatMLSample,
    max_seq_length: int = 512,
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Run all validation checks on a ChatML sample.
    
    Args:
        sample: ChatMLSample to validate
        max_seq_length: Maximum sequence length allowed
        strict: If True, any validation failure rejects the sample
    
    Returns:
        Tuple of (is_valid, list of error messages)
    
    EDUCATIONAL NOTE - Strict vs Lenient Validation:
    - Strict (recommended): Reject any sample with issues
    - Lenient: Log warnings but keep sample
    
    For fine-tuning, strict is better. One bad sample can corrupt training.
    For inference/evaluation, lenient might be acceptable.
    """
    errors = []
    
    # Validate structure
    valid, error = validate_chatml_structure(sample)
    if not valid:
        errors.append(f"Structure: {error}")
        if strict:
            return False, errors  # Stop early if structure is wrong
    
    # Validate assistant JSON
    valid, error = validate_assistant_json(sample)
    if not valid:
        errors.append(f"Assistant JSON: {error}")
    
    # Validate token length
    valid, error = validate_token_length(sample, max_seq_length)
    if not valid:
        errors.append(f"Token length: {error}")
    
    # Final decision
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_dataset(
    samples: List[ChatMLSample],
    max_seq_length: int = 512,
    strict: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate an entire dataset of ChatML samples.
    
    Args:
        samples: List of ChatMLSample objects
        max_seq_length: Maximum sequence length
        strict: Strict validation mode
        verbose: Print detailed error report
    
    Returns:
        Dictionary with validation statistics
    
    EDUCATIONAL NOTE - Dataset-Level Validation:
    Beyond individual samples, check:
    1. Do we have enough data? (< 100 samples is usually too few)
    2. Are samples diverse? (Same note repeated won't help)
    3. Is token distribution reasonable? (Most samples near max_seq_length is a red flag)
    """
    results = {
        "total_samples": len(samples),
        "valid_samples": 0,
        "invalid_samples": 0,
        "errors": {}
    }
    
    invalid_samples = []
    
    for i, sample in enumerate(samples):
        is_valid, errors = validate_sample(sample, max_seq_length, strict)
        
        if is_valid:
            results["valid_samples"] += 1
        else:
            results["invalid_samples"] += 1
            invalid_samples.append((i, sample.note_id, errors))
            results["errors"][sample.note_id] = errors
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("Dataset Validation Results")
        print(f"{'='*60}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Valid samples: {results['valid_samples']} ({results['valid_samples']/results['total_samples']*100:.1f}%)")
        print(f"Invalid samples: {results['invalid_samples']}")
        
        if invalid_samples:
            print("\nFirst 10 validation errors:")
            for i, (idx, note_id, errors) in enumerate(invalid_samples[:10]):
                print(f"\n  Sample {idx} (note_id: {note_id}):")
                for error in errors:
                    print(f"    - {error}")
            
            if len(invalid_samples) > 10:
                print(f"\n  ... and {len(invalid_samples) - 10} more invalid samples")
        
        print(f"{'='*60}\n")
    
    return results


def filter_valid_samples(
    samples: List[ChatMLSample],
    max_seq_length: int = 512
) -> List[ChatMLSample]:
    """
    Filter dataset to keep only valid samples.
    
    Args:
        samples: List of ChatMLSample objects
        max_seq_length: Maximum sequence length
    
    Returns:
        List of valid ChatMLSample objects
    
    EDUCATIONAL NOTE - To Filter or Fix?
    When you find invalid samples, you have two options:
    1. Filter them out (what this function does)
    2. Fix them (e.g., truncate long texts, correct malformed JSON)
    
    Filtering is simpler but loses data. Fixing is better if you have
    limited data, but requires careful logic to avoid introducing new bugs.
    
    For ICD-10 coding with generated data, filtering is usually fine
    since we can generate more samples if needed.
    """
    valid_samples = []
    
    for sample in samples:
        is_valid, _ = validate_sample(sample, max_seq_length, strict=True)
        if is_valid:
            valid_samples.append(sample)
    
    print(f"Filtered {len(samples)} samples -> {len(valid_samples)} valid samples")
    print(f"Rejected: {len(samples) - len(valid_samples)} samples")
    
    return valid_samples
