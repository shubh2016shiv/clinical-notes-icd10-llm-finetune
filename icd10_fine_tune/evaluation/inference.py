"""
Batch Inference Engine
======================

WHAT THIS MODULE DOES:
Runs inference on validation/test datasets and extracts ICD-10 codes from
model outputs. Handles batch processing, output parsing, and error recovery.

WHY WE NEED THIS:
1. **Evaluation Pipeline**: Need to generate predictions before computing metrics.

2. **Production Testing**: Before deploying, test model on held-out data.

3. **Output Parsing**: Model outputs JSON-formatted codes; we parse and validate them.

HOW IT WORKS:
1. Load model with LoRA adapters
2. Process samples in batches (to manage VRAM)
3. Generate text completions for each input
4. Parse JSON from model output
5. Extract ICD-10 codes, handle malformed outputs gracefully
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple

from icd10_fine_tune.config.settings import settings


def run_inference(
    model: Any,
    tokenizer: Any,
    input_texts: List[str],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 128,
    batch_size: int = 1
) -> List[str]:
    """
    Run inference to generate ICD-10 code predictions.
    
    Args:
        model: Model with LoRA adapters loaded
        tokenizer: Tokenizer matching the model
        input_texts: List of clinical note texts
        system_prompt: System prompt to prepend
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
    
    Returns:
        List of model output strings (raw, not parsed)
    
    EDUCATIONAL NOTE - Inference vs Training:
    During inference:
    - No gradient computation (torch.no_grad())
    - No dropout (model.eval())
    - Can use larger batch sizes (no gradients = less VRAM)
    - Temperature sampling or greedy decoding
    
    For ICD-10 coding (deterministic task):
    - temperature=0 or do_sample=False for consistent outputs
    - We want the MOST likely codes, not creative variation
    """
    import torch
    
    if system_prompt is None:
        system_prompt = "You are a medical coding assistant. Analyze clinical notes and assign accurate ICD-10 codes. Always respond with a JSON object containing an array of ICD-10 codes."
    
    model.eval()  # Set to evaluation mode
    outputs = []
    
    print(f"\nRunning inference on {len(input_texts)} samples...")
    
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        
        for text in batch:
            # Format as chat messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this clinical note:\n\n{text}"}
            ]
            
            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # Add assistant turn marker
            )
            
            # Tokenize
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=settings.max_seq_length
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding for consistency
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode only the new tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(output_text)
        
        # Progress update
        if (i + batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(input_texts))}/{len(input_texts)}")
    
    print("✓ Inference complete")
    return outputs


def parse_icd10_from_output(output: str) -> List[str]:
    """
    Extract ICD-10 codes from model output.
    
    Args:
        output: Raw model output string
    
    Returns:
        List of extracted ICD-10 codes
    
    EDUCATIONAL NOTE - Robust Parsing:
    Models don't always produce perfect JSON. We try multiple strategies:
    1. Standard JSON parsing (ideal case)
    2. Regex extraction (handles extra text before/after JSON)
    3. Code-pattern extraction (handles free-text outputs)
    
    This multi-strategy approach makes the system robust to
    slight variations in model output format.
    """
    codes = []
    
    # Strategy 1: Try direct JSON parsing
    try:
        data = json.loads(output.strip())
        if isinstance(data, dict) and "icd10_codes" in data:
            return data["icd10_codes"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from text (handle preamble/postamble)
    json_match = re.search(r'\{[^{}]*"icd10_codes"\s*:\s*\[[^\]]*\][^{}]*\}', output)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("icd10_codes", [])
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Extract ICD-10 codes directly using regex
    # Pattern: Letter followed by 2 digits, optionally dot and more digits
    icd_pattern = r'\b([A-Z][0-9]{2}(?:\.[0-9]{1,4})?)\b'
    codes = re.findall(icd_pattern, output)
    
    return list(set(codes))  # Remove duplicates


def evaluate_predictions(
    predictions: List[List[str]],
    ground_truth: List[List[str]]
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Compare predictions to ground truth.
    
    Args:
        predictions: List of predicted code lists
        ground_truth: List of ground truth code lists
    
    Returns:
        Tuple of (summary_metrics, per_sample_results)
    
    EDUCATIONAL NOTE - Per-Sample Analysis:
    In addition to aggregate metrics, we track per-sample results for:
    1. Error analysis (which samples are hardest?)
    2. Debugging (what patterns does model struggle with?)
    3. Data quality issues (are some ground truth labels wrong?)
    """
    per_sample = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        pred_set = set(pred)
        truth_set = set(truth)
        
        tp = len(pred_set & truth_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_sample.append({
            "index": i,
            "predicted": list(pred_set),
            "ground_truth": list(truth_set),
            "true_positives": list(pred_set & truth_set),
            "false_positives": list(pred_set - truth_set),
            "false_negatives": list(truth_set - pred_set),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": pred_set == truth_set
        })
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Aggregate metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    exact_matches = sum(1 for r in per_sample if r["exact_match"])
    
    summary = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "exact_match_accuracy": exact_matches / len(predictions) if predictions else 0,
        "total_samples": len(predictions)
    }
    
    return summary, per_sample
