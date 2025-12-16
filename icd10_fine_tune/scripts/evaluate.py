"""
Evaluation Script
=================

WHAT THIS SCRIPT DOES:
Evaluates a trained model on validation data and generates reports.
Run with: python -m icd10_fine_tune.scripts.evaluate

STEPS:
1. Load trained model with LoRA adapters
2. Run inference on validation set
3. Compute classification metrics
4. Push to Evidently Cloud
5. Generate local HTML report
"""

import sys

from icd10_fine_tune.config.settings import settings
from icd10_fine_tune.observability.logger import get_logger


def main():
    """Main evaluation entry point."""
    logger = get_logger()
    
    try:
        logger.info("Starting evaluation...")
        
        # ====================================================================
        # STEP 1: LOAD VALIDATION DATA
        # ====================================================================
        from datasets import load_dataset
        import json
        
        val_path = settings.processed_data_dir / "validation.jsonl"
        
        if not val_path.exists():
            logger.error(f"Validation data not found: {val_path}")
            return 1
        
        val_dataset = load_dataset("json", data_files=str(val_path), split="train")
        logger.info(f"Loaded {len(val_dataset)} validation samples")
        
        # Extract texts and ground truth codes
        input_texts = []
        ground_truth = []
        
        for sample in val_dataset:
            messages = sample["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            assistant_msg = next(m for m in messages if m["role"] == "assistant")
            
            input_texts.append(user_msg["content"])
            
            codes_data = json.loads(assistant_msg["content"])
            ground_truth.append(codes_data.get("icd10_codes", []))
        
        # ====================================================================
        # STEP 2: LOAD MODEL
        # ====================================================================
        logger.info("Loading model with LoRA adapters...")
        
        adapter_path = settings.output_dir / "lora_adapters"
        
        if not adapter_path.exists():
            logger.error(f"Adapters not found: {adapter_path}")
            logger.info("Run training first: python -m icd10_fine_tune.scripts.train")
            return 1
        
        from icd10_fine_tune.training.model_loader import load_model_for_training
        from peft import PeftModel
        
        base_model, tokenizer = load_model_for_training()
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # ====================================================================
        # STEP 3: RUN INFERENCE
        # ====================================================================
        logger.info("Running inference...")
        
        from icd10_fine_tune.evaluation.inference import (
            run_inference,
            parse_icd10_from_output
        )
        
        raw_outputs = run_inference(model, tokenizer, input_texts)
        predictions = [parse_icd10_from_output(out) for out in raw_outputs]
        
        # ====================================================================
        # STEP 4: COMPUTE METRICS & REPORT
        # ====================================================================
        logger.info("Computing metrics...")
        
        from icd10_fine_tune.evaluation.evidently_reporter import EvidentlyReporter
        
        reporter = EvidentlyReporter()
        
        # Push to cloud and get metrics
        metrics = reporter.log_evaluation(
            predictions=predictions,
            ground_truth=ground_truth,
            metadata={
                "model_variant": settings.model_variant,
                "adapter_path": str(adapter_path)
            }
        )
        
        # Generate local report
        report_path = reporter.generate_local_report(
            predictions=predictions,
            ground_truth=ground_truth,
            input_texts=input_texts
        )
        
        # Save metrics JSON
        json_path = reporter.save_metrics_json(metrics)
        
        # ====================================================================
        # DONE
        # ====================================================================
        print("\n✓ Evaluation complete!")
        print(f"  F1 Micro: {metrics['f1_micro']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  Exact Match: {metrics['exact_match_accuracy']:.4f}")
        print(f"\n  HTML Report: {report_path}")
        print(f"  JSON Metrics: {json_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main() or 0)
