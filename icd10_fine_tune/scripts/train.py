"""
Main Training Script
====================

WHAT THIS SCRIPT DOES:
End-to-end training orchestration: Load data → Load model → Train → Save adapters.
Run with: python -m icd10_fine_tune.scripts.train

WHY WE NEED THIS:
Single entry point for training that coordinates all layers.
Enables reproducible training runs with configurable parameters.

EDUCATIONAL NOTE - Script Structure:
1. Parse arguments / load config
2. Initialize logging
3. Load and validate data
4. Load model with LoRA adapters
5. Create trainer
6. Run training
7. Save adapters
8. Report results
"""

import sys

from icd10_fine_tune.config.settings import settings
from icd10_fine_tune.config.model_registry import estimate_vram_requirements
from icd10_fine_tune.observability.logger import get_logger


def main():
    """
    Main training entry point.
    
    EDUCATIONAL NOTE - Main Function Pattern:
    Using a main() function instead of top-level code allows:
    1. Easier testing (can import without running)
    2. Clear entry point
    3. Better error handling
    """
    logger = get_logger()
    
    try:
        # ====================================================================
        # STEP 1: CONFIGURATION
        # ====================================================================
        logger.training_start(
            model_variant=settings.model_variant,
            learning_rate=settings.learning_rate,
            epochs=settings.num_train_epochs,
            batch_size=settings.per_device_train_batch_size,
            gradient_accumulation=settings.gradient_accumulation_steps,
            max_seq_length=settings.max_seq_length,
        )
        
        # Check VRAM estimate
        vram_estimate = estimate_vram_requirements(
            settings.model_variant,
            settings.per_device_train_batch_size,
            settings.max_seq_length,
            settings.gradient_accumulation_steps
        )
        
        logger.info(
            f"Estimated VRAM: {vram_estimate['total_estimated_gb']:.2f}GB",
            **vram_estimate
        )
        
        if not vram_estimate["fits_6gb"]:
            logger.warning(
                "Configuration may exceed 6GB VRAM! "
                "Consider reducing max_seq_length or batch_size."
            )
        
        # ====================================================================
        # STEP 2: LOAD DATA
        # ====================================================================
        logger.info("Loading training data...")
        
        from datasets import load_dataset
        
        # Try to load processed data, or fall back to raw
        processed_train = settings.processed_data_dir / "train.jsonl"
        processed_val = settings.processed_data_dir / "validation.jsonl"
        
        if processed_train.exists():
            logger.info(f"Loading from processed: {processed_train}")
            train_dataset = load_dataset("json", data_files=str(processed_train), split="train")
            
            if processed_val.exists():
                eval_dataset = load_dataset("json", data_files=str(processed_val), split="train")
            else:
                eval_dataset = None
                logger.warning("No validation set found")
        else:
            logger.error(
                f"Processed data not found at {processed_train}. "
                f"Run 'python -m icd10_fine_tune.scripts.prepare_data' first."
            )
            return 1
        
        logger.info(f"Loaded {len(train_dataset)} training samples")
        
        # ====================================================================
        # STEP 3: LOAD MODEL
        # ====================================================================
        logger.info("Loading model with 4-bit quantization...")
        
        from icd10_fine_tune.training.model_loader import (
            load_model_for_training,
            prepare_model_for_lora
        )
        
        model, tokenizer = load_model_for_training()
        model = prepare_model_for_lora(model, use_unsloth=settings.use_unsloth)
        
        # ====================================================================
        # STEP 4: CREATE TRAINER
        # ====================================================================
        logger.info("Creating SFTTrainer...")
        
        from icd10_fine_tune.training.trainer import (
            create_sft_trainer,
            train_model,
            save_lora_adapters,
            TrainingConfig
        )
        
        config = TrainingConfig.from_settings()
        
        trainer = create_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
            dataset_text_field="messages"  # ChatML format
        )
        
        # ====================================================================
        # STEP 5: TRAIN
        # ====================================================================
        logger.info("Starting training...")
        
        metrics = train_model(trainer)
        
        logger.training_end(
            final_loss=metrics.get("train_loss"),
            runtime_seconds=metrics.get("train_runtime"),
            samples_per_second=metrics.get("train_samples_per_second")
        )
        
        # ====================================================================
        # STEP 6: SAVE ADAPTERS
        # ====================================================================
        logger.info("Saving LoRA adapters...")
        
        adapter_path = save_lora_adapters(model, tokenizer)
        logger.info(f"Adapters saved to: {adapter_path}")
        
        # ====================================================================
        # STEP 7: DONE
        # ====================================================================
        logger.info("Training complete!")
        print(f"\n✓ Training complete. Adapters saved to: {adapter_path}")
        print("  Run evaluation with: python -m icd10_fine_tune.scripts.evaluate")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main() or 0)
