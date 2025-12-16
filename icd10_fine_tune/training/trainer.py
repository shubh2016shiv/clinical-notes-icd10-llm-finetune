"""
Training Orchestration
======================

WHAT THIS MODULE DOES:
Orchestrates the complete training pipeline using HuggingFace's SFTTrainer.
Handles multi-GPU setup (DDP), training arguments, and epoch/step management.

WHY WE NEED THIS:
1. **Abstraction**: Hides complex trainer configuration behind clean API.
   Callers don't need to understand TrainingArguments intricacies.

2. **DDP-Ready**: Supports multi-GPU training for portfolio demonstration.
   Single-GPU is default, but architecture supports scaling.

3. **Observability**: Integrates with metrics tracking and logging.

HOW IT WORKS:
1. Configure TrainingArguments (batch size, learning rate, etc.)
2. Create SFTTrainer with model, tokenizer, and dataset
3. Run training with automatic gradient accumulation
4. Save LoRA adapters (not full model) to reduce storage

EDUCATIONAL NOTE - SFTTrainer (Supervised Fine-Tuning):
SFTTrainer is specialized for instruction-following fine-tuning:
- Automatically handles chat templates (ChatML, Llama, etc.)
- Masks non-assistant tokens in loss calculation
- Supports packing (multiple short examples per batch)
- Integrates with PEFT (LoRA) seamlessly
"""

from typing import Optional, Any, Dict
from pathlib import Path
from dataclasses import dataclass

from icd10_fine_tune.config.settings import settings


@dataclass
class TrainingConfig:
    """
    Configuration container for training run.
    
    EDUCATIONAL NOTE - Why Dataclass?:
    Using a dataclass instead of passing many arguments:
    1. Groups related parameters together
    2. Provides type hints and defaults
    3. Easy to serialize/log the full config
    4. Immutable snapshot of training settings
    """
    
    output_dir: Path
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_seq_length: int
    logging_steps: int
    save_steps: int
    save_total_limit: int
    optim: str
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    seed: int
    
    # Multi-GPU settings
    ddp_enabled: bool = False
    local_rank: int = -1
    
    @classmethod
    def from_settings(cls) -> "TrainingConfig":
        """Create TrainingConfig from global settings."""
        output_dir = settings.output_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return cls(
            output_dir=output_dir,
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            learning_rate=settings.learning_rate,
            warmup_ratio=settings.warmup_ratio,
            lr_scheduler_type=settings.lr_scheduler_type,
            max_seq_length=settings.max_seq_length,
            logging_steps=settings.logging_steps,
            save_steps=settings.save_steps,
            save_total_limit=settings.save_total_limit,
            optim=settings.optim,
            fp16=settings.bnb_4bit_compute_dtype == "float16",
            bf16=settings.bnb_4bit_compute_dtype == "bfloat16",
            gradient_checkpointing=settings.gradient_checkpointing,
            seed=settings.random_seed,
            ddp_enabled=settings.use_ddp,
        )


def create_training_arguments(config: TrainingConfig) -> Any:
    """
    Create HuggingFace TrainingArguments from our config.
    
    Args:
        config: TrainingConfig dataclass
    
    Returns:
        transformers.TrainingArguments object
    
    EDUCATIONAL NOTE - Key Training Arguments:
    
    1. Batch Size & Accumulation:
       - per_device_train_batch_size=1: Actual examples per forward pass
       - gradient_accumulation_steps=4: Accumulate before weight update
       - Effective batch size = 1 * 4 = 4 (without extra VRAM)
    
    2. Learning Rate:
       - 2e-4 is typical for LoRA fine-tuning
       - warmup_ratio=0.03: Gradually increase LR for first 3% of steps
       - cosine scheduler: LR decays smoothly to near-zero
    
    3. Memory Optimization:
       - gradient_checkpointing: Recompute activations during backprop
       - optim="paged_adamw_8bit": 8-bit optimizer states
       - fp16=True: Mixed precision training
    
    4. Checkpointing:
       - save_steps: Save every N steps (not epochs for long training)
       - save_total_limit: Keep only recent checkpoints (disk space)
    """
    from transformers import TrainingArguments
    
    return TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        
        # Additional optimizations
        group_by_length=True,  # Group similar-length sequences (faster)
        report_to="none",  # Disable wandb/tensorboard (using our own tracking)
        
        # DDP settings (for multi-GPU)
        ddp_find_unused_parameters=False if config.ddp_enabled else None,
        
        # Disable unnecessary features for memory
        remove_unused_columns=True,
        dataloader_num_workers=0,  # Windows compatibility
    )


def create_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    config: Optional[TrainingConfig] = None,
    dataset_text_field: str = "text",  # Kept for API compatibility but not used
    max_seq_length: Optional[int] = None  # Kept for API compatibility but not used
) -> Any:
    """
    Create an SFTTrainer for instruction fine-tuning.
    
    Args:
        model: Model with LoRA adapters (from prepare_model_for_lora)
        tokenizer: Tokenizer matching the model
        train_dataset: HuggingFace Dataset with training examples
        eval_dataset: Optional validation dataset
        config: TrainingConfig (defaults to from_settings())
        dataset_text_field: DEPRECATED - kept for API compatibility
        max_seq_length: DEPRECATED - kept for API compatibility
    
    Returns:
        trl.SFTTrainer ready to train
    
    EDUCATIONAL NOTE - trl 0.26.1 SFTTrainer API:
    
    The modern SFTTrainer (trl 0.26.1+) has a simplified API:
    - Automatically detects and handles "messages" format datasets
    - No need for dataset_text_field parameter
    - No need for max_seq_length (controlled via tokenizer)
    - No need for packing parameter (handled internally)
    - Uses processing_class instead of tokenizer parameter
    
    When your dataset has a "messages" field with ChatML format:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
    
    SFTTrainer automatically:
    1. Applies the tokenizer's chat_template
    2. Tokenizes the conversation
    3. Masks non-assistant tokens in loss calculation
    
    This is much simpler than older versions!
    """
    from trl import SFTTrainer
    
    if config is None:
        config = TrainingConfig.from_settings()
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    print(f"\n{'='*60}")
    print("Creating SFTTrainer")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset) if eval_dataset else 'None'}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"{'='*60}\n")
    
    # EDUCATIONAL NOTE - trl 0.26.1 API:
    # The SFTTrainer in trl 0.26.1 uses a much simpler API.
    # Key differences from older versions:
    # 1. Uses 'processing_class' instead of 'tokenizer'
    # 2. No 'dataset_text_field' - auto-detects 'messages' format
    # 3. No 'max_seq_length' - controlled via tokenizer.model_max_length
    # 4. No 'packing' parameter at top level
    # 5. No 'dataset_num_proc' parameter
    
    # Ensure tokenizer has model_max_length set
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e9:
        tokenizer.model_max_length = config.max_seq_length
        print(f"Set tokenizer.model_max_length to {config.max_seq_length}")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # trl 0.26.1 uses 'processing_class' instead of 'tokenizer'
    )
    
    return trainer


def train_model(
    trainer: Any,
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute the training loop.
    
    Args:
        trainer: SFTTrainer instance
        resume_from_checkpoint: Path to checkpoint directory to resume from
    
    Returns:
        Dictionary with training metrics
    
    EDUCATIONAL NOTE - Training Loop:
    The trainer internally handles:
    1. Data loading and batching
    2. Forward pass through model
    3. Loss calculation (only on assistant tokens)
    4. Backward pass (gradient computation)
    5. Gradient accumulation
    6. Optimizer step (weight update)
    7. Learning rate scheduling
    8. Checkpointing
    9. Logging
    
    You don't need to write any of this manually!
    """
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    # Run training
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Log final metrics
    metrics = train_result.metrics
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"  Total steps: {metrics.get('train_steps', 'N/A')}")
    print(f"  Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Runtime: {metrics.get('train_runtime', 0):.0f}s")
    print("="*60 + "\n")
    
    return metrics


def save_lora_adapters(
    model: Any,
    tokenizer: Any,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save only the LoRA adapters (not full model).
    
    Args:
        model: Model with trained LoRA adapters
        tokenizer: Tokenizer to save alongside
        output_dir: Directory to save to (defaults to settings)
    
    Returns:
        Path to saved adapter directory
    
    EDUCATIONAL NOTE - Why Save Only Adapters:
    
    Full model save:
    - Size: ~8GB (full precision) or ~4GB (quantized)
    - Contains frozen weights we didn't change
    - Slower to upload/download
    
    LoRA adapter save:
    - Size: ~50-100MB
    - Contains only the trained adapter matrices
    - Fast to upload/share
    - Can be applied to any compatible base model
    
    To use later:
    1. Load base model (same as training)
    2. Load LoRA adapters with PeftModel.from_pretrained()
    3. Or merge: model = model.merge_and_unload()
    """
    output_dir = output_dir or (settings.output_dir / "lora_adapters")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving LoRA adapters to: {output_dir}")
    
    # Save the model (only adapters since it's a PEFT model)
    model.save_pretrained(output_dir)
    
    # Save tokenizer for convenience
    tokenizer.save_pretrained(output_dir)
    
    print("✓ Adapters saved successfully")
    
    # Print size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    
    return output_dir
