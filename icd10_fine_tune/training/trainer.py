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

    EDUCATIONAL NOTE - Dataclass Field Ordering:
    In Python dataclasses, all fields with default values must come AFTER
    fields without defaults. To avoid this constraint, we give all fields
    defaults and populate them via from_settings() factory method.
    """

    # Core training parameters (populated via from_settings())
    output_dir: Path = None
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    optim: str = "paged_adamw_8bit"
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42

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


def create_formatting_func(tokenizer: Any, max_seq_length: int = 512):
    """
    Create a formatting function for Unsloth's SFTTrainer.

    Args:
        tokenizer: The tokenizer with a chat_template
        max_seq_length: Maximum sequence length for truncation

    Returns:
        A function that formats dataset examples into text prompts

    EDUCATIONAL NOTE - Why Unsloth Needs formatting_func:

    Unsloth's SFTTrainer requires a `formatting_func` that converts your
    dataset examples into formatted text strings. This function:

    1. Takes examples (either single dict or batch dict with lists)
    2. Extracts the "messages" field from each example
    3. Applies the tokenizer's chat_template to format the conversation
    4. Returns a list of formatted text strings

    The chat_template converts messages like:
    [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]

    Into formatted text like:
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>"

    This formatted text is then tokenized and used for training.

    EDUCATIONAL NOTE - Single Example vs Batch:
    Unsloth tests the formatting_func with a single example first, then uses
    it with batches during training. We need to handle both cases:
    - Single example: {"messages": [{"role": "user", ...}, ...]}
    - Batch: {"messages": [[{"role": "user", ...}, ...], [...]]}

    EDUCATIONAL NOTE - Truncation Strategy:
    We apply chat template first to get the formatted text, then let the
    SFTTrainer handle tokenization and truncation. This ensures proper
    handling of sequences that exceed max_seq_length.
    """

    def formatting_func(examples):
        """
        Format examples into text prompts.

        Args:
            examples: Either a single example dict or a batch dict with lists

        Returns:
            List of formatted text strings (or single string for single example)
        """
        messages_data = examples.get("messages", [])

        # EDUCATIONAL NOTE - Detecting Single vs Batch:
        # Check if this is a single example or a batch:
        # - Single example: messages_data is a list of message dicts
        # - Batch: messages_data is a list of lists of message dicts
        #
        # We detect this by checking if the first element is a dict (single)
        # or a list (batch).
        if messages_data and isinstance(messages_data[0], dict):
            # Single example: messages_data = [{"role": "user", ...}, ...]
            formatted_text = tokenizer.apply_chat_template(
                messages_data, tokenize=False, add_generation_prompt=False
            )
            return [formatted_text]  # Return as list for consistency
        else:
            # Batch: messages_data = [[{"role": "user", ...}, ...], [...]]
            formatted_texts = []
            for messages in messages_data:
                formatted_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)
            return formatted_texts

    return formatting_func


def create_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    config: Optional[TrainingConfig] = None,
    dataset_text_field: str = "text",  # Kept for API compatibility but not used
    max_seq_length: Optional[int] = None,  # Kept for API compatibility but not used
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

    EDUCATIONAL NOTE - Unsloth SFTTrainer Requirements:

    When using Unsloth's optimized SFTTrainer, you MUST provide a
    `formatting_func` parameter. This is different from the standard
    trl SFTTrainer which can auto-detect "messages" format.

    Unsloth requires explicit formatting because:
    1. It applies custom optimizations to the training loop
    2. It needs to know exactly how to convert your data to text
    3. This gives you full control over the prompt format

    The formatting_func takes a batch of examples and returns formatted
    text strings that will be tokenized and used for training.
    """
    from trl import SFTTrainer

    if config is None:
        config = TrainingConfig.from_settings()

    # Create training arguments
    training_args = create_training_arguments(config)

    print(f"\n{'=' * 60}")
    print("Creating SFTTrainer")
    print(f"{'=' * 60}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset) if eval_dataset else 'None'}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}"
    )
    print(f"  Learning rate: {config.learning_rate}")
    print(f"{'=' * 60}\n")

    # EDUCATIONAL NOTE - Unsloth-Specific Configuration:
    # When using Unsloth, we need to:
    # 1. Create a formatting_func that converts messages to text
    # 2. Set tokenizer.model_max_length for sequence truncation
    # 3. Pass formatting_func to SFTTrainer
    # 4. Disable packing to avoid batch size mismatch in fused loss

    # Ensure tokenizer has model_max_length set
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e9:
        tokenizer.model_max_length = config.max_seq_length
        print(f"Set tokenizer.model_max_length to {config.max_seq_length}")

    # Create the formatting function for Unsloth
    formatting_func = create_formatting_func(tokenizer, max_seq_length=config.max_seq_length)
    print("✓ Created formatting function for Unsloth")

    # EDUCATIONAL NOTE - Packing and Truncation:
    # Packing combines multiple short examples into one sequence to improve
    # efficiency. However, with Unsloth's fused cross-entropy loss, packing
    # can cause batch size mismatches when sequences need truncation.
    # We disable packing to ensure stable training.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,  # Required by Unsloth
        max_seq_length=config.max_seq_length,  # Explicit max length for Unsloth
        packing=False,  # Disable packing to avoid batch size mismatch
    )

    return trainer


def train_model(trainer: Any, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
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
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    # Run training
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Log final metrics
    metrics = train_result.metrics
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Total steps: {metrics.get('train_steps', 'N/A')}")
    print(f"  Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Runtime: {metrics.get('train_runtime', 0):.0f}s")
    print("=" * 60 + "\n")

    return metrics


def save_lora_adapters(model: Any, tokenizer: Any, output_dir: Optional[Path] = None) -> Path:
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
