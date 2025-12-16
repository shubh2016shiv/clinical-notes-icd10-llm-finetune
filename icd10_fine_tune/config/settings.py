"""
Centralized Configuration System
=================================

WHAT THIS MODULE DOES:
This module provides the single source of truth for all configurable parameters in the
fine-tuning system. It uses Pydantic's BaseSettings for type-safe configuration with
environment variable override support.

WHY WE NEED THIS:
1. **Single Source of Truth**: All tunable parameters in one place prevents scattered
   configuration across multiple files, reducing bugs and making it easy to adjust
   hyperparameters for experimentation.

2. **Type Safety**: Pydantic validates all configurations at startup, catching errors
   before training begins (e.g., invalid paths, wrong data types).

3. **Environment Variable Override**: Allows production deployments to override defaults
   via environment variables without code changes.

4. **Documentation**: Each parameter is documented with its purpose, making it clear
   what can be adjusted and why.

HOW IT WORKS:
- Reads from .env file if present
- Validates all settings on instantiation
- Provides typed access to configuration throughout the codebase
- Fails fast if required settings are missing or invalid
"""

from pathlib import Path
from typing import List, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FineTuningSettings(BaseSettings):
    """
    Centralized configuration for ICD-10 fine-tuning system.
    
    This class defines all tunable parameters for the fine-tuning pipeline.
    Parameters can be overridden via environment variables (prefixed with FT_).
    
    Design Philosophy:
    - Explicit over implicit: Every parameter has a clear purpose
    - Safe defaults: Default values are conservative (optimized for 6GB VRAM)
    - Validation: All paths and values are validated at instantiation
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FT_",  # All environment variables prefixed with FT_
        case_sensitive=False,
        extra="ignore"  # Ignore unknown env vars
    )
    
    # ============================================================================
    # DATA PATHS
    # ============================================================================
    # These define where to find input data and where to save outputs.
    
    raw_data_path: Path = Field(
        default=Path("data/extracted/multi_profile_clinical_notes_20251215_215353.json"),
        description="Path to raw clinical notes JSON file"
    )
    
    processed_data_dir: Path = Field(
        default=Path("icd10_fine_tune/data/processed"),
        description="Directory to save processed training/validation data"
    )
    
    output_dir: Path = Field(
        default=Path("icd10_fine_tune/outputs"),
        description="Root directory for all outputs (models, logs, reports)"
    )
    
    # ============================================================================
    # MODEL SELECTION
    # ============================================================================
    # These parameters define which model to use for fine-tuning.
    #
    # EDUCATIONAL NOTE - Model Selection Strategy:
    # For limited VRAM (6GB), we use quantized small models:
    # - Phi-3 Mini: 3.8B params, strong reasoning, medical domain-friendly
    # - Gemma-2b: 2B params, more conservative VRAM usage
    # - Qwen-1.5-1.8B: 1.8B params, smallest option if other models OOM
    
    model_id: str = Field(
        default="Qwen/Qwen1.5-1.8B-Chat",
        description="HuggingFace model identifier (use unsloth variants for better memory efficiency)"
    )
    
    model_variant: Literal["phi3", "gemma2b", "qwen1.5"] = Field(
        default="qwen1.5",
        description="Model variant for configuration lookup in model_registry"
    )
    
    use_unsloth: bool = Field(
        default=True,  # Enabled for Linux (2x faster, 50% less VRAM)
        description="Use Unsloth's optimized model loading (2x faster, 50% less VRAM)"
    )
    
    # ============================================================================
    # QUANTIZATION SETTINGS
    # ============================================================================
    # Quantization reduces model precision to save memory.
    #
    # EDUCATIONAL NOTE - Why 4-bit Quantization?:
    # - FP16 (standard): 16 bits per parameter = ~7.6GB for 3.8B model
    # - INT8: 8 bits per parameter = ~3.8GB
    # - INT4 (4-bit): 4 bits per parameter = ~1.9GB
    # With 6GB VRAM, 4-bit is essential to leave room for gradients and activations.
    
    load_in_4bit: bool = Field(
        default=True,  # Enabled for Linux with CUDA (critical for 6GB VRAM)
        description="Load model in 4-bit quantization (critical for 6GB VRAM)"
    )
    
    bnb_4bit_compute_dtype: str = Field(
        default="float16",
        description="Compute dtype for 4-bit quantization (float16 for RTX 2060)"
    )
    
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="Quantization type: 'nf4' (normal float 4) is optimal for LLMs"
    )
    
    use_nested_quant: bool = Field(
        default=True,
        description="Double quantization for additional memory savings"
    )
    
    # ============================================================================
    # LORA CONFIGURATION
    # ============================================================================
    # LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method.
    #
    # EDUCATIONAL NOTE - How LoRA Works:
    # Instead of updating all model parameters (expensive), LoRA injects small
    # "adapter" matrices into attention layers. These adapters learn the task
    # while keeping the base model frozen.
    #
    # Key Parameters:
    # - r (rank): Size of adapter matrices. Higher = more capacity but more memory.
    #   Typical range: 8-64. We use 16 as a balance.
    # - alpha: Scaling factor. Usually set equal to r or 2*r.
    # - dropout: Regularization to prevent overfitting on small datasets.
    
    lora_r: int = Field(
        default=16,
        description="LoRA rank (adapter matrix dimension). Higher = more capacity but more VRAM."
    )
    
    lora_alpha: int = Field(
        default=16,
        description="LoRA scaling factor. Common to set equal to rank."
    )
    
    lora_dropout: float = Field(
        default=0.05,
        description="Dropout probability for LoRA layers (prevents overfitting)"
    )
    
    lora_target_modules: List[str] = Field(
        default=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj"       # MLP layers
        ],
        description="Which model layers to apply LoRA adapters to"
    )
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    # These control the training process.
    #
    # EDUCATIONAL NOTE - Batch Size & Gradient Accumulation:
    # With 6GB VRAM, we can only fit batch_size=1 in memory. But small batches
    # lead to noisy gradients. Solution: Gradient Accumulation.
    # 
    # Instead of updating after every sample, we accumulate gradients over
    # multiple steps (e.g., 4), then update. This simulates batch_size=4 without
    # the memory cost.
    #
    # Effective Batch Size = per_device_batch_size * gradient_accumulation_steps * num_gpus
    
    per_device_train_batch_size: int = Field(
        default=1,
        description="Batch size per GPU. MUST be 1 for 6GB VRAM."
    )
    
    gradient_accumulation_steps: int = Field(
        default=4,
        description="Accumulate gradients over N steps before updating (simulates larger batch)"
    )
    
    learning_rate: float = Field(
        default=2e-4,
        description="Peak learning rate. Standard for LoRA fine-tuning is 1e-4 to 5e-4."
    )
    
    num_train_epochs: int = Field(
        default=3,
        description="Number of full passes through the training data"
    )
    
    max_steps: int = Field(
        default=-1,
        description="Max training steps (overrides num_train_epochs if > 0)"
    )
    
    warmup_ratio: float = Field(
        default=0.03,
        description="Fraction of training steps for learning rate warmup (stabilizes early training)"
    )
    
    lr_scheduler_type: str = Field(
        default="cosine",
        description="Learning rate schedule: 'cosine', 'linear', or 'constant'"
    )
    
    # ============================================================================
    # MEMORY OPTIMIZATION
    # ============================================================================
    # These settings are critical for fitting training on 6GB VRAM.
    
    max_seq_length: int = Field(
        default=512,
        description="Maximum sequence length in tokens. Reduced to 512 for 6GB VRAM (can increase if needed)."
    )
    
    gradient_checkpointing: bool = Field(
        default=True,
        description="Trade compute for memory (40% VRAM reduction, 20% slower training)"
    )
    
    optim: str = Field(
        default="paged_adamw_8bit",
        description="Optimizer: 'paged_adamw_8bit' uses less VRAM than standard AdamW"
    )
    
    # ============================================================================
    # MULTI-GPU SETTINGS (FOR FUTURE SCALING)
    # ============================================================================
    # These enable distributed training across multiple GPUs.
    #
    # EDUCATIONAL NOTE - Distributed Data Parallel (DDP):
    # DDP replicates the model across GPUs. Each GPU processes a different batch,
    # then gradients are averaged before updating. This scales training linearly
    # with GPU count.
    
    use_ddp: bool = Field(
        default=False,
        description="Enable Distributed Data Parallel (set to True for multi-GPU)"
    )
    
    world_size: int = Field(
        default=1,
        description="Number of GPUs to use (auto-detected if not specified)"
    )
    
    # ============================================================================
    # LOGGING & CHECKPOINTING
    # ============================================================================
    
    logging_steps: int = Field(
        default=10,
        description="Log training metrics every N steps"
    )
    
    save_steps: int = Field(
        default=100,
        description="Save checkpoint every N steps"
    )
    
    save_total_limit: int = Field(
        default=3,
        description="Keep only the last N checkpoints (saves disk space)"
    )
    
    # ============================================================================
    # EVIDENTLY AI INTEGRATION
    # ============================================================================
    
    evidently_api_key: str = Field(
        default="",
        description="Evidently AI API key for cloud dashboard (from .env)"
    )
    
    evidently_project_name: str = Field(
        default="icd10-fine-tuning",
        description="Project name in Evidently Cloud"
    )
    
    evidently_workspace: str = Field(
        default="default",
        description="Workspace name in Evidently Cloud"
    )
    
    # ============================================================================
    # DATA FORMAT SETTINGS
    # ============================================================================
    
    validation_split: float = Field(
        default=0.2,
        description="Fraction of data to use for validation"
    )
    
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    @field_validator("raw_data_path", "processed_data_dir", "output_dir")
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        """
        Validate that paths are properly formed.
        
        WHY THIS VALIDATION:
        We want to fail fast at startup if paths are misconfigured, rather than
        discovering path errors hours into a training run.
        """
        if not isinstance(v, Path):
            v = Path(v)
        return v
    
    def get_checkpoint_dir(self) -> Path:
        """Get the directory where model checkpoints will be saved."""
        return self.output_dir / "checkpoints"
    
    def get_logs_dir(self) -> Path:
        """Get the directory where training logs will be saved."""
        return self.output_dir / "logs"
    
    def get_reports_dir(self) -> Path:
        """Get the directory where Evidently reports will be saved."""
        return self.output_dir / "reports"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================
# We create a single global instance of settings that can be imported throughout
# the codebase. This ensures consistency and avoids re-parsing .env files.
#
# EDUCATIONAL NOTE - Singleton Pattern:
# Instead of creating settings instances everywhere, we create one here and
# import it. This is explicit dependency injection at the module level.

settings = FineTuningSettings()
