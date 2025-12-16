"""
Model Registry
==============

WHAT THIS MODULE DOES:
This module maintains a registry of all supported language models for fine-tuning,
including their metadata (parameter count, quantization options, HuggingFace IDs).

WHY WE NEED THIS:
1. **Model Swapping**: Makes it trivial to experiment with different models by
   changing a single configuration parameter.

2. **Documentation**: Each model's characteristics (size, memory requirements,
   strengths) are documented in one place.

3. **Validation**: Ensures only supported models are used, preventing runtime errors
   from typos or unsupported model IDs.

HOW IT WORKS:
- Define ModelConfig dataclass with all model metadata
- Maintain a registry dictionary mapping model variants to configs
- Provide lookup function to retrieve config by variant name
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for a specific language model variant.
    
    EDUCATIONAL NOTE - Model Characteristics:
    When selecting a model for fine-tuning on limited hardware, consider:
    
    1. **Parameter Count**: More parameters = better capability but more VRAM.
       - 1-2B: Fits easily on 6GB, but limited reasoning
       - 3-4B: Sweet spot for 6GB with quantization
       - 7B+: Requires 16GB+ VRAM even with quantization
    
    2. **Architecture**: Some architectures are more memory-efficient:
       - Llama-based (Phi): Good balance of quality and efficiency
       - Gemma: Google's efficient architecture
       - Qwen: Optimized for multilingual and reasoning tasks
    
    3. **Context Window**: Longer context = more VRAM per sample:
       - 2K-4K: Standard for most tasks
       - 8K+: Useful for long documents (EHRs, legal text)
    
    4. **Quantization Support**: All modern models support 4-bit quantization,
       but some have officially optimized quantized releases on HuggingFace.
    """
    
    variant: str
    """Human-readable variant name (e.g., 'phi3', 'gemma2b')"""
    
    hf_model_id: str
    """HuggingFace model identifier for download"""
    
    unsloth_model_id: Optional[str]
    """Unsloth-optimized model ID (if available). Unsloth provides 2x faster
    loading and 50% VRAM reduction through custom CUDA kernels."""
    
    parameter_count: str
    """Model size (e.g., '3.8B', '2B') for documentation"""
    
    context_length: int
    """Maximum sequence length the model was trained on"""
    
    recommended_max_seq_len: int
    """Recommended max sequence length for 6GB VRAM fine-tuning.
    This is often lower than context_length to leave room for gradients."""
    
    min_vram_gb: float
    """Minimum VRAM required for 4-bit quantized inference (no training)"""
    
    min_vram_training_gb: float
    """Minimum VRAM required for 4-bit quantized fine-tuning with batch_size=1"""
    
    strengths: str
    """Key strengths of this model variant (for documentation)"""
    
    chat_template: str
    """Chat template format this model expects (e.g., 'chatml', 'llama')"""


# ============================================================================
# MODEL REGISTRY
# ============================================================================
# This registry contains all supported models. To add a new model, simply add
# a new entry here with its configuration.

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "phi3": ModelConfig(
        variant="phi3",
        hf_model_id="microsoft/Phi-3-mini-4k-instruct",
        unsloth_model_id="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        parameter_count="3.8B",
        context_length=4096,
        recommended_max_seq_len=512,
        min_vram_gb=2.5,
        min_vram_training_gb=5.5,
        strengths=(
            "Strong reasoning and instruction-following. Microsoft's phi family "
            "is trained on high-quality curated data. Excellent for medical domain "
            "due to chain-of-thought capabilities."
        ),
        chat_template="chatml"
    ),
    
    "gemma2b": ModelConfig(
        variant="gemma2b",
        hf_model_id="google/gemma-2b-it",
        unsloth_model_id="unsloth/gemma-2b-it-bnb-4bit",
        parameter_count="2B",
        context_length=8192,
        recommended_max_seq_len=512,
        min_vram_gb=1.5,
        min_vram_training_gb=4.0,
        strengths=(
            "Google's efficient architecture with strong safety alignment. "
            "Lower VRAM requirements make it a safe fallback if Phi-3 OOMs. "
            "Good for classification tasks."
        ),
        chat_template="gemma"
    ),
    
    "qwen1.5": ModelConfig(
        variant="qwen1.5",
        hf_model_id="Qwen/Qwen1.5-1.8B-Chat",
        unsloth_model_id=None,  # Unsloth support may be added in future
        parameter_count="1.8B",
        context_length=32768,
        recommended_max_seq_len=512,
        min_vram_gb=1.2,
        min_vram_training_gb=3.5,
        strengths=(
            "Smallest model option with surprisingly good performance. "
            "Extremely memory-efficient. Good for initial prototyping or "
            "when GPU memory is severely constrained."
        ),
        chat_template="chatml"
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(variant: str) -> ModelConfig:
    """
    Retrieve model configuration by variant name.
    
    Args:
        variant: Model variant name (e.g., 'phi3', 'gemma2b', 'qwen1.5')
    
    Returns:
        ModelConfig object with all model metadata
    
    Raises:
        ValueError: If variant is not found in registry
    
    EDUCATIONAL NOTE - Fail Fast Design:
    We immediately raise an error if an unsupported model is requested.
    This is better than silently using a fallback, which could lead to
    confusion about which model is actually being used.
    """
    if variant not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model variant: '{variant}'. "
            f"Available variants: {available}"
        )
    
    return MODEL_REGISTRY[variant]


def list_available_models() -> Dict[str, str]:
    """
    List all available model variants with their descriptions.
    
    Returns:
        Dictionary mapping variant names to parameter counts and strengths
    
    EDUCATIONAL NOTE - Discoverability:
    This function makes it easy for users to see what models are available
    without digging through code. Useful for CLI help text or documentation.
    """
    return {
        variant: f"{config.parameter_count} - {config.strengths}"
        for variant, config in MODEL_REGISTRY.items()
    }


def estimate_vram_requirements(
    variant: str,
    batch_size: int = 1,
    max_seq_length: int = 512,
    gradient_accumulation_steps: int = 1
) -> Dict[str, float]:
    """
    Estimate VRAM requirements for a given configuration.
    
    Args:
        variant: Model variant name
        batch_size: Training batch size per device
        max_seq_length: Maximum sequence length
        gradient_accumulation_steps: Gradient accumulation steps
    
    Returns:
        Dictionary with VRAM estimates (model, gradients, activations, total)
    
    EDUCATIONAL NOTE - VRAM Breakdown:
    During fine-tuning, VRAM is used for:
    1. Model weights (quantized): ~2-3GB for 3.8B model in 4-bit
    2. Gradients: Roughly same size as trainable params (LoRA adapters)
    3. Optimizer states (AdamW): 2x gradient size for momentum and variance
    4. Activations: Depends on batch_size * seq_length * hidden_dim
    
    This is a rough estimate. Actual usage depends on:
    - Gradient checkpointing (saves 40% activation memory)
    - Optimizer choice (8-bit AdamW uses less memory)
    - LoRA rank (higher rank = more gradients)
    """
    config = get_model_config(variant)
    
    # Base model in 4-bit
    model_vram = config.min_vram_gb
    
    # LoRA adapters (trainable parameters)
    # Rough estimate: ~1-2% of model size for rank=16
    lora_vram = model_vram * 0.02
    
    # Optimizer states (paged_adamw_8bit uses ~1.5x gradient size)
    optimizer_vram = lora_vram * 1.5
    
    # Activations scale with batch size and sequence length
    # Rough formula: batch * seq_len * hidden_size / quantization_factor
    # For 4-bit quantized model with gradient checkpointing
    activation_scale = (batch_size * max_seq_length) / (512 * 1)  # Normalized
    activation_vram = 0.5 * activation_scale  # Base 0.5GB for batch=1, seq=512
    
    total = model_vram + lora_vram + optimizer_vram + activation_vram
    
    return {
        "model_weights_gb": round(model_vram, 2),
        "lora_adapters_gb": round(lora_vram, 2),
        "optimizer_states_gb": round(optimizer_vram, 2),
        "activations_gb": round(activation_vram, 2),
        "total_estimated_gb": round(total, 2),
        "fits_6gb": total <= 6.0
    }
