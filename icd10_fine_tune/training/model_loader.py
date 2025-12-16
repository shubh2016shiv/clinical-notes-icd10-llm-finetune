"""
Model Loader with Unsloth Optimization
=======================================

WHAT THIS MODULE DOES:
Loads quantized language models using Unsloth for memory-efficient fine-tuning.
Unsloth provides 2x faster training and 50% VRAM reduction through custom CUDA kernels.

WHY WE NEED THIS:
1. **VRAM Efficiency**: On 6GB GPU, we need every optimization possible.
   Unsloth's custom kernels reduce memory usage significantly.

2. **Abstraction**: Hides the complexity of quantization setup behind a clean API.
   Callers don't need to understand BitsAndBytesConfig or HF model loading details.

3. **Flexibility**: Supports switching between Unsloth (optimized) and standard
   HuggingFace loading, useful when Unsloth isn't available (e.g., CPU testing).

HOW IT WORKS:
1. Select model from registry based on variant name
2. Configure 4-bit quantization (NF4 with double quantization)
3. Load model using Unsloth's FastLanguageModel (or fallback to HF)
4. Configure model for training (enable gradient checkpointing, etc.)
5. Return model and tokenizer ready for LoRA adapter injection

EDUCATIONAL NOTE - Unsloth vs Standard Loading:
Standard HuggingFace Loading:
- Uses AutoModelForCausalLM.from_pretrained()
- Standard attention implementation
- Full precision or basic quantization

Unsloth Loading:
- Uses FastLanguageModel.from_pretrained()
- Optimized attention kernels (xformers-like)
- Better memory management during backpropagation
- Automatic gradient checkpointing optimization
"""

from typing import Tuple, Optional, Any

from icd10_fine_tune.config.settings import settings
from icd10_fine_tune.config.model_registry import get_model_config, ModelConfig


# ============================================================================
# TYPE ALIASES
# ============================================================================
# We use Any for PreTrainedModel and PreTrainedTokenizer to avoid importing
# heavy transformers dependencies at module load time.
ModelAndTokenizer = Tuple[Any, Any]


def check_gpu_availability() -> dict:
    """
    Check GPU availability and print diagnostics.
    
    Returns:
        Dictionary with GPU information
    
    Raises:
        RuntimeError: If CUDA is not available
    
    EDUCATIONAL NOTE - Why Check GPU First:
    We check GPU availability before loading the model because:
    1. Fail fast: No point downloading/loading if GPU isn't available
    2. Clear error: Better to show "No GPU" than silent crash
    3. Diagnostics: Helps debug driver/CUDA version issues
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Install with: "
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
    
    print("\n" + "="*60)
    print("GPU Diagnostics")
    print("="*60)
    
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n[ERROR] CUDA is not available!")
        print("\nPossible causes:")
        print("  1. No NVIDIA GPU detected")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch CPU-only version installed")
        print("\nTo fix:")
        print("  1. Check GPU: nvidia-smi")
        print("  2. Install CUDA drivers from NVIDIA")
        print("  3. Reinstall PyTorch with CUDA support")
        print("="*60 + "\n")
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    
    # Get GPU info
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"  GPU Count: {gpu_count}")
    print(f"  GPU Name: {gpu_name}")
    print(f"  GPU Memory: {gpu_memory:.2f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    # Check available memory
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    free = gpu_memory - reserved
    
    print(f"  Memory Allocated: {allocated:.2f} GB")
    print(f"  Memory Reserved: {reserved:.2f} GB")
    print(f"  Memory Free: {free:.2f} GB")
    print("="*60 + "\n")
    
    if free < 3.0:
        print("[WARNING] Less than 3GB free GPU memory!")
        print("   Consider closing other GPU applications.\n")
    
    return {
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "free_memory_gb": free
    }


def load_model_for_training(
    model_variant: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    use_unsloth: Optional[bool] = None,
    dtype: Optional[str] = None
) -> ModelAndTokenizer:
    """
    Load a quantized model ready for LoRA fine-tuning.
    
    Args:
        model_variant: Model variant name (e.g., 'phi3', 'gemma2b').
                      Defaults to settings.model_variant.
        max_seq_length: Maximum sequence length. Defaults to settings.max_seq_length.
        use_unsloth: Whether to use Unsloth optimizations. Defaults to settings.use_unsloth.
        dtype: Data type for computation ('float16', 'bfloat16'). Defaults based on settings.
    
    Returns:
        Tuple of (model, tokenizer) ready for LoRA adapter injection
    
    Raises:
        ImportError: If required libraries (unsloth, transformers) are not installed
        ValueError: If model variant is not found in registry
        RuntimeError: If CUDA is not available
    
    EDUCATIONAL NOTE - Why Return Tuple?
    We return (model, tokenizer) together because:
    1. They're always used together during training
    2. Ensures tokenizer matches model (same vocab, special tokens)
    3. Prevents accidentally mixing mismatched pairs
    
    VRAM Usage After Loading (approximate for Phi-3 Mini):
    - 4-bit quantized model: ~2.5GB
    - Tokenizer: ~50MB
    - Total: ~2.6GB (leaves ~3.4GB for training overhead)
    """
    # Check GPU availability first
    check_gpu_availability()
    
    # Apply defaults from settings
    model_variant = model_variant or settings.model_variant
    max_seq_length = max_seq_length or settings.max_seq_length
    use_unsloth = use_unsloth if use_unsloth is not None else settings.use_unsloth
    dtype = dtype or settings.bnb_4bit_compute_dtype
    
    # Get model configuration
    model_config = get_model_config(model_variant)
    
    # Log what we're loading
    print(f"\n{'='*60}")
    print(f"Loading Model: {model_config.variant}")
    print(f"{'='*60}")
    print(f"  Model ID: {model_config.hf_model_id}")
    print(f"  Parameters: {model_config.parameter_count}")
    print(f"  Max Seq Length: {max_seq_length}")
    print("  4-bit Quantization: Enabled")
    print(f"  Unsloth Optimization: {use_unsloth}")
    print(f"  Compute Dtype: {dtype}")
    print(f"{'='*60}\n")
    
    if use_unsloth:
        return _load_with_unsloth(model_config, max_seq_length, dtype)
    else:
        return _load_with_transformers(model_config, max_seq_length, dtype)


def _load_with_unsloth(
    model_config: ModelConfig,
    max_seq_length: int,
    dtype: str
) -> ModelAndTokenizer:
    """
    Load model using Unsloth's optimized loader.
    
    EDUCATIONAL NOTE - Unsloth FastLanguageModel:
    Unsloth provides FastLanguageModel.from_pretrained() which:
    1. Automatically configures 4-bit quantization
    2. Applies memory-efficient attention
    3. Sets up gradient checkpointing optimally
    4. Patches model for faster backprop
    
    The returned model is already quantized and optimized.
    You just need to add LoRA adapters and start training.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "Unsloth is not installed. Install with: "
            "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'\n"
            f"Original error: {e}"
        )
    
    # Use Unsloth's pre-quantized model if available
    model_id = model_config.unsloth_model_id or model_config.hf_model_id
    
    # Map dtype string to torch dtype
    import torch
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    
    print(f"Loading with Unsloth: {model_id}")
    
    # Load model with Unsloth optimizations
    # EDUCATIONAL NOTE - Key Parameters:
    # - max_seq_length: Controls memory allocation for sequences
    # - dtype: Compute precision (float16 for RTX 2060)
    # - load_in_4bit: Enable 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=True,  # Critical for 6GB VRAM
    )
    
    print("[OK] Model loaded successfully with Unsloth optimizations")
    _print_memory_usage()
    
    return model, tokenizer


def _load_with_transformers(
    model_config: ModelConfig,
    max_seq_length: int,
    dtype: str
) -> ModelAndTokenizer:
    """
    Load model using standard HuggingFace transformers.
    
    This is a fallback when Unsloth is not available (e.g., CPU, unsupported GPU).
    
    EDUCATIONAL NOTE - BitsAndBytesConfig:
    BitsAndBytesConfig controls quantization behavior:
    - load_in_4bit: Enable 4-bit quantization
    - bnb_4bit_compute_dtype: Precision for computations (float16)
    - bnb_4bit_quant_type: 'nf4' (Normal Float 4) is optimal for LLMs
    - bnb_4bit_use_double_quant: Nested quantization for extra savings
    """
    import time
    start_time = time.time()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
    except ImportError as e:
        raise ImportError(
            f"transformers or torch not installed: {e}\n"
            f"Install with: pip install transformers torch"
        )
    
    # Configure 4-bit quantization
    # EDUCATIONAL NOTE - NF4 Quantization:
    # NF4 (Normal Float 4) is specifically designed for neural network weights.
    # It assumes weights follow a normal distribution and optimizes the
    # quantization grid accordingly. This gives better quality than standard
    # 4-bit integer quantization.
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    
    # Check if quantization is enabled
    if settings.load_in_4bit:
        print(f"\n{'='*60}")
        print("Configuring 4-bit Quantization")
        print(f"{'='*60}")
        print("  Quantization Type: NF4 (Normal Float 4)")
        print(f"  Compute Dtype: {dtype}")
        print(f"  Double Quantization: {settings.use_nested_quant}")
        print(f"{'='*60}\n")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=settings.use_nested_quant,
        )
    else:
        print(f"\n{'='*60}")
        print("Loading WITHOUT Quantization (float16)")
        print(f"{'='*60}")
        print("  WARNING: This will use more VRAM (~3-4GB for Qwen 1.5)")
        print(f"{'='*60}\n")
        bnb_config = None
    
    print(f"Loading model: {model_config.hf_model_id}")
    print("This may take 5-10 minutes on first load (downloading model)...")
    print("Subsequent loads will be faster (using cached model).\n")
    
    try:
        # ================================================================
        # STEP 1: Load Tokenizer
        # ================================================================
        print("[1/3] Loading tokenizer...")
        tokenizer_start = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.hf_model_id,
            trust_remote_code=True
        )
        
        tokenizer_time = time.time() - tokenizer_start
        print(f"[OK] Tokenizer loaded in {tokenizer_time:.1f}s")
        
        # Set padding token if not present
        # EDUCATIONAL NOTE - Padding Token:
        # Some models (like LLaMA) don't have a padding token by default.
        # We need one for batched training. Using EOS token is a common workaround.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"  Set pad_token to eos_token (ID: {tokenizer.eos_token_id})")
        
        # ================================================================
        # STEP 2: Load Model
        # ================================================================
        if settings.load_in_4bit:
            print("\n[2/3] Loading model with 4-bit quantization...")
        else:
            print("\n[2/3] Loading model in float16...")
        print("  This is the slow step - please be patient...")
        model_start = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_model_id,
            quantization_config=bnb_config if settings.load_in_4bit else None,
            device_map="auto",  # Automatically place on GPU
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",  # Disable flash attention (not available on Windows)
            low_cpu_mem_usage=True,  # Reduce RAM usage during loading
        )
        
        model_time = time.time() - model_start
        print(f"[OK] Model loaded in {model_time:.1f}s")
        
        # ================================================================
        # STEP 3: Enable Gradient Checkpointing
        # ================================================================
        print("\n[3/3] Configuring model for training...")
        
        # Enable gradient checkpointing for memory efficiency
        # EDUCATIONAL NOTE - Gradient Checkpointing:
        # Normally, all activations are stored during forward pass for backprop.
        # Gradient checkpointing discards intermediate activations and recomputes
        # them during backward pass. This saves ~40% VRAM at cost of ~20% slower training.
        if settings.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("  [OK] Gradient checkpointing enabled")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[OK] Model loaded successfully in {total_time:.1f}s")
        print(f"{'='*60}")
        _print_memory_usage()
        
        return model, tokenizer
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n{'='*60}")
        print("[ERROR] GPU OUT OF MEMORY")
        print(f"{'='*60}")
        print(f"Error: {str(e)}\n")
        print("The model is too large for your GPU memory.")
        print("\nSuggested fixes:")
        print("  1. Reduce max_seq_length in config (currently: {max_seq_length})")
        print("  2. Try a smaller model: set MODEL_VARIANT=gemma2b in .env")
        print("  3. Close other GPU applications and try again")
        print("  4. Ensure no other Python processes are using GPU")
        print(f"{'='*60}\n")
        raise
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("[ERROR] MODEL LOADING FAILED")
        print(f"{'='*60}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}\n")
        
        # Provide specific guidance based on error type
        if "CUDA" in str(e).upper():
            print("This appears to be a CUDA-related error.")
            print("\nSuggested fixes:")
            print("  1. Check GPU status: nvidia-smi")
            print("  2. Restart your terminal/IDE")
            print("  3. Update NVIDIA drivers")
            print("  4. Reinstall PyTorch with CUDA support")
        elif "flash" in str(e).lower() or "attention" in str(e).lower():
            print("This appears to be an attention mechanism error.")
            print("\nSuggested fixes:")
            print("  1. The code already sets attn_implementation='eager'")
            print("  2. Try updating transformers: pip install -U transformers")
            print("  3. Check if model is compatible with your PyTorch version")
        elif "memory" in str(e).lower():
            print("This appears to be a memory-related error.")
            print("\nSuggested fixes:")
            print("  1. Close other applications to free RAM")
            print("  2. Reduce max_seq_length in config")
            print("  3. Try a smaller model (gemma2b)")
        else:
            print("Unexpected error during model loading.")
            print("\nSuggested fixes:")
            print("  1. Check internet connection (model may need downloading)")
            print("  2. Clear HuggingFace cache: rm -rf ~/.cache/huggingface")
            print("  3. Try updating transformers: pip install -U transformers")
            print("  4. Check GitHub issues for this model")
        
        print(f"\n{'='*60}")
        print("Full error traceback:")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise



def _print_memory_usage() -> None:
    """Print current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception:
        pass  # Silently fail if torch not available


def prepare_model_for_lora(
    model: Any,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    target_modules: Optional[list] = None,
    use_unsloth: Optional[bool] = None
) -> Any:
    """
    Add LoRA adapters to a loaded model.
    
    Args:
        model: Pre-loaded model (from load_model_for_training)
        lora_r: LoRA rank. Defaults to settings.lora_r
        lora_alpha: LoRA scaling factor. Defaults to settings.lora_alpha
        lora_dropout: Dropout probability. Defaults to settings.lora_dropout
        target_modules: Modules to apply LoRA. Defaults to settings.lora_target_modules
        use_unsloth: Whether to use Unsloth for LoRA. Defaults to settings.use_unsloth
    
    Returns:
        Model with LoRA adapters attached
    
    EDUCATIONAL NOTE - LoRA (Low-Rank Adaptation):
    Instead of updating all 3.8B parameters, LoRA:
    1. Freezes the base model weights
    2. Injects small adapter matrices into attention/MLP layers
    3. Only trains these adapters (~0.1% of parameters)
    
    LoRA Decomposition:
    For a weight matrix W (d x k), LoRA adds:
    W_new = W + BA where B is (d x r) and A is (r x k)
    
    With r=16, we add 16*(d+k) parameters instead of d*k
    For Phi-3's attention layers, this can be 1000x fewer parameters!
    """
    # Apply defaults from settings
    lora_r = lora_r or settings.lora_r
    lora_alpha = lora_alpha or settings.lora_alpha
    lora_dropout = lora_dropout or settings.lora_dropout
    target_modules = target_modules or settings.lora_target_modules
    use_unsloth = use_unsloth if use_unsloth is not None else settings.use_unsloth
    
    print("\nConfiguring LoRA Adapters:")
    print(f"  Rank (r): {lora_r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Dropout: {lora_dropout}")
    print(f"  Target Modules: {target_modules}")
    print(f"  Using Unsloth: {use_unsloth}")
    
    # CRITICAL FIX: Only try Unsloth if explicitly enabled
    # This prevents hanging on Windows where Unsloth may have compatibility issues
    if use_unsloth:
        try:
            # Try Unsloth's optimized LoRA
            from unsloth import FastLanguageModel
            
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",  # Don't train biases
                use_gradient_checkpointing="unsloth",  # Optimized checkpointing
                random_state=settings.random_seed,
            )
            print("[OK] LoRA adapters added with Unsloth optimization")
            
        except ImportError:
            print("[WARNING] Unsloth not available, falling back to standard PEFT")
            use_unsloth = False  # Fall through to standard PEFT
    
    if not use_unsloth:
        # Use standard PEFT
        print("[DEBUG] Importing PEFT modules...")
        from peft import LoraConfig, get_peft_model, TaskType
        print("[DEBUG] PEFT modules imported successfully")
        
        # CRITICAL FIX: Disable gradient checkpointing before applying PEFT
        # get_peft_model() hangs when gradient checkpointing is enabled
        # We'll re-enable it after PEFT is applied
        print("[DEBUG] Temporarily disabling gradient checkpointing...")
        gradient_checkpointing_was_enabled = False
        if hasattr(model, 'is_gradient_checkpointing') and model.is_gradient_checkpointing:
            gradient_checkpointing_was_enabled = True
            model.gradient_checkpointing_disable()
            print("[DEBUG] Gradient checkpointing disabled")
        
        # Disable cache (incompatible with gradient checkpointing)
        if hasattr(model, 'config'):
            model.config.use_cache = False
            print("[DEBUG] Disabled model cache")
        
        print("[DEBUG] Creating LoraConfig...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        print("[DEBUG] LoraConfig created successfully")
        
        print("[DEBUG] Calling get_peft_model (this may take 30-60 seconds)...")
        model = get_peft_model(model, lora_config)
        print("[DEBUG] get_peft_model completed!")
        
        # Re-enable gradient checkpointing if it was enabled before
        if gradient_checkpointing_was_enabled:
            print("[DEBUG] Re-enabling gradient checkpointing...")
            model.gradient_checkpointing_enable()
            print("[DEBUG] Gradient checkpointing re-enabled")
        
        # Enable input gradients for training
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
            print("[DEBUG] Enabled input gradients for training")
        
        print("[OK] LoRA adapters added with standard PEFT")
    
    # Print trainable parameters
    _print_trainable_parameters(model)
    
    return model


def _print_trainable_parameters(model: Any) -> None:
    """
    Print the number of trainable parameters in the model.
    
    EDUCATIONAL NOTE - Parameter Efficiency:
    A well-configured LoRA setup should have:
    - Trainable: ~1-10 million parameters (0.1-0.3%)
    - Total: ~3-4 billion parameters
    
    If trainable % is too high, consider:
    - Reducing lora_r (rank)
    - Targeting fewer modules
    """
    trainable_params = 0
    total_params = 0
    
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / total_params
    
    print("\n  Trainable Parameters:")
    print(f"    Trainable: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"    Total: {total_params:,}")
    print(f"    Frozen: {total_params - trainable_params:,}")
