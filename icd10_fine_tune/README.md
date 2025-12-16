# ICD-10 Fine-Tuning System

Enterprise-grade fine-tuning system for extracting ICD-10 codes from clinical notes using small language models.

## Overview

This system demonstrates production-quality fine-tuning on limited hardware (RTX 2060, 6GB VRAM) using:
- **Phi-3 Mini (3.8B)** with 4-bit quantization
- **QLoRA** (parameter-efficient fine-tuning)
- **Unsloth** (2x faster, 50% less VRAM)
- **Evidently AI** (evaluation and monitoring)
- **Multi-GPU support** (DDP-ready for scaling)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_finetune.txt
```

**Note**: Unsloth requires CUDA. If installation fails on Windows, follow [Unsloth Windows installation guide](https://github.com/unslothai/unsloth).

### 2. Configure

Create or update `.env` with your settings:

```bash
# Model Selection
FT_MODEL_VARIANT=phi3  # Options: phi3, gemma2b, qwen1.5

# Hardware Settings
FT_MAX_SEQ_LENGTH=512
FT_PER_DEVICE_TRAIN_BATCH_SIZE=1
FT_GRADIENT_ACCUMULATION_STEPS=4

# Evidently AI
EVIDENTLY_AI=your_api_key_here
```

### 3. Prepare Data

```bash
python -m icd10_fine_tune.scripts.prepare_data
```

### 4. Train (Coming Soon)

```bash
python -m icd10_fine_tune.scripts.train
```

### 5. Evaluate (Coming Soon)

```bash
python -m icd10_fine_tune.scripts.evaluate
```

## Architecture

```
icd10_fine_tune/
├── config/          # Settings & model registry
├── data/            # Data loading, formatting, validation
├── training/        # Model loading & training (TODO)
├── evaluation/      # Inference & metrics (TODO)
├── observability/   # Logging & tracking (TODO)
└── scripts/         # Executable pipelines
```

## Educational Content

Every module includes comprehensive comments explaining:
- **Fine-tuning concepts**: Quantization, LoRA, gradient accumulation
- **VRAM optimization**: How to fit 3.8B model on 6GB GPU
- **Data formatting**: ChatML, JSON outputs, token budgets
- **Multi-GPU**: DDP architecture for scaling

Perfect for learning fine-tuning fundamentals!

## Features

✅ Centralized configuration (Pydantic with env override)  
✅ Model registry (Phi-3, Gemma, Qwen with VRAM estimates)  
✅ Data loader (Pydantic validation, EDA utilities)  
✅ ChatML formatter (instruction-tuning format)  
✅ Format validator (ICD-10 regex, token length checks)  
⏳ Unsloth model loading with 4-bit quantization  
⏳ LoRA fine-tuning with SFTTrainer  
⏳ Multi-GPU support (DDP)  
⏳ Evidently AI integration  

## Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 2060, 3060, 4060)
- **CUDA**: 11.8 or 12.1
- **Python**: 3.10+
- **OS**: Linux (recommended) or Windows with WSL2

## VRAM Optimization

For 6GB GPU:
1. 4-bit quantization (~2.5GB for model)
2. LoRA adapters only (~50MB trainable)
3. Batch size = 1, gradient accumulation = 4
4. Max sequence length = 512
5. Gradient checkpointing enabled
6. 8-bit optimizer (paged_adamw)

Estimated VRAM usage: **~5.5GB** (safe for 6GB)

## Next Steps

See [walkthrough.md](file:///C:/Users/Shubham%20Singh/.gemini/antigravity/brain/79ce59bf-6b70-4ccd-9e7a-104eb0a2102a/walkthrough.md) for detailed progress and remaining work.

## License

MIT
