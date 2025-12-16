"""
ICD-10 Fine-Tuning System
=========================

An enterprise-grade fine-tuning system for extracting ICD-10 codes from clinical notes
using small language models (SLMs) with parameter-efficient fine-tuning (QLoRA).

This package follows a layered architecture:
- config: Centralized configuration and model registry
- data: Data loading, preprocessing, formatting, and validation
- training: Model loading, LoRA configuration, and training orchestration
- evaluation: Inference, metrics calculation, and Evidently AI reporting
- observability: Structured logging and metrics tracking
"""

__version__ = "1.0.0"
