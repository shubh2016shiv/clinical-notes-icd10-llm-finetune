"""
Clients Layer - LLM API Client Abstractions

This layer provides clean abstractions over LLM providers (Gemini, OpenAI),
enabling the rest of the system to work with any provider interchangeably.

Submodules:
    llm_client.py → Protocol and base implementation
    gemini_client.py → Google Gemini implementation
    openai_client.py → OpenAI implementation

Why Abstraction Layer:
    1. Swappable providers without changing business logic
    2. Centralized rate limiting and retry logic
    3. Testability via mock implementations
    4. Single place for API configuration

Author: Shubham Singh
Date: December 2025
"""

from clinical_note_generation.clients.llm_client import (
    LLMClientProtocol,
    BaseLLMClient,
)
from clinical_note_generation.clients.gemini_client import GeminiClient
from clinical_note_generation.clients.openai_client import OpenAIClient

__all__ = [
    "LLMClientProtocol",
    "BaseLLMClient",
    "GeminiClient",
    "OpenAIClient",
]
