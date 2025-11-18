"""
Clients module for clinical notes generation.

Contains API client classes for interacting with external services.
"""

from .gemini_api_client import GeminiAPIClient
from .openai_api_client import OpenAIAPIClient

__all__ = ["GeminiAPIClient", "OpenAIAPIClient"]

