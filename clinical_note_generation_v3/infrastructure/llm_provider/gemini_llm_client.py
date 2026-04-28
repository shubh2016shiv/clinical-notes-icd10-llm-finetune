"""
Gemini JSON generation client.

LAYER: infrastructure/llm_provider
ARCHITECTURE:
  GeminiJSONClient
      <- google.generativeai SDK
      -> implements JSONGenerationClient protocol (core/ports/)
      -> generate_json() -> dict

  Supports structured output via Gemini's response_schema parameter to improve
  JSON syntax adherence. Pydantic validation in the pipeline remains the final
  reliability gate regardless of structured output.

DATA FLOW:
  prompt + optional response_schema -> Gemini API -> parse_json_from_provider_response() -> dict

DEPENDENCIES:
  - core/ports/llm_generation_port.py (JSONGenerationClient — structural)
  - infrastructure/llm_provider/fallback_llm_client.py (parse_json_from_provider_response)
  - google.generativeai (external)
  - stdlib (os)
"""

import os

from clinical_note_generation_v3.infrastructure.llm_provider.fallback_llm_client import (
    parse_json_from_provider_response,
)


class GeminiJSONClient:
    """
    Gemini JSON generation client using the configured generation model.

    Requests JSON-only output via response_mime_type="application/json" and
    optionally constrains the schema via Gemini's structured-output feature.

    Args:
        model_name: Gemini model name (e.g., "models/gemini-2.5-flash-preview-04-17").
        api_key: Optional API key. Falls back to GEMINI_API_KEY environment variable.

    Returns:
        Client ready for JSON-focused Gemini calls.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not available via arg or environment.

    Example:
        >>> client = GeminiJSONClient("models/gemini-2.5-flash-preview-04-17", api_key="dummy")
        >>> client.model_name
        'models/gemini-2.5-flash-preview-04-17'
    """

    provider_name = "gemini"

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for Gemini generation. "
                "Set it in your environment or .env file."
            )

        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(model_name)

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        """
        Generate and return a parsed JSON object from Gemini.

        Args:
            prompt: Full prompt text requesting JSON-only output.
            response_schema: Optional Gemini structured-output schema dict.
                             When provided, Gemini enforces this schema server-side.

        Returns:
            Parsed JSON dictionary from the Gemini response.

        Raises:
            ValueError: If the response text cannot be parsed as a JSON object.
            Exception: Propagates Gemini API errors (rate limits, auth failures, etc.).

        Example:
            >>> isinstance("prompt", str)
            True
        """
        generation_config: dict = {
            "temperature": 0,
            "response_mime_type": "application/json",
        }
        if response_schema is not None:
            generation_config["response_schema"] = response_schema

        response = self._model.generate_content(prompt, generation_config=generation_config)
        return parse_json_from_provider_response(response.text)
