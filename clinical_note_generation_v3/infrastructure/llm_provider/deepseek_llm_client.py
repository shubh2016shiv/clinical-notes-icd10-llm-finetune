"""
DeepSeek fallback JSON generation client via OpenAI-compatible API.

LAYER: infrastructure/llm_provider
ARCHITECTURE:
  DeepSeekJSONClient
      <- openai SDK (OpenAI-compatible base_url)
      -> implements JSONGenerationClient protocol (core/ports/)
      -> generate_json() -> dict

  DeepSeek does not support Gemini-style response_schema. The schema must be
  included in the prompt text for the model to follow. The response_schema
  parameter is accepted but intentionally ignored.

DATA FLOW:
  prompt -> DeepSeek chat completions (json_object mode) -> parse_json_from_provider_response() -> dict

DEPENDENCIES:
  - core/ports/llm_generation_port.py (JSONGenerationClient — structural)
  - infrastructure/llm_provider/fallback_llm_client.py (parse_json_from_provider_response)
  - openai SDK (external)
  - stdlib (os)
"""

import os

from clinical_note_generation_v3.infrastructure.llm_provider.fallback_llm_client import (
    parse_json_from_provider_response,
)


class DeepSeekJSONClient:
    """
    DeepSeek fallback client via the OpenAI-compatible chat completions API.

    Used as the fallback when Gemini is unavailable. Operates in JSON mode via
    response_format={"type": "json_object"} rather than provider-level schemas.

    Args:
        model_name: DeepSeek model name (e.g., "deepseek-v4-flash").
        base_url: OpenAI-compatible DeepSeek API base URL.
        api_key: Optional API key. Falls back to DEEPSEEK_API_KEY, DEEP_SEEK_API_KEY,
                 or DEEP_SEEK environment variables in that order.

    Returns:
        Client for fallback JSON generation.

    Raises:
        RuntimeError: If no DeepSeek API key is found.

    Example:
        >>> client = DeepSeekJSONClient("deepseek-v4-flash", "https://api.deepseek.com", api_key="dummy")
        >>> client.provider_name
        'deepseek'
    """

    provider_name = "deepseek"

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("DEEP_SEEK_API_KEY")
            or os.getenv("DEEP_SEEK")
        )
        if not self._api_key:
            raise RuntimeError(
                "A DeepSeek API key is required. Set DEEPSEEK_API_KEY, "
                "DEEP_SEEK_API_KEY, or DEEP_SEEK in your environment or .env file."
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key, base_url=base_url)

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        """
        Generate and return a parsed JSON object from DeepSeek.

        Args:
            prompt: Full prompt text requesting JSON-only output. Include the
                    required JSON schema in the prompt for schema compliance.
            response_schema: Ignored — DeepSeek does not support provider-level schemas.

        Returns:
            Parsed JSON dictionary from the DeepSeek response.

        Raises:
            ValueError: If the response cannot be parsed as a JSON object.
            Exception: Propagates DeepSeek API errors.

        Example:
            >>> isinstance("prompt", str)
            True
        """
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON. Do not include markdown fences. "
                        "The user prompt contains the required JSON schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        response_text = response.choices[0].message.content or ""
        return parse_json_from_provider_response(response_text)
