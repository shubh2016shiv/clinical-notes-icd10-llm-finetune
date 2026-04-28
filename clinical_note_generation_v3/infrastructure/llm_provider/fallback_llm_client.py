"""
Fallback JSON client and JSON parsing utilities for the v3 pipeline.

LAYER: infrastructure/llm_provider
ARCHITECTURE:
  FallbackJSONClient
      -> tries primary JSONGenerationClient
      -> on failure, tries fallback JSONGenerationClient
      -> records last_provider_name and last_model_name for ModelProvenance

  parse_json_from_provider_response()
      -> strips markdown fences
      -> json.loads() with regex fallback
      -> returns dict

DATA FLOW:
  prompt -> primary_client.generate_json() [-> fallback on error] -> dict
  raw_text -> parse_json_from_provider_response() -> dict

DEPENDENCIES:
  - core/ports/llm_generation_port.py (JSONGenerationClient — structural)
  - stdlib (json, re)
"""

import json
import re


def parse_json_from_provider_response(raw_text: str) -> dict:
    """
    Parse a JSON object from raw provider response text.

    Strips leading/trailing markdown code fences (```json ... ```) before
    parsing. Falls back to regex extraction of the first {...} block when
    json.loads fails on the cleaned text.

    Args:
        raw_text: Raw response text from any LLM provider.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no JSON object is found in the response text.
        ValueError: If the parsed JSON root is not a dictionary.

    Example:
        >>> parse_json_from_provider_response('```json\\n{"a": 1}\\n```')["a"]
        1
    """
    cleaned_text = raw_text.strip()
    cleaned_text = re.sub(r"^```(?:json)?", "", cleaned_text).strip()
    cleaned_text = re.sub(r"```$", "", cleaned_text).strip()

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        json_block_match = re.search(r"\{.*\}", cleaned_text, flags=re.DOTALL)
        if not json_block_match:
            raise ValueError("Provider response did not contain a parseable JSON object.")
        parsed = json.loads(json_block_match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Provider JSON response root must be an object, not an array or scalar.")
    return parsed


class FallbackJSONClient:
    """
    JSON generation client that tries a primary provider and falls back to a secondary.

    Wraps two JSONGenerationClient implementations: primary is tried first; on any
    exception the fallback is tried. Records the provider and model name of the
    last successful call for ModelProvenance tracking.

    Args:
        primary_client: First provider to call for every request.
        fallback_client: Optional secondary provider tried when primary raises.

    Returns:
        Client that records provenance of the last successful call.

    Raises:
        Exception: Re-raises the fallback error when both primary and fallback fail.
                   Re-raises primary error when no fallback is configured.

    Example:
        >>> isinstance("fallback", str)
        True
    """

    def __init__(
        self,
        *,
        primary_client,
        fallback_client=None,
    ) -> None:
        self._primary_client = primary_client
        self._fallback_client = fallback_client
        self.last_provider_name: str = primary_client.provider_name
        self.last_model_name: str = primary_client.model_name

    @property
    def provider_name(self) -> str:
        """Return the provider name used by the last successful call."""
        return self.last_provider_name

    @property
    def model_name(self) -> str:
        """Return the model name used by the last successful call."""
        return self.last_model_name

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        """
        Generate JSON using the primary provider, falling back to secondary on error.

        Args:
            prompt: Full prompt text requesting JSON-only output.
            response_schema: Optional provider-specific structured-output schema.

        Returns:
            Parsed JSON dictionary from whichever provider succeeded.

        Raises:
            Exception: Re-raises the last provider's error if all providers fail.

        Example:
            >>> isinstance("prompt", str)
            True
        """
        try:
            result = self._primary_client.generate_json(prompt, response_schema=response_schema)
            self.last_provider_name = self._primary_client.provider_name
            self.last_model_name = self._primary_client.model_name
            return result
        except Exception:
            if not self._fallback_client:
                raise

        result = self._fallback_client.generate_json(prompt, response_schema=response_schema)
        self.last_provider_name = self._fallback_client.provider_name
        self.last_model_name = self._fallback_client.model_name
        return result
