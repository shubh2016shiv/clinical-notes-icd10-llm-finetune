"""
OpenAI JSON generation client for the v3 pipeline.
"""

from __future__ import annotations

import os

from clinical_note_generation_v3.infrastructure.llm_provider.fallback_llm_client import (
    parse_json_from_provider_response,
)


class OpenAIJSONClient:
    """
    OpenAI JSON client via the chat completions API.
    """

    provider_name = "openai"

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI JSON generation.")

        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key)

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
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
