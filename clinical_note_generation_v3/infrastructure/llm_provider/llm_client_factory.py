"""
Default v3 JSON LLM wiring.
"""

from __future__ import annotations

import logging

from clinical_note_generation_v3.config.settings import V3PipelineSettings
from clinical_note_generation_v3.infrastructure.llm_provider.deepseek_llm_client import (
    DeepSeekJSONClient,
)
from clinical_note_generation_v3.infrastructure.llm_provider.fallback_llm_client import (
    FallbackJSONClient,
)
from clinical_note_generation_v3.infrastructure.llm_provider.openai_llm_client import (
    OpenAIJSONClient,
)

logger = logging.getLogger(__name__)


def create_default_json_generation_client(
    settings: V3PipelineSettings | None = None
) -> FallbackJSONClient:
    """
    Create the default v3 JSON client: DeepSeek primary, OpenAI fallback.
    """
    resolved_settings = settings or V3PipelineSettings()
    try:
        primary_client = DeepSeekJSONClient(
            api_key=resolved_settings.deepseek_api_key,
            model_name=resolved_settings.deepseek_model,
            base_url=resolved_settings.deepseek_base_url,
        )
    except Exception as error:
        logger.warning(
            "DeepSeek JSON client could not be initialized; using OpenAI only: %s",
            error,
        )
        fallback_client = OpenAIJSONClient(
            api_key=resolved_settings.openai_api_key,
            model_name=resolved_settings.openai_generation_model,
        )
        return FallbackJSONClient(primary_client=fallback_client)

    fallback_client = OpenAIJSONClient(
        api_key=resolved_settings.openai_api_key,
        model_name=resolved_settings.openai_generation_model,
    )
    return FallbackJSONClient(
        primary_client=primary_client,
        fallback_client=fallback_client,
    )
