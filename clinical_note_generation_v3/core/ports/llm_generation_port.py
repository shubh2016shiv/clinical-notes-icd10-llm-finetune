"""
LLM generation provider port (interface) for the v3 pipeline.

LAYER: core/ports
ARCHITECTURE:
  core/services depends on this protocol for concept extraction.
  application/note_generation depends on this protocol for generation and selection.
  Concrete implementations live in infrastructure/llm_provider/.

  JSONGenerationClient (Protocol)
      <- GeminiJSONClient   (infrastructure/llm_provider/)
      <- DeepSeekJSONClient (infrastructure/llm_provider/)
      <- FallbackJSONClient (infrastructure/llm_provider/, wraps above two)

DATA FLOW:
  prompt + optional response_schema -> JSONGenerationClient -> dict

DEPENDENCIES:
  - typing (Protocol, runtime_checkable)
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class JSONGenerationClient(Protocol):
    """
    Structural protocol for LLM providers that return JSON dictionaries.

    All implementations must expose provider_name and model_name attributes so
    ModelProvenance can record which provider serviced each pipeline step.

    Note:
        Use this protocol as the type annotation for pipeline components that
        call LLMs, so concrete providers are swappable without modifying business
        logic. Annotate constructor parameters with this protocol, not with
        GeminiJSONClient or FallbackJSONClient directly.
    """

    @property
    def provider_name(self) -> str:
        """
        Provider that serviced the most recent successful generation call.
        """
        ...

    @property
    def model_name(self) -> str:
        """
        Model that serviced the most recent successful generation call.
        """
        ...

    def generate_json(self, prompt: str, response_schema: dict | None = None) -> dict:
        """
        Generate and return a parsed JSON object from the LLM.

        Args:
            prompt: Full prompt text requesting JSON-only output.
            response_schema: Optional provider-specific structured-output schema.
                             Ignored by providers that do not support structured output.

        Returns:
            Parsed JSON dictionary from the provider response.

        Raises:
            ValueError: If the response cannot be parsed as a JSON object.
            Exception: Propagates provider-specific network or API errors.
        """
        ...
