"""
DeepSeek API model availability diagnostic.

LAYER: diagnostics
ARCHITECTURE:
  V3PipelineSettings (deepseek_model, deepseek_base_url, deepseek_api_key)
      -> OpenAI-compatible chat completions ping
      -> stdout result (no markdown output needed)

DATA FLOW:
  settings -> DeepSeek chat completion -> response preview -> stdout

DEPENDENCIES:
  - config/settings (V3PipelineSettings)
  - openai SDK (external)
"""

from openai import OpenAI

from clinical_note_generation_v3.config.settings import V3PipelineSettings


def main() -> int:
    """Ping the configured DeepSeek chat model and print the response preview."""
    settings = V3PipelineSettings()
    if not settings.deepseek_api_key:
        print(
            "Error: No DeepSeek API key found. "
            "Set DEEPSEEK_API_KEY, DEEP_SEEK_API_KEY, or DEEP_SEEK in your environment or .env."
        )
        return 1

    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": 'Return JSON: {"ok": true}'}],
            temperature=0,
            max_tokens=32,
        )
    except Exception as error:
        print(f"DeepSeek model check failed for '{settings.deepseek_model}': {error}")
        return 1

    response_content = response.choices[0].message.content
    print(f"DeepSeek model OK: {settings.deepseek_model}")
    print(f"Response preview: {response_content[:80] if response_content else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
