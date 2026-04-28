"""
Diagnostics — API health checks and provider availability scripts.

LAYER: diagnostics
ARCHITECTURE:
  check_gemini_api_models.py  -> Gemini model availability table
  check_openai_api_models.py  -> OpenAI model availability table
  check_deepseek_api_models.py -> DeepSeek model ping

DEPENDENCY RULE:
  diagnostics/ may import from infrastructure/ and config/ only.
  diagnostics/ MUST NEVER import from core/ or application/.
"""
