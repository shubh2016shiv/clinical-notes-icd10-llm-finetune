"""
infrastructure — concrete adapters for external systems and resources.

Sub-packages:
  data_preprocessing/   — ICD order-file parser and in-memory repository
  llm_provider/         — Gemini, DeepSeek, and fallback JSON LLM clients
  embedding_provider/   — OpenAI, Gemini, and deterministic hash embedding clients
  vector_store/         — FAISS-backed vector index with GPU/CPU/NumPy fallback

Nothing in infrastructure/ may be imported by core/.
"""
