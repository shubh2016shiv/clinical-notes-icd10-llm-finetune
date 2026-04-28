"""
Centralized settings for the v3 seeded clinical note quality pipeline.

LAYER: config
ARCHITECTURE:
  .env / process environment
      -> V3PipelineSettings (Pydantic BaseSettings, validated on load)
      -> repository / embedding / LLM clients / artifact paths

DEPENDENCIES:
  - stdlib (os, pathlib)
  - pydantic-settings
  - python-dotenv
"""

from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def read_secret_from_environment(secret_name: str) -> str | None:
    """
    Read a secret from the process environment or project .env without retaining it.

    Args:
        secret_name: Environment variable name to read.

    Returns:
        Secret value string, or None if not found.

    Raises:
        None.

    Example:
        >>> read_secret_from_environment("__MISSING_SECRET__") is None
        True
    """
    import os

    process_value = os.getenv(secret_name)
    if process_value:
        return process_value
    dotenv_map = dotenv_values(PROJECT_ROOT / ".env")
    raw_value = dotenv_map.get(secret_name)
    return str(raw_value) if raw_value else None


class V3PipelineSettings(BaseSettings):
    """
    Central environment-driven settings for the v3 synthetic data pipeline.

    Loaded from environment variables prefixed with CLINICAL_V3_ or from the
    project-root .env file. No secrets are stored in this object beyond what
    pydantic-settings loads at construction time.

    Args:
        official_icd_directory: Directory containing official ICD-10-CM April 1 2026 files.
        icd_order_filename: Official order-file name used as source of truth.
        faiss_persist_directory: Local path for persisted FAISS index and metadata.
        prefer_faiss_gpu: Whether to prefer GPU FAISS when available.
        bundle_template_directory: Directory containing active clinical bundle templates.
        sample_data_directory: Directory for tracked quality-check artifacts.
        raw_run_directory: Directory for ignored raw run logs.
        gemini_generation_model: Gemini model for note generation and evaluation.
        gemini_embedding_model: Gemini embedding model for FAISS retrieval.
        openai_embedding_model: OpenAI embedding model for FAISS retrieval.
        openai_generation_model: OpenAI fallback model for JSON generation.
        deepseek_model: Primary DeepSeek model identifier.
        deepseek_base_url: OpenAI-compatible DeepSeek API base URL.
        candidate_count: Number of ICD candidates to pass to the ICD resolver.
        quality_sample_count: Number of bundle-driven note examples to generate per run.
        fail_on_quality_flags: Reserved compatibility flag from earlier iterations.
        random_seed: Default deterministic seed for reproducibility.

    Returns:
        A settings object validated by Pydantic on construction.

    Raises:
        pydantic.ValidationError: If environment overrides have invalid types or values.

    Example:
        >>> settings = V3PipelineSettings()
        >>> settings.official_icd_order_path.suffix
        '.txt'
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="CLINICAL_V3_",
        case_sensitive=False,
        extra="ignore",
    )

    official_icd_directory: Path = Field(
        default=PROJECT_ROOT / "official_icd10cm_2026_april_1",
        description="Directory with official ICD-10-CM April 1 2026 files.",
    )
    bundle_template_directory: Path = Field(
        default=PROJECT_ROOT / "clinical_note_generation_v3" / "data" / "bundle_templates",
        description="Directory containing active clinical bundle template JSONL files.",
    )
    icd_order_filename: str = Field(
        default="icd10cm-order-April-1-2026.txt",
        description="Official ICD-10-CM order file name.",
    )
    faiss_persist_directory: Path = Field(
        default=PROJECT_ROOT / "clinical_note_generation_v3" / ".faiss" / "icd10cm_2026_april_1",
        description="Persistent FAISS directory for ICD candidate retrieval.",
    )
    prefer_faiss_gpu: bool = Field(
        default=True,
        description="Use GPU FAISS when available.",
    )
    faiss_index_batch_size: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Embedding batch size used by the FAISS ICD index build CLI.",
    )
    sample_data_directory: Path = Field(
        default=PROJECT_ROOT / "clinical_note_generation_v3" / "sample_data",
        description="Tracked sample data and human-review artifacts.",
    )
    raw_run_directory: Path = Field(
        default=PROJECT_ROOT / "clinical_note_generation_v3" / "runs",
        description="Ignored raw run logs and intermediate artifacts.",
    )
    gemini_generation_model: str = Field(
        default="models/gemini-3.1-flash-lite-preview",
        description="Primary Gemini model for generation and selection.",
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-2-preview",
        description="Gemini embedding model for FAISS retrieval.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model for FAISS retrieval.",
    )
    openai_generation_model: str = Field(
        default_factory=lambda: read_secret_from_environment("OPENAI_MODEL") or "gpt-4o-mini",
        description="OpenAI fallback model for v3 JSON generation.",
    )
    deepseek_model: str = Field(
        default="deepseek-v4-flash",
        description="Primary DeepSeek model identifier.",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="OpenAI-compatible DeepSeek API base URL.",
    )
    candidate_count: int = Field(
        default=60,
        ge=10,
        le=150,
        description="Default number of ICD candidate codes passed to the selector.",
    )
    quality_sample_count: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of quality-check examples to generate per run.",
    )
    fail_on_quality_flags: bool = Field(
        default=False,
        description="Treat selector quality flags as run failures when True.",
    )
    random_seed: int = Field(
        default=42,
        description="Default deterministic seed for reproducible sampling.",
    )
    minimum_note_character_count: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Minimum generated note length required by deterministic pre-checks.",
    )
    near_duplicate_similarity_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Threshold for deterministic near-duplicate rejection.",
    )
    accept_score_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Combined score threshold for direct acceptance.",
    )
    revise_score_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Combined score threshold for revision eligibility.",
    )
    reject_below_score_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Combined score below which notes are rejected instead of revised.",
    )
    max_revision_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of revision attempts allowed per note.",
    )
    recent_accepted_note_window_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Rolling history size used for deterministic near-duplicate checks.",
    )
    gemini_api_key: str | None = Field(
        default_factory=lambda: read_secret_from_environment("GEMINI_API_KEY"),
        description="Gemini API key loaded from environment or .env at construction time.",
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: read_secret_from_environment("OPENAI_API_KEY"),
        description="OpenAI API key loaded from environment or .env at construction time.",
    )
    deepseek_api_key: str | None = Field(
        default_factory=lambda: (
            read_secret_from_environment("DEEPSEEK_API_KEY")
            or read_secret_from_environment("DEEP_SEEK_API_KEY")
            or read_secret_from_environment("DEEP_SEEK")
        ),
        description="DeepSeek API key loaded from environment or .env at construction time.",
    )

    @property
    def official_icd_order_path(self) -> Path:
        """
        Return the absolute path to the official ICD-10-CM order file.

        Returns:
            Combined path of official_icd_directory / icd_order_filename.

        Raises:
            None.

        Example:
            >>> V3PipelineSettings().official_icd_order_path.name
            'icd10cm-order-April-1-2026.txt'
        """
        return self.official_icd_directory / self.icd_order_filename

    @property
    def training_artifact_output_path(self) -> Path:
        """
        Default JSONL path for accepted-note training rows.
        """
        return self.sample_data_directory / "training_rows.jsonl"

    @property
    def audit_artifact_output_path(self) -> Path:
        """
        Default JSONL path for accepted and rejected audit rows.
        """
        return self.sample_data_directory / "audit_rows.jsonl"

    @property
    def batch_metrics_output_path(self) -> Path:
        """
        Default JSON path for aggregate batch metrics.
        """
        return self.sample_data_directory / "batch_metrics.json"
