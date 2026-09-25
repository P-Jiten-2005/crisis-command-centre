"""Centralised application configuration.

All values are read from environment variables (optionally loaded from a
``.env`` file via python-dotenv) and validated with Pydantic. No secrets are
hardcoded anywhere in the code base.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

PROJECT_ROOT: Path = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


class ConfigurationError(RuntimeError):
    """Raised when a required configuration value is missing or invalid."""


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        The parsed boolean value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Validated runtime settings for the Crisis Command Center."""

    groq_api_key: SecretStr = Field(default=SecretStr(""), description="Groq API key.")
    groq_model: str = Field(default="openai/gpt-oss-120b")
    llm_reasoning_effort: str | None = Field(
        default="low", description="Reasoning effort for reasoning models (low/medium/high); empty to disable."
    )
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=4, ge=0, le=10)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    database_url: str = Field(default="", description="SQLAlchemy database URL.")

    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "chroma_data")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    rag_top_k_incidents: int = Field(default=3, ge=1, le=10)
    rag_top_k_protocols: int = Field(default=2, ge=1, le=10)
    rag_min_similarity: float = Field(default=0.2, ge=0.0, le=1.0)
    auto_ingest: bool = Field(default=True)

    max_report_chars: int = Field(default=10_000, ge=100)
    injection_threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    memory_window: int = Field(default=5, ge=0, le=50)
    log_level: str = Field(default="INFO")
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a :class:`Settings` instance from environment variables.

        Returns:
            Validated settings.
        """
        persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_data")))
        if not persist_dir.is_absolute():
            persist_dir = (PROJECT_ROOT / persist_dir).resolve()
        return cls(
            groq_api_key=SecretStr(os.getenv("GROQ_API_KEY", "")),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low").strip().lower() or None,
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "4")),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            database_url=os.getenv("DATABASE_URL", ""),
            chroma_persist_dir=persist_dir,
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            rag_top_k_incidents=int(os.getenv("RAG_TOP_K_INCIDENTS", "3")),
            rag_top_k_protocols=int(os.getenv("RAG_TOP_K_PROTOCOLS", "2")),
            rag_min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.2")),
            auto_ingest=_env_bool("AUTO_INGEST", True),
            max_report_chars=int(os.getenv("MAX_REPORT_CHARS", "10000")),
            injection_threshold=float(os.getenv("INJECTION_THRESHOLD", "0.7")),
            memory_window=int(os.getenv("MEMORY_WINDOW", "5")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            cors_allow_origins=[o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()],
        )

    def require_groq_key(self) -> str:
        """Return the Groq API key or fail loudly if it is not configured.

        Returns:
            The raw API key string.

        Raises:
            ConfigurationError: If ``GROQ_API_KEY`` is not set.
        """
        key = self.groq_api_key.get_secret_value()
        if not key:
            raise ConfigurationError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return key

    def require_database_url(self) -> str:
        """Return the database URL or fail loudly if it is not configured.

        Returns:
            The SQLAlchemy database URL.

        Raises:
            ConfigurationError: If ``DATABASE_URL`` is not set.
        """
        if not self.database_url:
            raise ConfigurationError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
        return self.database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance.

    Returns:
        The singleton :class:`Settings`.
    """
    return Settings.from_env()
