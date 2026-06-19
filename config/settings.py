"""Central configuration for ArguMind."""
from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve .env relative to this file, not the process's cwd — Streamlit can be
# launched from any directory, and a relative "env_file" silently fails to load.
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    # LLM — Groq priority → Gemini → OpenAI
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Google Gemini (fallback)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # OpenAI (last resort)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # arXiv
    arxiv_max_results: int = 5
    arxiv_timeout: int = 30

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    chroma_persist_dir: str = "./outputs/chroma_db"

    # Agent behaviour
    max_iterations: int = 2          # critic retry limit
    confidence_threshold: float = 0.6
    min_evidence_count: int = 2

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = str(ENV_FILE)
        extra = "ignore"


settings = Settings()
