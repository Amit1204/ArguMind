"""
LLM factory — Groq → Gemini → OpenAI fallback.

_skip_groq flag allows the orchestrator to bypass Groq at runtime
when a 429 rate-limit error is hit, without restarting the process.
"""
from functools import lru_cache
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Set to True at runtime when Groq hits its daily token limit
_skip_groq = False


def skip_groq():
    """Call this when Groq returns a 429. Clears cache so next get_llm() returns Gemini."""
    global _skip_groq
    _skip_groq = True
    get_llm.cache_clear()
    logger.warning("Groq marked as rate-limited — next LLM call will use Gemini")


@lru_cache(maxsize=1)
def get_llm():
    """Returns the active LLM. Cached until skip_groq() clears it."""

    # 1. Groq (fast, free — 100k tokens/day)
    if settings.groq_api_key and not _skip_groq:
        from langchain_groq import ChatGroq
        logger.info(f"Using Groq: {settings.groq_model}")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.1,
        )

    # 2. Gemini via Google's OpenAI-compatible endpoint (no langchain-google-genai needed)
    if settings.gemini_api_key:
        from langchain_openai import ChatOpenAI
        logger.info(f"Using Gemini (OpenAI-compat): {settings.gemini_model}")
        return ChatOpenAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.1,
        )

    # 3. OpenAI (last resort)
    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        logger.info(f"Using OpenAI: {settings.openai_model}")
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )

    raise RuntimeError(
        "No LLM configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env"
    )
