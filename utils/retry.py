"""
Retry helper for Groq's per-minute rate limits (TPM).

Groq's free tier caps tokens-per-minute per model (e.g. 6000 TPM for
llama-3.1-8b-instant). ArguMind's pipeline fires 20-30+ LLM calls per query
(claim extraction per paper, cluster labeling per cluster, consensus, critic,
response), which can blow through the per-minute cap even when the daily
quota is nowhere near exhausted. Previously, hitting a 429 here meant the
caller silently fell back to a low-quality placeholder (e.g. ConsensusAgent's
"strength=weak" fallback) — even though the request would have succeeded a
couple seconds later.

invoke_with_retry() parses the "Please try again in Xs" hint Groq includes in
the 429 error body, sleeps that long (plus a small buffer), and retries
in-process rather than giving up immediately.
"""
import re
import time

from utils.logger import get_logger

logger = get_logger(__name__)

_WAIT_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)
_RATE_LIMIT_MARKERS = ("429", "rate_limit_exceeded", "tokens per minute", "TPM")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def invoke_with_retry(chain, inputs: dict, max_retries: int = 3, default_wait: float = 3.0):
    """
    Invoke a LangChain chain, retrying on Groq TPM rate-limit (429) errors.

    Non-rate-limit exceptions are raised immediately (callers already have
    their own fallback logic for genuine failures).
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_exc = e
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            match = _WAIT_RE.search(str(e))
            wait = float(match.group(1)) + 0.5 if match else default_wait
            logger.warning(
                f"Rate limited (attempt {attempt + 1}/{max_retries}) — "
                f"waiting {wait:.1f}s before retry"
            )
            time.sleep(wait)
    raise last_exc
