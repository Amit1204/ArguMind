"""
Retry helper for Groq's rate limits.

Groq's free tier has TWO separate caps that both return 429s:
  - Per-minute (TPM): e.g. 6000 TPM for llama-3.1-8b-instant. Recovers in
    seconds.
  - Per-day (TPD): e.g. 100k tokens/day for llama-3.3-70b-versatile. The wait
    hint here is minutes (e.g. "try again in 8m14.208s") rather than seconds.

Both cases are handled the same way: parse Groq's "try again in Xm Y.Zs"
hint and sleep that long before retrying in-process, rather than falling
back to a different provider. The old wait-time regex only matched
plain-seconds hints ("4.2s") and didn't understand the minute+second format
Groq uses for TPD errors ("8m14.208s") — that caused every TPD 429 to fall
back to a useless 3s default wait and fail 3x in a row. That's now fixed:
the regex parses both formats, so a TPD hit waits out the real cooldown
(however many minutes Groq reports) and then retries against Groq again,
instead of switching to Gemini.

Note: with TPD errors this means a single query can block for several
minutes if the daily quota is genuinely exhausted — that's intentional
per the current configuration (stay on Groq rather than degrade to a
different model).
"""
import re
import time

from utils.logger import get_logger

logger = get_logger(__name__)

# Matches "try again in 8m14.208s" (TPD) or "try again in 4.2s" (TPM) —
# the minute component is optional.
_WAIT_RE = re.compile(r"try again in(?: (\d+)m)? ?([\d.]+)s", re.IGNORECASE)
_RATE_LIMIT_MARKERS = ("429", "rate_limit_exceeded", "tokens per minute", "TPM", "tokens per day", "TPD")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def _parse_wait_seconds(msg: str, default: float) -> float:
    match = _WAIT_RE.search(msg)
    if not match:
        return default
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds + 0.5


def invoke_with_retry(chain, inputs: dict, max_retries: int = 3, default_wait: float = 3.0):
    """
    Invoke a LangChain chain, retrying on Groq rate-limit (429) errors.

    Sleeps the wait time Groq reports (correctly parsed for both per-minute
    and per-day hints) and retries against the same provider, up to
    max_retries times. Non-rate-limit exceptions are raised immediately.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_exc = e
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            wait = _parse_wait_seconds(str(e), default_wait)
            logger.warning(
                f"Rate limited (attempt {attempt + 1}/{max_retries}) — "
                f"waiting {wait:.1f}s before retry"
            )
            time.sleep(wait)
    raise last_exc
