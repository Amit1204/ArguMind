"""
Retry helper for Groq's rate limits.

Groq's free tier has TWO separate caps that both return 429s:
  - Per-minute (TPM): e.g. 6000 TPM for llama-3.1-8b-instant. Recovers in
    seconds, so it's worth sleeping and retrying in-process.
  - Per-day (TPD): e.g. 100k tokens/day for llama-3.3-70b-versatile. Once hit,
    the wait hint is minutes-to-hours (e.g. "try again in 8m14.208s") — far
    too long to sleep through mid-request, and futile to retry against since
    the quota won't replenish.

Previously this module only handled the TPM case, and its wait-time regex
only matched plain-seconds hints ("try again in 4.2s"). Groq's TPD error
messages use a minute+second format ("8m14.208s") that the old regex didn't
match at all, so every TPD 429 fell back to a useless 3s default wait,
retried 3 times, and still failed — multiplied across every one of
ArguMind's 20-30+ LLM calls per query. That's the "everything is rate
limited" cascades seen in production logs.

Fix: parse both wait formats correctly, and treat TPD specifically — instead
of retrying with a short sleep, immediately flip llm_factory.skip_groq()
so every *subsequent* LLM call in this process (including calls from other
users sharing this same deployed Space's single Groq key) routes to Gemini
instead of repeatedly hammering an exhausted daily quota. The call that
triggered the TPD error still fails (its chain is already bound to Groq) —
callers already have fallback logic for that — but it stops wasting ~10s
of pointless retries per call, and every later call in the process recovers
immediately instead of repeating the same dead end.
"""
import re
import time

from utils.logger import get_logger

logger = get_logger(__name__)

# Matches "try again in 8m14.208s" (TPD) or "try again in 4.2s" (TPM) —
# the minute component is optional.
_WAIT_RE = re.compile(r"try again in(?: (\d+)m)? ?([\d.]+)s", re.IGNORECASE)
_RATE_LIMIT_MARKERS = ("429", "rate_limit_exceeded", "tokens per minute", "TPM", "tokens per day", "TPD")
_DAILY_CAP_MARKERS = ("tokens per day", "TPD")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def _is_daily_cap_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _DAILY_CAP_MARKERS)


def _parse_wait_seconds(msg: str, default: float) -> float:
    match = _WAIT_RE.search(msg)
    if not match:
        return default
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds + 0.5


def invoke_with_retry(chain, inputs: dict, max_retries: int = 3, default_wait: float = 3.0):
    """
    Invoke a LangChain chain, handling Groq rate-limit (429) errors.

    TPM (per-minute) errors: sleep the parsed wait time and retry in-process,
    up to max_retries times.

    TPD (per-day) errors: not retried (the wait is minutes-to-hours). Instead,
    immediately marks Groq as exhausted process-wide via skip_groq() so future
    calls fall through to Gemini, then re-raises for the caller's existing
    fallback logic to handle this specific call.

    Non-rate-limit exceptions are raised immediately.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_exc = e
            if not _is_rate_limit_error(e):
                raise

            msg = str(e)

            if _is_daily_cap_error(e):
                logger.warning(
                    "Groq daily token cap (TPD) hit — switching future calls "
                    f"in this process to Gemini fallback. Error: {msg[:200]}"
                )
                try:
                    from agents.llm_factory import skip_groq
                    skip_groq()
                except Exception:
                    logger.warning("Could not switch to Gemini fallback (skip_groq failed)")
                raise

            if attempt == max_retries:
                raise
            wait = _parse_wait_seconds(msg, default_wait)
            logger.warning(
                f"Rate limited (attempt {attempt + 1}/{max_retries}) — "
                f"waiting {wait:.1f}s before retry"
            )
            time.sleep(wait)
    raise last_exc
