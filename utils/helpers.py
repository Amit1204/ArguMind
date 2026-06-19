"""Shared utilities."""
import time
import functools
from utils.logger import get_logger

logger = get_logger(__name__)


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} took {time.perf_counter() - start:.2f}s")
        return result
    return wrapper


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
