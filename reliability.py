"""Reliability layer for LLM calls.

Production agents fail for boring reasons: rate limits, timeouts, brief
outages. Two techniques absorb almost all of it:

- **Retry with exponential backoff and jitter** for transient errors
  (429s, timeouts, 5xx). Jitter randomizes the delay so concurrent
  agents don't retry in lockstep and re-trigger the limit.
- **Model fallback cascade** — if the primary model keeps failing, try
  the configured fallbacks in order before giving up.

Non-transient errors (auth, malformed request) fail fast: retrying or
switching models cannot fix a bad API key.
"""

import random
import time
from typing import Callable

# Substrings that mark an error as transient (safe to retry)
TRANSIENT_MARKERS = (
    "rate limit",
    "429",
    "timeout",
    "timed out",
    "connection",
    "overloaded",
    "temporarily",
    "service unavailable",
    "server error",
    "502",
    "503",
    "504",
)


def is_transient_error(error: Exception) -> bool:
    """Whether an exception looks like a transient service condition."""
    text = str(error).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def retry_call(
    fn: Callable,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep=time.sleep,
    jitter: bool = True,
):
    """Call ``fn`` retrying transient failures with capped backoff.

    Args:
        fn: The call to execute.
        attempts: Total attempts (retry + first try). 3-5 is the usual
            production range; beyond that the outage is probably not
            transient.
        base_delay: Seconds before the first retry.
        max_delay: Ceiling on any single wait.
        sleep: Injected for tests.
        jitter: Randomize delays 50-150% so parallel agents decorrelate.

    Raises:
        The last exception, after attempts are exhausted or immediately
        for non-transient errors.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if not is_transient_error(e) or attempt == attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if jitter:
                delay *= 0.5 + random.random()
            sleep(delay)
    raise last_error
