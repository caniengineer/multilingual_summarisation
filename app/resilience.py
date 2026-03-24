import time
import asyncio
import random
import anthropic
from app.logging_config import get_logger

logger = get_logger(__name__)


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and rejecting calls."""
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30, window: int = 60):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._window = window
        self._failures: list[float] = []
        self._opened_at: float | None = None
        self._state = "closed"

    @property
    def state(self) -> str:
        if self._state == "open" and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self._recovery_timeout:
                return "half_open"
        return self._state

    async def call(self, func, *args, **kwargs):
        current_state = self.state

        if current_state == "open":
            raise CircuitOpenError(
                f"Circuit is open — {self._failure_threshold} failures in {self._window}s window"
            )

        try:
            result = await func(*args, **kwargs)
        except Exception:
            self._record_failure()
            if current_state == "half_open":
                self._trip()
            raise

        if current_state == "half_open":
            self._reset()

        return result

    def _record_failure(self):
        now = time.monotonic()
        self._failures.append(now)
        # Prune failures outside the window
        cutoff = now - self._window
        self._failures = [t for t in self._failures if t > cutoff]

        if len(self._failures) >= self._failure_threshold and self._state == "closed":
            self._trip()

    def _trip(self):
        self._state = "open"
        self._opened_at = time.monotonic()

    def _reset(self):
        self._state = "closed"
        self._failures = []
        self._opened_at = None


_RETRYABLE_STATUS_CODES = {429, 500, 529}


async def retry_with_backoff(
    func,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs,
):
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except anthropic.APIStatusError as e:
            if e.response.status_code not in _RETRYABLE_STATUS_CODES:
                raise
            last_exception = e
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                logger.warning(
                    "llm_call_retrying",
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    status_code=e.response.status_code,
                    delay=round(delay, 2),
                )
                await asyncio.sleep(delay)
    raise last_exception


class ConcurrencyExceededError(Exception):
    """Raised when the concurrency semaphore is full."""
    pass


class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent

    async def call(self, func, *args, **kwargs):
        if self._semaphore.locked():
            raise ConcurrencyExceededError(
                f"Max concurrency ({self._max}) reached"
            )
        async with self._semaphore:
            return await func(*args, **kwargs)
