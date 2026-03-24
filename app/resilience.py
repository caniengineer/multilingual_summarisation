import time
import asyncio


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
