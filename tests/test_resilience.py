import asyncio
import pytest
from unittest.mock import AsyncMock
import anthropic
from app.resilience import CircuitBreaker, CircuitOpenError, retry_with_backoff


class TestCircuitBreaker:
    """Test circuit breaker state machine transitions."""

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(failure_threshold=3, recovery_timeout=1, window=10)

    @pytest.mark.asyncio
    async def test_starts_closed(self, breaker):
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, breaker):
        func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(3):
            with pytest.raises(Exception, match="fail"):
                await breaker.call(func)
        assert breaker.state == "open"

    @pytest.mark.asyncio
    async def test_open_raises_circuit_open_error(self, breaker):
        func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(func)

        with pytest.raises(CircuitOpenError):
            await breaker.call(func)

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, window=10)
        func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(func)
        assert breaker.state == "open"

        await asyncio.sleep(0.15)
        assert breaker.state == "half_open"

    @pytest.mark.asyncio
    async def test_closes_on_half_open_success(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, window=10)
        fail_func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        await asyncio.sleep(0.15)

        success_func = AsyncMock(return_value="ok")
        result = await breaker.call(success_func)
        assert result == "ok"
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_reopens_on_half_open_failure(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, window=10)
        fail_func = AsyncMock(side_effect=Exception("fail"))
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(fail_func)

        await asyncio.sleep(0.15)

        with pytest.raises(Exception, match="fail"):
            await breaker.call(fail_func)
        assert breaker.state == "open"


def _mock_response(status_code, text):
    """Create a mock httpx response with required attributes for anthropic exceptions."""
    request = type("HttpxRequest", (), {"method": "POST", "url": "https://api.anthropic.com/v1/messages"})()
    return type("HttpxResponse", (), {
        "status_code": status_code,
        "headers": {},
        "text": text,
        "request": request,
    })()


class TestRetryWithBackoff:
    """Test retry with exponential backoff and jitter."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        func = AsyncMock(return_value="ok")
        result = await retry_with_backoff(func, max_attempts=3, base_delay=0.01)
        assert result == "ok"
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self):
        func = AsyncMock(
            side_effect=[
                anthropic.APIStatusError(
                    message="rate limited",
                    response=_mock_response(429, "rate limited"),
                    body=None,
                ),
                "ok",
            ]
        )
        result = await retry_with_backoff(func, max_attempts=3, base_delay=0.01)
        assert result == "ok"
        assert func.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_400(self):
        func = AsyncMock(
            side_effect=anthropic.APIStatusError(
                message="bad request",
                response=_mock_response(400, "bad request"),
                body=None,
            )
        )
        with pytest.raises(anthropic.APIStatusError):
            await retry_with_backoff(func, max_attempts=3, base_delay=0.01)
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausts_attempts(self):
        func = AsyncMock(
            side_effect=anthropic.APIStatusError(
                message="server error",
                response=_mock_response(500, "server error"),
                body=None,
            )
        )
        with pytest.raises(anthropic.APIStatusError):
            await retry_with_backoff(func, max_attempts=3, base_delay=0.01)
        assert func.call_count == 3


from app.resilience import ConcurrencyLimiter, ConcurrencyExceededError


class TestConcurrencyLimiter:
    """Test concurrency limiter with asyncio semaphore."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        limiter = ConcurrencyLimiter(max_concurrent=2)
        func = AsyncMock(return_value="ok")
        result = await limiter.call(func)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_rejects_when_full(self):
        limiter = ConcurrencyLimiter(max_concurrent=1)
        started = asyncio.Event()
        blocked = asyncio.Event()

        async def slow():
            started.set()
            await blocked.wait()
            return "done"

        # Fill the semaphore
        task = asyncio.create_task(limiter.call(slow))
        await started.wait()

        # Next call should be rejected
        with pytest.raises(ConcurrencyExceededError):
            await limiter.call(AsyncMock(return_value="nope"))

        blocked.set()
        await task
