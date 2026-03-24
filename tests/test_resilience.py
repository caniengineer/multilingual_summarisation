import asyncio
import pytest
from unittest.mock import AsyncMock
from app.resilience import CircuitBreaker, CircuitOpenError


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
