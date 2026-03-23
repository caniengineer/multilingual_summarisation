# Production Resilience Patterns Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add production-grade resilience patterns (structured logging, retry, timeout, circuit breaker, concurrency control, metrics, deep health checks) to the multilingual summarization API.

**Architecture:** New resilience primitives live in `app/resilience.py`. Middleware in `app/middleware.py`. Logging config in `app/logging_config.py`. Metrics in `app/metrics.py`. All wired into the existing FastAPI app via `app/main.py`. Provider calls in `app/providers/anthropic.py` get wrapped with retry+timeout+circuit breaker.

**Tech Stack:** structlog, prometheus-client, asyncio (stdlib), FastAPI middleware

---

### Task 1: Add dependencies and config settings

**Files:**
- Modify: `pyproject.toml:6-15`
- Modify: `app/config.py:1-14`
- Test: `tests/test_config.py` (existing)

**Step 1: Add structlog and prometheus-client to dependencies**

In `pyproject.toml`, add to `dependencies`:
```toml
"structlog>=24.0",
"prometheus-client>=0.21.0",
```

**Step 2: Add resilience config settings**

In `app/config.py`, add these fields to the `Settings` class:
```python
# Retry
RETRY_MAX_ATTEMPTS: int = 3
RETRY_BASE_DELAY: float = 1.0
RETRY_MAX_DELAY: float = 30.0

# Timeouts
REQUEST_TIMEOUT: int = 120
LLM_CALL_TIMEOUT: int = 60

# Circuit breaker
CB_FAILURE_THRESHOLD: int = 5
CB_RECOVERY_TIMEOUT: int = 30
CB_WINDOW: int = 60

# Concurrency
MAX_CONCURRENT_LLM_CALLS: int = 10
```

**Step 3: Install dependencies**

Run: `uv sync`

**Step 4: Run existing tests to verify nothing breaks**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add pyproject.toml app/config.py uv.lock
git commit -m "feat: add resilience config settings and dependencies"
```

---

### Task 2: Structured logging with structlog

**Files:**
- Create: `app/logging_config.py`
- Test: `tests/test_logging_config.py`

**Step 1: Write the failing test**

Create `tests/test_logging_config.py`:
```python
import json
import logging
import structlog
from app.logging_config import setup_logging, get_logger, request_id_var


def test_setup_logging_configures_structlog():
    setup_logging("info")
    logger = get_logger("test")
    assert logger is not None


def test_request_id_var_default_is_dash():
    request_id_var.set("-")
    assert request_id_var.get() == "-"


def test_logger_outputs_json(capsys):
    setup_logging("info")
    logger = get_logger("test.json")
    request_id_var.set("test-req-123")
    logger.info("hello", foo="bar")
    output = capsys.readouterr().err
    # structlog JSON output should contain our fields
    assert "hello" in output or "test-req-123" in output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_logging_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.logging_config'`

**Step 3: Write the implementation**

Create `app/logging_config.py`:
```python
import logging
import sys
from contextvars import ContextVar

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _add_request_id(logger, method_name, event_dict):
    event_dict["request_id"] = request_id_var.get()
    return event_dict


def setup_logging(log_level: str = "info") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_request_id,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to route through structlog
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_logging_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/logging_config.py tests/test_logging_config.py
git commit -m "feat: add structured logging with structlog and request ID context"
```

---

### Task 3: Request ID middleware

**Files:**
- Create: `app/middleware.py`
- Test: `tests/test_middleware.py`

**Step 1: Write the failing test**

Create `tests/test_middleware.py`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from app.middleware import RequestIDMiddleware


@pytest.fixture
def test_app():
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    return app


@pytest.mark.asyncio
async def test_adds_request_id_header(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/test")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    # Should be a UUID
    req_id = resp.headers["x-request-id"]
    assert len(req_id) == 36  # UUID4 format


@pytest.mark.asyncio
async def test_preserves_caller_request_id(test_app):
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/test", headers={"X-Request-ID": "caller-id-123"})
    assert resp.headers["x-request-id"] == "caller-id-123"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_middleware.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.middleware'`

**Step 3: Write the implementation**

Create `app/middleware.py`:
```python
import uuid
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import request_id_var, get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request_id_var.set(request_id)

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
        )

        start = time.time()
        response: Response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)

        response.headers["x-request-id"] = request_id
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_middleware.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/middleware.py tests/test_middleware.py
git commit -m "feat: add request ID middleware with structured logging"
```

---

### Task 4: Circuit breaker

**Files:**
- Create: `app/resilience.py`
- Test: `tests/test_resilience.py`

**Step 1: Write the failing tests for circuit breaker**

Create `tests/test_resilience.py`:
```python
import asyncio
import pytest
from unittest.mock import AsyncMock
from app.resilience import CircuitBreaker, CircuitOpenError


@pytest.fixture
def breaker():
    return CircuitBreaker(failure_threshold=3, recovery_timeout=1, window=10)


@pytest.mark.asyncio
async def test_circuit_starts_closed(breaker):
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_failures(breaker):
    func = AsyncMock(side_effect=Exception("fail"))
    for _ in range(3):
        with pytest.raises(Exception, match="fail"):
            await breaker.call(func)
    assert breaker.state == "open"


@pytest.mark.asyncio
async def test_circuit_open_raises_circuit_open_error(breaker):
    func = AsyncMock(side_effect=Exception("fail"))
    for _ in range(3):
        with pytest.raises(Exception):
            await breaker.call(func)

    with pytest.raises(CircuitOpenError):
        await breaker.call(func)


@pytest.mark.asyncio
async def test_circuit_half_open_after_recovery_timeout():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, window=10)
    func = AsyncMock(side_effect=Exception("fail"))
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(func)
    assert breaker.state == "open"

    await asyncio.sleep(0.15)
    assert breaker.state == "half_open"


@pytest.mark.asyncio
async def test_circuit_closes_on_half_open_success():
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
async def test_circuit_reopens_on_half_open_failure():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, window=10)
    fail_func = AsyncMock(side_effect=Exception("fail"))
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(fail_func)

    await asyncio.sleep(0.15)

    with pytest.raises(Exception, match="fail"):
        await breaker.call(fail_func)
    assert breaker.state == "open"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_resilience.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.resilience'`

**Step 3: Write the circuit breaker implementation**

Create `app/resilience.py`:
```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_resilience.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/resilience.py tests/test_resilience.py
git commit -m "feat: add circuit breaker with sliding window failure detection"
```

---

### Task 5: Retry with exponential backoff + jitter

**Files:**
- Modify: `app/resilience.py`
- Modify: `tests/test_resilience.py`

**Step 1: Write the failing tests for retry**

Append to `tests/test_resilience.py`:
```python
import anthropic
from app.resilience import retry_with_backoff


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_try():
    func = AsyncMock(return_value="ok")
    result = await retry_with_backoff(func, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retry_retries_on_retryable_error():
    error_resp = type("Response", (), {"status_code": 429})()
    func = AsyncMock(
        side_effect=[
            anthropic.APIStatusError(
                message="rate limited",
                response=type("HttpxResponse", (), {"status_code": 429, "headers": {}, "text": "rate limited"})(),
                body=None,
            ),
            "ok",
        ]
    )
    result = await retry_with_backoff(func, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_retry_does_not_retry_400():
    func = AsyncMock(
        side_effect=anthropic.APIStatusError(
            message="bad request",
            response=type("HttpxResponse", (), {"status_code": 400, "headers": {}, "text": "bad request"})(),
            body=None,
        )
    )
    with pytest.raises(anthropic.APIStatusError):
        await retry_with_backoff(func, max_attempts=3, base_delay=0.01)
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retry_exhausts_attempts():
    func = AsyncMock(
        side_effect=anthropic.APIStatusError(
            message="server error",
            response=type("HttpxResponse", (), {"status_code": 500, "headers": {}, "text": "server error"})(),
            body=None,
        )
    )
    with pytest.raises(anthropic.APIStatusError):
        await retry_with_backoff(func, max_attempts=3, base_delay=0.01)
    assert func.call_count == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_resilience.py::test_retry_succeeds_on_first_try -v`
Expected: FAIL — `ImportError: cannot import name 'retry_with_backoff'`

**Step 3: Add retry implementation to `app/resilience.py`**

Add to `app/resilience.py`:
```python
import random
import anthropic
from app.logging_config import get_logger

logger = get_logger(__name__)

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
```

**Step 4: Run tests**

Run: `pytest tests/test_resilience.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/resilience.py tests/test_resilience.py
git commit -m "feat: add retry with exponential backoff and jitter for LLM calls"
```

---

### Task 6: Concurrency semaphore

**Files:**
- Modify: `app/resilience.py`
- Modify: `tests/test_resilience.py`

**Step 1: Write the failing tests**

Append to `tests/test_resilience.py`:
```python
from app.resilience import ConcurrencyLimiter, ConcurrencyExceededError


@pytest.mark.asyncio
async def test_semaphore_allows_within_limit():
    limiter = ConcurrencyLimiter(max_concurrent=2)
    func = AsyncMock(return_value="ok")
    result = await limiter.call(func)
    assert result == "ok"


@pytest.mark.asyncio
async def test_semaphore_rejects_when_full():
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_resilience.py::test_semaphore_allows_within_limit -v`
Expected: FAIL — `ImportError`

**Step 3: Add concurrency limiter to `app/resilience.py`**

Add to `app/resilience.py`:
```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_resilience.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/resilience.py tests/test_resilience.py
git commit -m "feat: add concurrency limiter with asyncio semaphore"
```

---

### Task 7: Prometheus metrics

**Files:**
- Create: `app/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write the failing test**

Create `tests/test_metrics.py`:
```python
from app.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
    TOKENS_USED,
    LLM_CALL_COUNT,
)


def test_metrics_exist():
    assert REQUEST_COUNT is not None
    assert REQUEST_LATENCY is not None
    assert ERROR_COUNT is not None
    assert TOKENS_USED is not None
    assert LLM_CALL_COUNT is not None


def test_request_count_increments():
    before = REQUEST_COUNT.labels(method="GET", endpoint="/test", status="200")._value.get()
    REQUEST_COUNT.labels(method="GET", endpoint="/test", status="200").inc()
    after = REQUEST_COUNT.labels(method="GET", endpoint="/test", status="200")._value.get()
    assert after == before + 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL

**Step 3: Write the implementation**

Create `app/metrics.py`:
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP errors",
    ["method", "endpoint", "error_type"],
)

LLM_CALL_COUNT = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["method", "status"],
)

LLM_CALL_LATENCY = Histogram(
    "llm_call_duration_seconds",
    "LLM API call latency in seconds",
    ["method"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

TOKENS_USED = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["direction"],  # "input" or "output"
)
```

**Step 4: Run tests**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/metrics.py tests/test_metrics.py
git commit -m "feat: add Prometheus metric definitions for RED metrics and token tracking"
```

---

### Task 8: Wire middleware and metrics into FastAPI app

**Files:**
- Modify: `app/main.py:1-97`
- Modify: `app/middleware.py`
- Modify: `tests/test_main.py`

**Step 1: Update middleware to record metrics**

Add metrics recording to `app/middleware.py` `RequestIDMiddleware.dispatch`:
```python
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT

# After getting the response, add:
REQUEST_COUNT.labels(
    method=request.method,
    endpoint=request.url.path,
    status=str(response.status_code),
).inc()
REQUEST_LATENCY.labels(
    method=request.method,
    endpoint=request.url.path,
).observe(duration_ms / 1000)

if response.status_code >= 400:
    ERROR_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        error_type=str(response.status_code),
    ).inc()
```

**Step 2: Add request deadline middleware**

Add to `app/middleware.py`:
```python
class RequestDeadlineMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout: int = 120):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            response = await asyncio.wait_for(
                call_next(request), timeout=self.timeout
            )
            return response
        except asyncio.TimeoutError:
            logger.error("request_timeout", timeout=self.timeout)
            return Response(
                content='{"detail":"Request timeout"}',
                status_code=504,
                media_type="application/json",
            )
```

**Step 3: Wire everything into `app/main.py`**

Update `app/main.py` to:
- Replace `import logging` with structlog setup
- Add middleware registration
- Add `/metrics` endpoint using `prometheus_client.generate_latest`
- Add `/v1/health/live` and `/v1/health/ready` endpoints

```python
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import Settings
from app.evaluation import evaluate_summary
from app.logging_config import setup_logging, get_logger
from app.metrics import TOKENS_USED
from app.middleware import RequestIDMiddleware, RequestDeadlineMiddleware
from app.models import (
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
    SummaryMetadata,
)
from app.processor import DocumentProcessor
from app.providers.anthropic import AnthropicProvider
from app.providers.claude_code import ClaudeCodeProvider
from app.resilience import CircuitBreaker, ConcurrencyLimiter, CircuitOpenError, ConcurrencyExceededError
from app.summarizer import summarize_document


def create_app(provider=None) -> FastAPI:
    settings = Settings() if provider is None else None
    if settings:
        setup_logging(settings.LOG_LEVEL)
    else:
        setup_logging("info")

    logger = get_logger(__name__)

    app = FastAPI(
        title="Multilingual Document Summarization API",
        version="0.1.0",
    )
    processor = DocumentProcessor()

    # Resilience primitives
    circuit_breaker = CircuitBreaker(
        failure_threshold=settings.CB_FAILURE_THRESHOLD if settings else 5,
        recovery_timeout=settings.CB_RECOVERY_TIMEOUT if settings else 30,
        window=settings.CB_WINDOW if settings else 60,
    )
    concurrency_limiter = ConcurrencyLimiter(
        max_concurrent=settings.MAX_CONCURRENT_LLM_CALLS if settings else 10,
    )

    # Store on app state for health checks
    app.state.circuit_breaker = circuit_breaker

    # Middleware (order matters: outermost first)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        RequestDeadlineMiddleware,
        timeout=settings.REQUEST_TIMEOUT if settings else 120,
    )

    if provider is None:
        if settings.PROVIDER == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is required when PROVIDER=anthropic")
            provider = AnthropicProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.DEFAULT_MODEL,
            )
        elif settings.PROVIDER == "claude-code":
            provider = ClaudeCodeProvider(model=settings.CLAUDE_CODE_MODEL)
        else:
            raise ValueError(f"Unknown provider: {settings.PROVIDER}")

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            provider=provider.name,
            model=provider.name,
        )

    @app.get("/v1/health/live")
    async def health_live():
        return {"status": "alive"}

    @app.get("/v1/health/ready")
    async def health_ready():
        cb_state = circuit_breaker.state
        ready = cb_state != "open"
        status_code = 200 if ready else 503
        return PlainTextResponse(
            content=json.dumps({
                "status": "ready" if ready else "not_ready",
                "circuit_breaker": cb_state,
                "provider": provider.name,
            }),
            status_code=status_code,
            media_type="application/json",
        )

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/v1/summarize", response_model=SummarizeResponse)
    async def summarize(request: SummarizeRequest):
        start = time.time()

        try:
            doc = processor.process(request.document, request.document_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            result = await concurrency_limiter.call(
                circuit_breaker.call,
                summarize_document, doc.text, request.config, provider,
                settings=settings,
            )
        except CircuitOpenError:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable — circuit breaker open")
        except ConcurrencyExceededError:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"detail": "Too many concurrent requests"},
                headers={"Retry-After": "5"},
            )
        except ValueError as e:
            raise HTTPException(status_code=502, detail=str(e))

        latency_ms = int((time.time() - start) * 1000)

        # Track token usage
        TOKENS_USED.labels(direction="input").inc(result.input_tokens)
        TOKENS_USED.labels(direction="output").inc(result.output_tokens)

        evaluation = None
        if request.config.evaluate:
            evaluation = await evaluate_summary(
                provider=provider,
                source=request.document,
                summary=result.summary,
                target_language=result.detected_language,
            )

        return SummarizeResponse(
            summary=result.summary,
            metadata=SummaryMetadata(
                detected_language=result.detected_language,
                code_switching_detected=result.code_switching_detected,
                model_used=result.model_used,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=latency_ms,
                evaluation=evaluation,
                chunks_used=result.chunks_used,
            ),
        )

    return app
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS (existing tests should still work since middleware is additive)

**Step 5: Commit**

```bash
git add app/main.py app/middleware.py
git commit -m "feat: wire middleware, metrics, health checks, and resilience into app"
```

---

### Task 9: Wire retry + timeout into AnthropicProvider

**Files:**
- Modify: `app/providers/anthropic.py:1-102`
- Modify: `app/summarizer.py:1-81`

**Step 1: Add retry and timeout wrapping to provider calls**

Update `app/providers/anthropic.py` to accept settings and wrap each `_client.messages.create` call with `retry_with_backoff` and `asyncio.wait_for`:

```python
import asyncio
import anthropic

from app.config import Settings
from app.logging_config import get_logger
from app.metrics import LLM_CALL_COUNT, LLM_CALL_LATENCY, TOKENS_USED
from app.models import SummarizeConfig, EvaluationScores
from app.prompt_loader import PromptLoader
from app.providers.base import SumResult, parse_llm_json
from app.resilience import retry_with_backoff

logger = get_logger(__name__)


class AnthropicProvider:
    def __init__(self, api_key: str, model: str, settings: Settings | None = None):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._prompt_loader = PromptLoader()
        self._settings = settings

    @property
    def _call_timeout(self) -> int:
        return self._settings.LLM_CALL_TIMEOUT if self._settings else 60

    @property
    def _max_attempts(self) -> int:
        return self._settings.RETRY_MAX_ATTEMPTS if self._settings else 3

    @property
    def _base_delay(self) -> float:
        return self._settings.RETRY_BASE_DELAY if self._settings else 1.0

    @property
    def _max_delay(self) -> float:
        return self._settings.RETRY_MAX_DELAY if self._settings else 30.0

    async def _call_llm(self, messages, max_tokens=4096, method="summarize"):
        import time
        start = time.time()

        async def _do_call():
            return await asyncio.wait_for(
                self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=messages,
                ),
                timeout=self._call_timeout,
            )

        try:
            response = await retry_with_backoff(
                _do_call,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
                max_delay=self._max_delay,
            )
            LLM_CALL_COUNT.labels(method=method, status="success").inc()
            return response
        except Exception:
            LLM_CALL_COUNT.labels(method=method, status="error").inc()
            raise
        finally:
            duration = time.time() - start
            LLM_CALL_LATENCY.labels(method=method).observe(duration)
    # ... rest of methods use self._call_llm instead of self._client.messages.create
```

Each of the `summarize`, `reduce`, `evaluate` methods replaces:
```python
response = await self._client.messages.create(...)
```
with:
```python
response = await self._call_llm(
    messages=[{"role": "user", "content": rendered}],
    max_tokens=4096,
    method="summarize",  # or "reduce" / "evaluate"
)
```

**Step 2: Update summarizer to pass settings through circuit breaker**

The circuit breaker wrapping in `app/main.py` already wraps the `summarize_document` call. No changes needed to `app/summarizer.py` itself — the retry/timeout lives inside the provider.

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS. Existing tests use `AsyncMock` providers so they bypass the real retry/timeout logic.

**Step 4: Commit**

```bash
git add app/providers/anthropic.py
git commit -m "feat: add retry, timeout, and metrics instrumentation to AnthropicProvider"
```

---

### Task 10: Add logging to key lifecycle points

**Files:**
- Modify: `app/providers/anthropic.py`
- Modify: `app/summarizer.py`
- Modify: `app/processor.py`

**Step 1: Add structured log lines**

Add `logger.info(...)` calls at:
- `processor.py`: document processing start/complete, PDF extraction
- `summarizer.py`: chunk decision, map phase progress, reduce phase
- `providers/anthropic.py`: LLM call start (already in `_call_llm`)

Example additions:
```python
# summarizer.py
logger.info("summarize_started", estimated_tokens=estimated_tokens, chunking_needed=estimated_tokens > token_budget)
logger.info("chunk_summarized", chunk=i+1, total_chunks=len(chunks))
logger.info("reduce_started", section_count=len(section_summaries))
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add app/providers/anthropic.py app/summarizer.py app/processor.py
git commit -m "feat: add structured logging at key lifecycle points"
```

---

### Task 11: Final integration test and full test suite run

**Files:**
- All test files

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All PASS

**Step 2: Verify metrics endpoint manually**

Run: `python -c "from app.main import create_app; print('App creates successfully')"` (with env vars set)

**Step 3: Commit any remaining fixes**

```bash
git add -A
git commit -m "test: verify all resilience patterns integrate correctly"
```
