# Production Resilience Patterns — Design

## Context

Multilingual document summarization API (FastAPI + Anthropic Claude) for YTL AI Labs interview demo. The API currently has zero resilience patterns — no retries, timeouts, circuit breaking, structured logging, or metrics. This design adds production-grade distributed systems patterns.

## 1. Request ID Middleware + Structured Logging

**Middleware** (`app/middleware.py`):
- Generates UUID4 per request, accepts `X-Request-ID` header override
- Sets request ID on response header
- Stores request ID in contextvars for structlog binding

**Structured logging** (`app/logging_config.py`):
- structlog with JSON output
- Request ID automatically bound to every log line
- Log points: request start, document processing, LLM call start/end, errors, response with latency

## 2. Retry with Exponential Backoff + Jitter

**Location:** `app/resilience.py`

- Async retry decorator wrapping LLM calls
- Formula: `min(base * 2^attempt + random_jitter, max_delay)`
- Retryable errors: 429 (rate limit), 500 (server error), 529 (Anthropic overload)
- Never retry 400-class errors
- Max 3 retries, configurable

## 3. Timeouts

- Per-LLM-call timeout via `asyncio.wait_for()` (default 60s)
- Overall request deadline middleware (default 120s)
- Timeout raises `asyncio.TimeoutError` → HTTP 504

## 4. Circuit Breaker

**Custom state machine** in `app/resilience.py`:

```
CLOSED → (N failures in M seconds) → OPEN
OPEN → (cooldown expires) → HALF_OPEN
HALF_OPEN → success → CLOSED
HALF_OPEN → failure → OPEN
```

- Opens after 5 failures within 60 seconds (configurable)
- 30-second cooldown before half-open test request
- Prevents cascading failures when Anthropic API is down

## 5. Concurrency Semaphore

- `asyncio.Semaphore` (default 10 concurrent LLM calls)
- Non-blocking acquire — immediate 503 + `Retry-After` header when at capacity
- Configurable via `MAX_CONCURRENT_LLM_CALLS`

## 6. /metrics Endpoint (Prometheus)

Using `prometheus-client`:

- **Rate:** request count per endpoint (Counter)
- **Errors:** error count by type/status code (Counter)
- **Duration:** latency histogram with P50/P95/P99 buckets (Histogram)
- **Tokens:** input/output token counters (Counter)
- Exposed at `/metrics` in Prometheus exposition format

## 7. Deep Health Checks

- `/v1/health/live` — liveness: process is running (always 200)
- `/v1/health/ready` — readiness: Anthropic API reachable, circuit breaker state, config valid
- Replaces current static `/v1/health`

## File Structure

```
app/
├── middleware.py          # Request ID + request deadline middleware
├── logging_config.py      # structlog configuration
├── resilience.py          # Retry, circuit breaker, semaphore
├── metrics.py             # Prometheus metrics definitions
```

## New Dependencies

- `structlog` — structured JSON logging
- `prometheus-client` — metrics exposition

## Config Additions (app/config.py)

| Variable | Default | Purpose |
|----------|---------|---------|
| RETRY_MAX_ATTEMPTS | 3 | Max retry attempts for LLM calls |
| RETRY_BASE_DELAY | 1.0 | Base delay in seconds |
| RETRY_MAX_DELAY | 30.0 | Max delay cap in seconds |
| REQUEST_TIMEOUT | 120 | Overall request deadline (seconds) |
| LLM_CALL_TIMEOUT | 60 | Per-LLM-call timeout (seconds) |
| CB_FAILURE_THRESHOLD | 5 | Failures before circuit opens |
| CB_RECOVERY_TIMEOUT | 30 | Cooldown before half-open (seconds) |
| CB_WINDOW | 60 | Failure counting window (seconds) |
| MAX_CONCURRENT_LLM_CALLS | 10 | Semaphore limit |
