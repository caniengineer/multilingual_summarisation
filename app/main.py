import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import Settings
from app.evaluation import evaluate_summary
from app.logging_config import setup_logging
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
