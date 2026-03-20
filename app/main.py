# app/main.py
import logging
import time

from fastapi import FastAPI, HTTPException

from app.evaluation import evaluate_summary
from app.models import (
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
    SummaryMetadata,
)
from app.processor import DocumentProcessor
from app.providers.anthropic import AnthropicProvider
from app.config import Settings


logger = logging.getLogger(__name__)


def create_app(provider=None) -> FastAPI:
    app = FastAPI(
        title="Multilingual Document Summarization API",
        version="0.1.0",
    )
    processor = DocumentProcessor()

    if provider is None:
        settings = Settings()
        provider = AnthropicProvider(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.DEFAULT_MODEL,
        )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            provider=provider.name,
            model=provider.name,
        )

    @app.post("/v1/summarize", response_model=SummarizeResponse)
    async def summarize(request: SummarizeRequest):
        start = time.time()

        try:
            doc = processor.process(request.document, request.document_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = await provider.summarize(doc.text, request.config)

        latency_ms = int((time.time() - start) * 1000)

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
            ),
        )

    return app
