import pytest
from app.models import (
    SummarizeRequest,
    SummarizeConfig,
    SummarizeResponse,
    SummaryMetadata,
    EvaluationScores,
    HealthResponse,
)


def test_summarize_request_defaults():
    req = SummarizeRequest(document="Hello world")
    assert req.document_type == "txt"
    assert req.config.target_language == "auto"
    assert req.config.summary_type == "brief"
    assert req.config.max_length == 500
    assert req.config.preserve_domain_terms is True
    assert req.config.evaluate is False


def test_summarize_request_custom_config():
    req = SummarizeRequest(
        document="Teks dalam Bahasa Melayu",
        document_type="txt",
        config=SummarizeConfig(
            target_language="ms",
            summary_type="executive",
            max_length=200,
            evaluate=True,
        ),
    )
    assert req.config.target_language == "ms"
    assert req.config.summary_type == "executive"
    assert req.config.max_length == 200
    assert req.config.evaluate is True


def test_summarize_config_validates_language():
    with pytest.raises(Exception):
        SummarizeConfig(target_language="fr")


def test_summarize_config_validates_summary_type():
    with pytest.raises(Exception):
        SummarizeConfig(summary_type="ultra")


def test_summarize_response_without_evaluation():
    resp = SummarizeResponse(
        summary="A summary",
        metadata=SummaryMetadata(
            detected_language="en",
            code_switching_detected=False,
            model_used="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            latency_ms=800,
        ),
    )
    assert resp.metadata.evaluation is None


def test_summarize_response_with_evaluation():
    scores = EvaluationScores(
        faithfulness=4.5,
        coherence=4.0,
        coverage=3.8,
        language_quality=4.2,
        conciseness=4.0,
        justification="Good summary overall.",
    )
    resp = SummarizeResponse(
        summary="A summary",
        metadata=SummaryMetadata(
            detected_language="ms",
            code_switching_detected=True,
            model_used="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1200,
            evaluation=scores,
        ),
    )
    assert resp.metadata.evaluation.faithfulness == 4.5


def test_request_accepts_pdf_type():
    req = SummarizeRequest(document="JVBERi0xLjQK", document_type="pdf")
    assert req.document_type == "pdf"


def test_request_rejects_docx_type():
    with pytest.raises(Exception):
        SummarizeRequest(document="data", document_type="docx")


def test_request_accepts_large_pdf_payload():
    """Base64-encoded PDFs can be very large — 33MB PDF = ~44MB base64."""
    large = "A" * 2_000_000
    req = SummarizeRequest(document=large, document_type="pdf")
    assert len(req.document) == 2_000_000


def test_health_response():
    health = HealthResponse(
        status="healthy",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
    )
    assert health.status == "healthy"
