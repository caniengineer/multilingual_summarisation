"""End-to-end API tests for PDF summarization.

These tests call the /v1/summarize endpoint with real PDF fixtures
and a mock provider, verifying the full request → extraction → response flow.

Unlike test_main.py (which tests generic API behavior), these focus specifically
on PDF document handling: base64 encoding, extraction, error responses, and
verifying the provider receives extracted text rather than raw base64.

Run: uv run pytest tests/test_api_pdf.py -v
"""

import base64
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock

from app.main import create_app
from app.models import EvaluationScores
from app.providers.base import SumResult

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdf"


def _load_b64(name: str) -> str:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture not found: {name}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.name = "test-model"
    provider.max_context_tokens = 200_000
    provider.summarize = AsyncMock(
        return_value=SumResult(
            summary="Test summary of the document.",
            detected_language="en",
            code_switching_detected=False,
            model_used="test-model",
            input_tokens=500,
            output_tokens=80,
        )
    )
    provider.evaluate = AsyncMock(
        return_value=EvaluationScores(
            faithfulness=4.5,
            coherence=4.0,
            coverage=3.8,
            language_quality=4.2,
            conciseness=4.0,
            justification="Good summary.",
        )
    )
    return provider


@pytest.fixture
def app(mock_provider):
    return create_app(provider=mock_provider)


# ---------- P0: Demo blockers ----------


@pytest.mark.asyncio
async def test_pdf_summarize_returns_200(app, mock_provider):
    """Full API flow: base64 PDF → extraction → provider → 200 response."""
    b64 = _load_b64("budget_speech_2026_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
                "config": {"target_language": "en", "summary_type": "brief"},
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "metadata" in data
    assert data["metadata"]["detected_language"] == "en"


@pytest.mark.asyncio
async def test_pdf_provider_receives_extracted_text(app, mock_provider):
    """Provider must receive extracted text, not raw base64."""
    b64 = _load_b64("budget_speech_2026_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
            },
        )
    mock_provider.summarize.assert_called_once()
    text_sent = mock_provider.summarize.call_args[0][0]
    # Must be readable text, not base64 gibberish
    assert len(text_sent) > 100
    assert "JVBERi0" not in text_sent  # PDF magic bytes in base64
    assert "AAAA" * 10 not in text_sent  # not base64 padding patterns


@pytest.mark.asyncio
async def test_pdf_provider_receives_substantial_text(app, mock_provider):
    """Extracted text should be substantial — not just a few header lines."""
    b64 = _load_b64("budget_speech_2026_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
            },
        )
    text_sent = mock_provider.summarize.call_args[0][0]
    assert len(text_sent) > 1000, f"Expected >1000 chars, got {len(text_sent)}"


@pytest.mark.asyncio
async def test_pdf_invalid_base64_returns_400(app):
    """Invalid base64 → 400, not 500."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": "!!!this-is-not-base64!!!",
                "document_type": "pdf",
            },
        )
    assert resp.status_code == 400
    assert "base64" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_pdf_corrupted_bytes_returns_400(app):
    """Valid base64 but not a PDF → 400, not 500."""
    b64 = base64.b64encode(b"this is plain text, not a PDF").decode()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
            },
        )
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "pdf" in detail or "empty" in detail


# ---------- P1: Correctness ----------


@pytest.mark.asyncio
async def test_pdf_response_metadata_structure(app, mock_provider):
    """Response metadata has all expected fields."""
    b64 = _load_b64("budget_speech_2026_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
            },
        )
    data = resp.json()
    meta = data["metadata"]
    assert "detected_language" in meta
    assert "model_used" in meta
    assert "input_tokens" in meta
    assert "output_tokens" in meta
    assert "latency_ms" in meta
    assert meta["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_pdf_with_evaluation(app, mock_provider):
    """PDF + evaluate=True returns evaluation scores."""
    b64 = _load_b64("budget_speech_2026_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
                "config": {"evaluate": True},
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["evaluation"] is not None
    assert data["metadata"]["evaluation"]["faithfulness"] == 4.5


@pytest.mark.asyncio
async def test_pdf_empty_pdf_returns_400(app):
    """A valid PDF with zero extractable text → 400."""
    import fitz

    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    b64 = base64.b64encode(pdf_bytes).decode()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
            },
        )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_pdf_dosm_statistics_returns_200(app, mock_provider):
    """DOSM statistics PDF (2.5MB) processes successfully."""
    b64 = _load_b64("dosm_mesr_2024_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
                "config": {"target_language": "en", "summary_type": "detailed"},
            },
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_pdf_bnm_annual_report_returns_200(app, mock_provider):
    """BNM annual report (8.8MB) processes successfully."""
    b64 = _load_b64("bnm_annual_report_2021_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
                "config": {"target_language": "en", "summary_type": "executive"},
            },
        )
    assert resp.status_code == 200


# ---------- P2: Robustness ----------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_pdf_bm_budget_speech_returns_200(app, mock_provider):
    """BM budget speech (33MB, graphically heavy) processes without OOM or timeout."""
    b64 = _load_b64("budget_speech_2026_bm.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
                "config": {"target_language": "ms", "summary_type": "executive"},
            },
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_pdf_default_config_works(app, mock_provider):
    """PDF with no explicit config uses defaults (target_language=auto, type=brief)."""
    b64 = _load_b64("budget_speech_2026_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
            },
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_pdf_text_is_normalized(app, mock_provider):
    """Extracted text sent to provider should be normalized (no triple newlines)."""
    b64 = _load_b64("budget_speech_2026_en.pdf")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/summarize",
            json={
                "document": b64,
                "document_type": "pdf",
            },
        )
    text_sent = mock_provider.summarize.call_args[0][0]
    assert "\n\n\n" not in text_sent
    assert text_sent == text_sent.strip()


@pytest.mark.asyncio
async def test_txt_still_works_after_pdf_addition(app, mock_provider):
    """Regression: adding PDF support must not break existing TXT flow."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/summarize",
            json={
                "document": "This is a plain text document about Malaysian economics.",
                "document_type": "txt",
                "config": {"target_language": "en"},
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] == "Test summary of the document."
