# tests/test_main.py
import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.providers.base import SumResult
from app.models import EvaluationScores


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.name = "test-model"
    provider.max_context_tokens = 200_000
    provider.summarize = AsyncMock(return_value=SumResult(
        summary="Test summary output.",
        detected_language="en",
        code_switching_detected=False,
        model_used="test-model",
        input_tokens=100,
        output_tokens=50,
    ))
    provider.evaluate = AsyncMock(return_value=EvaluationScores(
        faithfulness=4.5, coherence=4.0, coverage=3.8,
        language_quality=4.2, conciseness=4.0,
        justification="Good summary.",
    ))
    return provider


@pytest.fixture
def app(mock_provider):
    return create_app(provider=mock_provider)


@pytest.mark.asyncio
async def test_health_endpoint(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_summarize_basic(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "This is a test document about Malaysian technology.",
            "document_type": "txt",
            "config": {
                "target_language": "en",
                "summary_type": "brief",
                "max_length": 100,
            },
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] == "Test summary output."
    assert data["metadata"]["detected_language"] == "en"
    assert data["metadata"]["evaluation"] is None


@pytest.mark.asyncio
async def test_summarize_with_evaluation(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "Test document content.",
            "config": {"evaluate": True},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["evaluation"] is not None
    assert data["metadata"]["evaluation"]["faithfulness"] == 4.5


@pytest.mark.asyncio
async def test_summarize_without_evaluation(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "Test document content.",
            "config": {"evaluate": False},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["evaluation"] is None
    mock_provider.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_empty_document(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "",
        })
    assert resp.status_code == 422 or resp.status_code == 400


@pytest.mark.asyncio
async def test_summarize_default_config(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "Just a document with defaults.",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["evaluation"] is None


def test_create_app_with_claude_code_provider(monkeypatch):
    monkeypatch.setenv("PROVIDER", "claude-code")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    app = create_app()
    assert app is not None


def test_create_app_with_unknown_provider(monkeypatch):
    monkeypatch.setenv("PROVIDER", "unknown")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    with pytest.raises(ValueError, match="Unknown provider"):
        create_app()
