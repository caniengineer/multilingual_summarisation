import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings

import pytest
from app.providers.base import SummarizationProvider, SumResult
from app.models import SummarizeConfig
from app.providers.anthropic import AnthropicProvider
from app.providers.claude_code import ClaudeCodeProvider


def _mock_anthropic_response(content_text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock Anthropic API response."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=content_text)]
    mock_resp.usage.input_tokens = input_tokens
    mock_resp.usage.output_tokens = output_tokens
    return mock_resp


def test_sum_result_creation():
    result = SumResult(
        summary="Test summary",
        detected_language="en",
        code_switching_detected=False,
        model_used="test-model",
        input_tokens=100,
        output_tokens=50,
    )
    assert result.summary == "Test summary"
    assert result.detected_language == "en"
    assert result.model_used == "test-model"


def test_protocol_is_runtime_checkable():
    """SummarizationProvider should be a runtime-checkable Protocol."""
    assert hasattr(SummarizationProvider, '__protocol_attrs__') or hasattr(SummarizationProvider, '__abstractmethods__') or callable(getattr(SummarizationProvider, '_is_protocol', None))
    # The key test is that isinstance() works with it
    class FakeProvider:
        async def summarize(self, text, config): ...
        async def evaluate(self, source, summary, target_language): ...
        @property
        def name(self): return "fake"
        @property
        def max_context_tokens(self): return 1000

    assert isinstance(FakeProvider(), SummarizationProvider)


@pytest.mark.asyncio
async def test_anthropic_summarize():
    mock_response = _mock_anthropic_response(
        json.dumps({
            "summary": "This is a test summary.",
            "detected_language": "en",
            "code_switching_detected": False,
        })
    )

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        config = SummarizeConfig(target_language="en", summary_type="brief", max_length=100)
        result = await provider.summarize("Test document text.", config)

    assert result.summary == "This is a test summary."
    assert result.detected_language == "en"
    assert result.code_switching_detected is False
    assert result.model_used == "claude-sonnet-4-20250514"
    assert result.input_tokens == 100
    assert result.output_tokens == 50


@pytest.mark.asyncio
async def test_anthropic_summarize_auto_language():
    mock_response = _mock_anthropic_response(
        json.dumps({
            "summary": "Ringkasan dokumen ini.",
            "detected_language": "ms",
            "code_switching_detected": True,
        })
    )

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        config = SummarizeConfig(target_language="auto", summary_type="brief", max_length=100)
        result = await provider.summarize("Dokumen campuran with English.", config)

    assert result.detected_language == "ms"
    assert result.code_switching_detected is True


@pytest.mark.asyncio
async def test_anthropic_evaluate():
    mock_response = _mock_anthropic_response(
        json.dumps({
            "faithfulness": 4.5,
            "coherence": 4.0,
            "coverage": 3.8,
            "language_quality": 4.2,
            "conciseness": 4.0,
            "justification": "Good summary with minor gaps.",
        })
    )

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.evaluate(
            source="Original document text here.",
            summary="A brief summary.",
            target_language="en",
        )

    assert result.faithfulness == 4.5
    assert result.justification == "Good summary with minor gaps."


def test_anthropic_provider_name():
    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")
    assert provider.name == "claude-sonnet-4-20250514"


def test_anthropic_provider_max_tokens():
    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")
    assert provider.max_context_tokens == 200_000


def test_claude_code_provider_satisfies_protocol():
    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")
    assert isinstance(provider, SummarizationProvider)


def test_claude_code_provider_name():
    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")
    assert provider.name == "claude-code/claude-sonnet-4-20250514"


def test_claude_code_provider_max_tokens():
    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")
    assert provider.max_context_tokens == 200_000


def _mock_claude_process(stdout_data: dict, returncode: int = 0, stderr: str = ""):
    """Create a mock for asyncio.create_subprocess_exec returning a claude response."""
    mock_proc = AsyncMock()
    mock_proc.returncode = returncode
    mock_proc.communicate = AsyncMock(
        return_value=(json.dumps(stdout_data).encode(), stderr.encode())
    )
    return mock_proc


@pytest.mark.asyncio
async def test_claude_code_summarize():
    claude_response = {
        "result": json.dumps({
            "summary": "This is a test summary.",
            "detected_language": "en",
            "code_switching_detected": False,
        }),
        "is_error": False,
        "usage": {"input_tokens": 150, "output_tokens": 30},
    }

    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")

    with patch("app.providers.claude_code.asyncio.create_subprocess_exec",
               new_callable=AsyncMock,
               return_value=_mock_claude_process(claude_response)):
        config = SummarizeConfig(target_language="en", summary_type="brief", max_length=100)
        result = await provider.summarize("Test document text.", config)

    assert result.summary == "This is a test summary."
    assert result.detected_language == "en"
    assert result.code_switching_detected is False
    assert result.model_used == "claude-sonnet-4-20250514"
    assert result.input_tokens == 150
    assert result.output_tokens == 30


@pytest.mark.asyncio
async def test_claude_code_evaluate():
    claude_response = {
        "result": json.dumps({
            "faithfulness": 4.5,
            "coherence": 4.0,
            "coverage": 3.8,
            "language_quality": 4.2,
            "conciseness": 4.0,
            "justification": "Good summary.",
        }),
        "is_error": False,
        "usage": {"input_tokens": 200, "output_tokens": 40},
    }

    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")

    with patch("app.providers.claude_code.asyncio.create_subprocess_exec",
               new_callable=AsyncMock,
               return_value=_mock_claude_process(claude_response)):
        result = await provider.evaluate(
            source="Original text.",
            summary="A summary.",
            target_language="en",
        )

    assert result.faithfulness == 4.5
    assert result.justification == "Good summary."


@pytest.mark.asyncio
async def test_claude_code_nonzero_exit():
    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")

    mock_proc = _mock_claude_process({}, returncode=1, stderr="command not found")
    mock_proc.returncode = 1

    with patch("app.providers.claude_code.asyncio.create_subprocess_exec",
               new_callable=AsyncMock, return_value=mock_proc):
        config = SummarizeConfig(target_language="en", summary_type="brief", max_length=100)
        with pytest.raises(RuntimeError, match="claude exited with code 1"):
            await provider.summarize("Test.", config)


@pytest.mark.asyncio
async def test_claude_code_is_error_flag():
    claude_response = {
        "result": "Something went wrong",
        "is_error": True,
        "usage": {},
    }

    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")

    with patch("app.providers.claude_code.asyncio.create_subprocess_exec",
               new_callable=AsyncMock,
               return_value=_mock_claude_process(claude_response)):
        config = SummarizeConfig(target_language="en", summary_type="brief", max_length=100)
        with pytest.raises(RuntimeError, match="claude returned an error"):
            await provider.summarize("Test.", config)


def test_settings_claude_code_no_api_key_required(monkeypatch):
    """ANTHROPIC_API_KEY should be optional when PROVIDER is claude-code."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("PROVIDER", "claude-code")
    settings = Settings()
    assert settings.PROVIDER == "claude-code"
    assert settings.ANTHROPIC_API_KEY is None


def test_settings_claude_code_model_default(monkeypatch):
    monkeypatch.setenv("PROVIDER", "claude-code")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings()
    assert settings.CLAUDE_CODE_MODEL == "claude-sonnet-4-20250514"
