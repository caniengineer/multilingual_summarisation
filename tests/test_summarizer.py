import pytest
from unittest.mock import AsyncMock, PropertyMock

from app.chunker import DocumentChunker
from app.models import SummarizeConfig
from app.providers.base import SumResult
from app.summarizer import summarize_document


def _make_provider(max_tokens: int = 200_000):
    """Create a mock provider with configurable context window."""
    provider = AsyncMock()
    type(provider).max_context_tokens = PropertyMock(return_value=max_tokens)
    type(provider).name = PropertyMock(return_value="mock-model")
    return provider


def _make_sum_result(summary: str, input_tokens: int = 100, output_tokens: int = 50):
    return SumResult(
        summary=summary,
        detected_language="en",
        code_switching_detected=False,
        model_used="mock-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class TestSummarizeDocument:

    @pytest.mark.asyncio
    async def test_short_text_no_chunking(self):
        """Text within context window goes directly to provider.summarize()."""
        provider = _make_provider(max_tokens=200_000)
        provider.summarize.return_value = _make_sum_result("Short summary.")
        config = SummarizeConfig()

        result = await summarize_document("Short text.", config, provider)

        provider.summarize.assert_called_once()
        provider.reduce.assert_not_called()
        assert result.summary == "Short summary."
        assert result.chunks_used is None

    @pytest.mark.asyncio
    async def test_long_text_triggers_map_reduce(self):
        """Text exceeding context window triggers chunking + map + reduce."""
        provider = _make_provider(max_tokens=100)  # Very small budget to force chunking
        provider.summarize.return_value = _make_sum_result("Section summary.", input_tokens=50, output_tokens=20)
        provider.reduce.return_value = _make_sum_result("Final unified summary.", input_tokens=80, output_tokens=30)
        config = SummarizeConfig()

        long_text = "First paragraph. " * 50 + "\n\n" + "Second paragraph. " * 50
        result = await summarize_document(long_text, config, provider)

        assert provider.summarize.call_count >= 2  # At least 2 map calls
        provider.reduce.assert_called_once()
        assert result.summary == "Final unified summary."
        assert result.chunks_used >= 2

    @pytest.mark.asyncio
    async def test_tokens_aggregated_across_calls(self):
        """Input/output tokens should be summed across map + reduce calls."""
        provider = _make_provider(max_tokens=100)
        provider.summarize.return_value = _make_sum_result("Chunk.", input_tokens=50, output_tokens=20)
        provider.reduce.return_value = _make_sum_result("Final.", input_tokens=80, output_tokens=30)
        config = SummarizeConfig()

        long_text = "Para one. " * 50 + "\n\n" + "Para two. " * 50
        result = await summarize_document(long_text, config, provider)

        # Tokens = sum of all map calls + reduce call
        assert result.input_tokens > 80  # More than just the reduce call
        assert result.output_tokens > 30

    @pytest.mark.asyncio
    async def test_context_bridge_included_after_first_chunk(self):
        """Chunks after the first should include context from the previous summary."""
        provider = _make_provider(max_tokens=100)
        provider.summarize.return_value = _make_sum_result("Previous context.", input_tokens=50, output_tokens=20)
        provider.reduce.return_value = _make_sum_result("Final.", input_tokens=80, output_tokens=30)
        config = SummarizeConfig()

        long_text = "Para one content. " * 50 + "\n\n" + "Para two content. " * 50
        result = await summarize_document(long_text, config, provider)

        # Second summarize call should have context bridge in the text
        second_call_text = provider.summarize.call_args_list[1][0][0]
        assert "[Context:" in second_call_text
