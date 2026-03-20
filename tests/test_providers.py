import pytest
from app.providers.base import SummarizationProvider, SumResult
from app.models import SummarizeConfig, EvaluationScores


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
