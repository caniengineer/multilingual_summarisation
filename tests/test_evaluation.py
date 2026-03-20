# tests/test_evaluation.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.evaluation import evaluate_summary
from app.providers.anthropic import AnthropicProvider
from app.models import EvaluationScores


@pytest.mark.asyncio
async def test_evaluate_summary():
    mock_scores = EvaluationScores(
        faithfulness=4.5,
        coherence=4.0,
        coverage=3.8,
        language_quality=4.2,
        conciseness=4.0,
        justification="Solid summary.",
    )

    mock_provider = AsyncMock()
    mock_provider.evaluate = AsyncMock(return_value=mock_scores)

    result = await evaluate_summary(
        provider=mock_provider,
        source="Original document text.",
        summary="A summary.",
        target_language="en",
    )

    assert result.faithfulness == 4.5
    assert result.justification == "Solid summary."
    mock_provider.evaluate.assert_called_once_with(
        source="Original document text.",
        summary="A summary.",
        target_language="en",
    )


@pytest.mark.asyncio
async def test_evaluate_summary_truncates_source():
    mock_provider = AsyncMock()
    mock_provider.evaluate = AsyncMock(return_value=EvaluationScores(
        faithfulness=4.0, coherence=4.0, coverage=4.0,
        language_quality=4.0, conciseness=4.0, justification="OK",
    ))

    long_source = "x" * 5000
    await evaluate_summary(
        provider=mock_provider,
        source=long_source,
        summary="Summary.",
        target_language="en",
    )

    # Verify the source was truncated to 2000 chars
    call_args = mock_provider.evaluate.call_args
    assert len(call_args.kwargs["source"]) == 2000
