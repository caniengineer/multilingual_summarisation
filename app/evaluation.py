# app/evaluation.py
from app.models import EvaluationScores

SOURCE_EXCERPT_LIMIT = 2000


async def evaluate_summary(
    provider,
    source: str,
    summary: str,
    target_language: str,
) -> EvaluationScores:
    truncated_source = source[:SOURCE_EXCERPT_LIMIT]
    return await provider.evaluate(
        source=truncated_source,
        summary=summary,
        target_language=target_language,
    )
