import json
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.models import SummarizeConfig, EvaluationScores


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned malformed JSON: {e}. Response was: {text[:200]}"
        ) from e


@dataclass
class SumResult:
    summary: str
    detected_language: str
    code_switching_detected: bool
    model_used: str
    input_tokens: int
    output_tokens: int


@runtime_checkable
class SummarizationProvider(Protocol):
    async def summarize(self, text: str, config: SummarizeConfig) -> SumResult: ...
    async def reduce(self, section_summaries: list[str], config: SummarizeConfig) -> SumResult: ...
    async def evaluate(
        self, source: str, summary: str, target_language: str
    ) -> EvaluationScores: ...

    @property
    def name(self) -> str: ...

    @property
    def max_context_tokens(self) -> int: ...
