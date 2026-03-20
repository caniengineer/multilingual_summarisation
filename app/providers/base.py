from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.models import SummarizeConfig, EvaluationScores


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
    async def evaluate(self, source: str, summary: str, target_language: str) -> EvaluationScores: ...

    @property
    def name(self) -> str: ...

    @property
    def max_context_tokens(self) -> int: ...
