import json

import anthropic

from app.models import SummarizeConfig, EvaluationScores
from app.prompt_loader import PromptLoader
from app.providers.base import SumResult


class AnthropicProvider:
    def __init__(self, api_key: str, model: str):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._prompt_loader = PromptLoader()

    @property
    def name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        return 200_000

    async def summarize(self, text: str, config: SummarizeConfig) -> SumResult:
        prompt_data = self._prompt_loader.load("summarize", language=config.target_language)
        rendered = self._prompt_loader.render(prompt_data["template"], {
            "max_length": config.max_length,
            "summary_type": config.summary_type,
            "preserve_domain_terms": config.preserve_domain_terms,
            "document_text": text,
        })

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": rendered}],
        )

        result = json.loads(response.content[0].text)

        return SumResult(
            summary=result["summary"],
            detected_language=result["detected_language"],
            code_switching_detected=result["code_switching_detected"],
            model_used=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def evaluate(self, source: str, summary: str, target_language: str) -> EvaluationScores:
        prompt_data = self._prompt_loader.load("evaluate")
        rendered = self._prompt_loader.render(prompt_data["template"], {
            "source_excerpt": source[:2000],
            "summary": summary,
            "target_language": target_language,
        })

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": rendered}],
        )

        result = json.loads(response.content[0].text)
        return EvaluationScores(**result)
