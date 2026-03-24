import asyncio
import time

import anthropic

from app.config import Settings
from app.metrics import LLM_CALL_COUNT, LLM_CALL_LATENCY
from app.models import SummarizeConfig, EvaluationScores
from app.prompt_loader import PromptLoader
from app.providers.base import SumResult, parse_llm_json
from app.resilience import retry_with_backoff


class AnthropicProvider:
    def __init__(self, api_key: str, model: str, settings: Settings | None = None):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._prompt_loader = PromptLoader()
        self._settings = settings

    @property
    def name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        return 200_000

    @property
    def _call_timeout(self) -> int:
        return self._settings.LLM_CALL_TIMEOUT if self._settings else 60

    @property
    def _max_attempts(self) -> int:
        return self._settings.RETRY_MAX_ATTEMPTS if self._settings else 3

    @property
    def _base_delay(self) -> float:
        return self._settings.RETRY_BASE_DELAY if self._settings else 1.0

    @property
    def _max_delay(self) -> float:
        return self._settings.RETRY_MAX_DELAY if self._settings else 30.0

    async def _call_llm(self, messages, max_tokens=4096, method="summarize"):
        start = time.time()

        async def _do_call():
            return await asyncio.wait_for(
                self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=messages,
                ),
                timeout=self._call_timeout,
            )

        try:
            response = await retry_with_backoff(
                _do_call,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
                max_delay=self._max_delay,
            )
            LLM_CALL_COUNT.labels(method=method, status="success").inc()
            return response
        except Exception:
            LLM_CALL_COUNT.labels(method=method, status="error").inc()
            raise
        finally:
            duration = time.time() - start
            LLM_CALL_LATENCY.labels(method=method).observe(duration)

    async def summarize(self, text: str, config: SummarizeConfig) -> SumResult:
        prompt_data = self._prompt_loader.load(
            "summarize", language=config.target_language
        )
        rendered = self._prompt_loader.render(
            prompt_data["template"],
            {
                "max_length": config.max_length,
                "summary_type": config.summary_type,
                "preserve_domain_terms": config.preserve_domain_terms,
                "document_text": text,
            },
        )

        response = await self._call_llm(
            messages=[{"role": "user", "content": rendered}],
            max_tokens=4096,
            method="summarize",
        )

        result = parse_llm_json(response.content[0].text)

        return SumResult(
            summary=result["summary"],
            detected_language=result["detected_language"],
            code_switching_detected=result["code_switching_detected"],
            model_used=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def reduce(self, section_summaries: list[str], config: SummarizeConfig) -> SumResult:
        prompt_data = self._prompt_loader.load("reduce")
        numbered = "\n\n".join(
            f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
        )
        rendered = self._prompt_loader.render(prompt_data["template"], {
            "max_length": config.max_length,
            "summary_type": config.summary_type,
            "section_summaries": numbered,
        })

        response = await self._call_llm(
            messages=[{"role": "user", "content": rendered}],
            max_tokens=4096,
            method="reduce",
        )

        result = parse_llm_json(response.content[0].text)

        return SumResult(
            summary=result["summary"],
            detected_language=result["detected_language"],
            code_switching_detected=result["code_switching_detected"],
            model_used=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def evaluate(
        self, source: str, summary: str, target_language: str
    ) -> EvaluationScores:
        prompt_data = self._prompt_loader.load("evaluate")
        rendered = self._prompt_loader.render(
            prompt_data["template"],
            {
                "source_excerpt": source,
                "summary": summary,
                "target_language": target_language,
            },
        )

        response = await self._call_llm(
            messages=[{"role": "user", "content": rendered}],
            max_tokens=1024,
            method="evaluate",
        )

        result = parse_llm_json(response.content[0].text)
        return EvaluationScores(**result)
