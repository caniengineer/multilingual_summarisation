import asyncio
import json

from app.models import SummarizeConfig, EvaluationScores
from app.prompt_loader import PromptLoader
from app.providers.base import SumResult, parse_llm_json


class ClaudeCodeProvider:
    def __init__(self, model: str):
        self._model = model
        self._prompt_loader = PromptLoader()

    @property
    def name(self) -> str:
        return f"claude-code/{self._model}"

    @property
    def max_context_tokens(self) -> int:
        return 200_000

    async def _call_claude(self, prompt: str) -> dict:
        """Call claude CLI and return parsed JSON response."""
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            self._model,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=prompt.encode())

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude exited with code {proc.returncode}: {stderr.decode()}"
            )

        try:
            response = json.loads(stdout.decode())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"claude returned invalid JSON: {e}. Output: {stdout.decode()[:200]}"
            ) from e

        if response.get("is_error"):
            raise RuntimeError(
                f"claude returned an error: {response.get('result', 'unknown error')}"
            )

        return response

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

        response = await self._call_claude(rendered)
        result = parse_llm_json(response["result"])

        return SumResult(
            summary=result["summary"],
            detected_language=result["detected_language"],
            code_switching_detected=result["code_switching_detected"],
            model_used=self._model,
            input_tokens=response.get("usage", {}).get("input_tokens", 0),
            output_tokens=response.get("usage", {}).get("output_tokens", 0),
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

        response = await self._call_claude(rendered)
        result = parse_llm_json(response["result"])

        return EvaluationScores(**result)
