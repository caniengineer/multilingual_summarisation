# Claude Code Local Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `ClaudeCodeProvider` that calls the local `claude` CLI via subprocess, and wire up provider selection via the `PROVIDER` config field.

**Architecture:** New provider class in `app/providers/claude_code.py` using `asyncio.create_subprocess_exec` to call `claude -p --output-format json --model <model>`. Provider selection in `main.py` via simple if/elif on `settings.PROVIDER`. Config updated to make `ANTHROPIC_API_KEY` optional when using `claude-code`.

**Tech Stack:** Python asyncio subprocess, existing PromptLoader, pydantic-settings

---

### Task 1: Create ClaudeCodeProvider with TDD

**Files:**
- Create: `app/providers/claude_code.py`
- Test: `tests/test_providers.py`

**Step 1: Write the protocol conformance test**

Add to `tests/test_providers.py`:

```python
from app.providers.claude_code import ClaudeCodeProvider


def test_claude_code_provider_satisfies_protocol():
    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")
    assert isinstance(provider, SummarizationProvider)


def test_claude_code_provider_name():
    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")
    assert provider.name == "claude-code/claude-sonnet-4-20250514"


def test_claude_code_provider_max_tokens():
    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")
    assert provider.max_context_tokens == 200_000
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py::test_claude_code_provider_satisfies_protocol -v`
Expected: FAIL with `ImportError` — module doesn't exist yet.

**Step 3: Write minimal ClaudeCodeProvider skeleton**

Create `app/providers/claude_code.py`:

```python
import asyncio
import json
import re

from app.models import SummarizeConfig, EvaluationScores
from app.prompt_loader import PromptLoader
from app.providers.base import SumResult


def _parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned malformed JSON: {e}. Response was: {text[:200]}"
        ) from e


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
            "claude", "-p",
            "--output-format", "json",
            "--model", self._model,
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
        prompt_data = self._prompt_loader.load("summarize", language=config.target_language)
        rendered = self._prompt_loader.render(prompt_data["template"], {
            "max_length": config.max_length,
            "summary_type": config.summary_type,
            "preserve_domain_terms": config.preserve_domain_terms,
            "document_text": text,
        })

        response = await self._call_claude(rendered)
        result = _parse_llm_json(response["result"])

        return SumResult(
            summary=result["summary"],
            detected_language=result["detected_language"],
            code_switching_detected=result["code_switching_detected"],
            model_used=self._model,
            input_tokens=response.get("usage", {}).get("input_tokens", 0),
            output_tokens=response.get("usage", {}).get("output_tokens", 0),
        )

    async def evaluate(self, source: str, summary: str, target_language: str) -> EvaluationScores:
        prompt_data = self._prompt_loader.load("evaluate")
        rendered = self._prompt_loader.render(prompt_data["template"], {
            "source_excerpt": source,
            "summary": summary,
            "target_language": target_language,
        })

        response = await self._call_claude(rendered)
        result = _parse_llm_json(response["result"])

        return EvaluationScores(**result)
```

**Step 4: Run protocol tests to verify they pass**

Run: `python -m pytest tests/test_providers.py::test_claude_code_provider_satisfies_protocol tests/test_providers.py::test_claude_code_provider_name tests/test_providers.py::test_claude_code_provider_max_tokens -v`
Expected: All 3 PASS.

**Step 5: Commit**

```bash
git add app/providers/claude_code.py tests/test_providers.py
git commit -m "feat: add ClaudeCodeProvider skeleton satisfying SummarizationProvider protocol"
```

---

### Task 2: Test summarize and evaluate via mocked subprocess

**Files:**
- Modify: `tests/test_providers.py`

**Step 1: Write the summarize test**

Add to `tests/test_providers.py`:

```python
from unittest.mock import patch


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
```

**Step 2: Run to verify it passes**

Run: `python -m pytest tests/test_providers.py::test_claude_code_summarize -v`
Expected: PASS

**Step 3: Write the evaluate test**

```python
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
```

**Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_providers.py::test_claude_code_evaluate -v`
Expected: PASS

**Step 5: Write error handling tests**

```python
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
```

**Step 6: Run all Claude Code tests**

Run: `python -m pytest tests/test_providers.py -k "claude_code" -v`
Expected: All PASS.

**Step 7: Commit**

```bash
git add tests/test_providers.py
git commit -m "test: add unit tests for ClaudeCodeProvider summarize, evaluate, and error handling"
```

---

### Task 3: Update config to support provider selection

**Files:**
- Modify: `app/config.py`

**Step 1: Write failing test for optional API key**

Add to `tests/test_providers.py` (or a new `tests/test_config.py` if preferred):

```python
import os
from app.config import Settings


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
```

**Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_providers.py::test_settings_claude_code_no_api_key_required -v`
Expected: FAIL — `ANTHROPIC_API_KEY` is currently required.

**Step 3: Update `app/config.py`**

```python
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: Optional[str] = None
    PROVIDER: str = "anthropic"
    DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_CODE_MODEL: str = "claude-sonnet-4-20250514"
    LOG_LEVEL: str = "info"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

**Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_providers.py::test_settings_claude_code_no_api_key_required tests/test_providers.py::test_settings_claude_code_model_default -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/config.py tests/test_providers.py
git commit -m "feat: make ANTHROPIC_API_KEY optional, add CLAUDE_CODE_MODEL config"
```

---

### Task 4: Wire up provider selection in main.py

**Files:**
- Modify: `app/main.py`

**Step 1: Write failing test for provider selection**

Add to `tests/test_main.py`:

```python
from unittest.mock import patch

def test_create_app_with_claude_code_provider(monkeypatch):
    monkeypatch.setenv("PROVIDER", "claude-code")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from app.main import create_app
    app = create_app()
    # App should be created without error
    assert app is not None


def test_create_app_with_unknown_provider(monkeypatch):
    monkeypatch.setenv("PROVIDER", "unknown")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from app.main import create_app
    with pytest.raises(ValueError, match="Unknown provider"):
        create_app()
```

**Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_main.py::test_create_app_with_claude_code_provider -v`
Expected: FAIL — `main.py` doesn't handle `claude-code` yet.

**Step 3: Update `app/main.py`**

Replace the provider instantiation block (lines 29-34) with:

```python
from app.providers.claude_code import ClaudeCodeProvider

# ... in create_app():
    if provider is None:
        settings = Settings()
        if settings.PROVIDER == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is required when PROVIDER=anthropic")
            provider = AnthropicProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.DEFAULT_MODEL,
            )
        elif settings.PROVIDER == "claude-code":
            provider = ClaudeCodeProvider(
                model=settings.CLAUDE_CODE_MODEL,
            )
        else:
            raise ValueError(f"Unknown provider: {settings.PROVIDER}")
```

**Step 4: Run to verify tests pass**

Run: `python -m pytest tests/test_main.py -v`
Expected: All PASS (existing tests still work, new tests pass).

**Step 5: Run full test suite**

Run: `python -m pytest -v`
Expected: All PASS.

**Step 6: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: wire up PROVIDER config to select between anthropic and claude-code"
```

---

### Task 5: Verify end-to-end and run full test suite

**Step 1: Run full test suite**

Run: `python -m pytest -v`
Expected: All PASS.

**Step 2: Run linting if configured**

Run: `make lint` (or `ruff check .`)
Expected: No errors.

**Step 3: Quick manual smoke test (optional)**

Set `PROVIDER=claude-code` in `.env`, then:

```bash
python -m uvicorn app.main:create_app --factory --port 8000 &
curl -s http://localhost:8000/v1/health | python -m json.tool
```

Expected: Health response shows `"provider": "claude-code/claude-sonnet-4-20250514"`

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final cleanup for claude-code provider"
```
