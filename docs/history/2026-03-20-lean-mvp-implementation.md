# Lean MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working multilingual summarization API (TXT input, Anthropic Claude, optional LLM-as-Judge evaluation) that can be demo'd live in an interview.

**Architecture:** FastAPI service with a provider-agnostic `SummarizationProvider` protocol. Anthropic Claude is the working provider, ILMU is stubbed. Versioned YAML prompts for BM/EN/auto. Optional per-request evaluation via LLM-as-Judge.

**Tech Stack:** Python 3.11+, FastAPI, Anthropic SDK, PyYAML, pydantic-settings

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/providers/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create `pyproject.toml`**

```toml
[project]
name = "multilingual-summarizer"
version = "0.1.0"
description = "Multilingual document summarization service"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "pydantic-settings>=2.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "pytest-httpx>=0.30.0",
    "ruff>=0.6.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
```

**Step 2: Create `.env.example`**

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
PROVIDER=anthropic
DEFAULT_MODEL=claude-sonnet-4-20250514
LOG_LEVEL=info
```

**Step 3: Create package `__init__.py` files**

Create empty `app/__init__.py`, `app/providers/__init__.py`, `tests/__init__.py`.

**Step 4: Install dependencies**

Run: `pip install -e ".[dev]"`
Expected: Successful install, no errors.

**Step 5: Commit**

```bash
git add pyproject.toml .env.example app/__init__.py app/providers/__init__.py tests/__init__.py
git commit -m "feat: scaffold project with dependencies"
```

---

### Task 2: Config Module

**Files:**
- Create: `tests/test_config.py`
- Create: `app/config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest
from app.config import Settings


def test_settings_defaults():
    settings = Settings(ANTHROPIC_API_KEY="sk-test-key")
    assert settings.PROVIDER == "anthropic"
    assert settings.DEFAULT_MODEL == "claude-sonnet-4-20250514"
    assert settings.LOG_LEVEL == "info"


def test_settings_requires_api_key():
    # Clear env var if set
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(Exception):
        Settings()


def test_settings_provider_validation():
    settings = Settings(ANTHROPIC_API_KEY="sk-test", PROVIDER="ilmu")
    assert settings.PROVIDER == "ilmu"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

**Step 3: Write minimal implementation**

```python
# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    PROVIDER: str = "anthropic"
    DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
    LOG_LEVEL: str = "info"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add config module with pydantic-settings"
```

---

### Task 3: Pydantic Request/Response Models

**Files:**
- Create: `tests/test_models.py`
- Create: `app/models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
import pytest
from app.models import (
    SummarizeRequest,
    SummarizeConfig,
    SummarizeResponse,
    SummaryMetadata,
    EvaluationScores,
    HealthResponse,
)


def test_summarize_request_defaults():
    req = SummarizeRequest(document="Hello world")
    assert req.document_type == "txt"
    assert req.config.target_language == "auto"
    assert req.config.summary_type == "brief"
    assert req.config.max_length == 500
    assert req.config.preserve_domain_terms is True
    assert req.config.evaluate is False


def test_summarize_request_custom_config():
    req = SummarizeRequest(
        document="Teks dalam Bahasa Melayu",
        document_type="txt",
        config=SummarizeConfig(
            target_language="ms",
            summary_type="executive",
            max_length=200,
            evaluate=True,
        ),
    )
    assert req.config.target_language == "ms"
    assert req.config.summary_type == "executive"
    assert req.config.max_length == 200
    assert req.config.evaluate is True


def test_summarize_config_validates_language():
    with pytest.raises(Exception):
        SummarizeConfig(target_language="fr")


def test_summarize_config_validates_summary_type():
    with pytest.raises(Exception):
        SummarizeConfig(summary_type="ultra")


def test_summarize_response_without_evaluation():
    resp = SummarizeResponse(
        summary="A summary",
        metadata=SummaryMetadata(
            detected_language="en",
            code_switching_detected=False,
            model_used="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            latency_ms=800,
        ),
    )
    assert resp.metadata.evaluation is None


def test_summarize_response_with_evaluation():
    scores = EvaluationScores(
        faithfulness=4.5,
        coherence=4.0,
        coverage=3.8,
        language_quality=4.2,
        conciseness=4.0,
        justification="Good summary overall.",
    )
    resp = SummarizeResponse(
        summary="A summary",
        metadata=SummaryMetadata(
            detected_language="ms",
            code_switching_detected=True,
            model_used="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1200,
            evaluation=scores,
        ),
    )
    assert resp.metadata.evaluation.faithfulness == 4.5


def test_health_response():
    health = HealthResponse(
        status="healthy",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
    )
    assert health.status == "healthy"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# app/models.py
from typing import Literal, Optional
from pydantic import BaseModel


class SummarizeConfig(BaseModel):
    target_language: Literal["en", "ms", "auto"] = "auto"
    summary_type: Literal["brief", "detailed", "executive"] = "brief"
    max_length: int = 500
    preserve_domain_terms: bool = True
    evaluate: bool = False


class SummarizeRequest(BaseModel):
    document: str
    document_type: Literal["txt"] = "txt"
    config: SummarizeConfig = SummarizeConfig()


class EvaluationScores(BaseModel):
    faithfulness: float
    coherence: float
    coverage: float
    language_quality: float
    conciseness: float
    justification: str


class SummaryMetadata(BaseModel):
    detected_language: str
    code_switching_detected: bool
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    evaluation: Optional[EvaluationScores] = None


class SummarizeResponse(BaseModel):
    summary: str
    metadata: SummaryMetadata


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add Pydantic request/response models"
```

---

### Task 4: Document Processor (TXT)

**Files:**
- Create: `tests/test_processor.py`
- Create: `app/processor.py`

**Step 1: Write the failing test**

```python
# tests/test_processor.py
import pytest
from app.processor import DocumentProcessor


def test_process_txt_basic():
    processor = DocumentProcessor()
    result = processor.process("Hello world", doc_type="txt")
    assert result.text == "Hello world"


def test_process_txt_unicode_normalization():
    """NFC normalization: decomposed é (e + combining accent) -> composed é"""
    processor = DocumentProcessor()
    decomposed = "caf\u0065\u0301"  # e + combining acute accent
    result = processor.process(decomposed, doc_type="txt")
    assert result.text == "caf\u00e9"  # composed é


def test_process_txt_whitespace_cleanup():
    processor = DocumentProcessor()
    messy = "  Hello   world  \n\n\n  foo  "
    result = processor.process(messy, doc_type="txt")
    assert "  " not in result.text.replace("\n\n", "")
    assert result.text.strip() == result.text


def test_process_txt_empty_raises():
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="empty"):
        processor.process("", doc_type="txt")


def test_process_txt_whitespace_only_raises():
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="empty"):
        processor.process("   \n\n  ", doc_type="txt")


def test_process_unsupported_type_raises():
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="Unsupported"):
        processor.process("data", doc_type="pdf")


def test_process_returns_token_estimate():
    processor = DocumentProcessor()
    result = processor.process("Hello world this is a test", doc_type="txt")
    assert result.estimated_tokens > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_processor.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# app/processor.py
import re
import unicodedata
from dataclasses import dataclass


@dataclass
class ProcessedDocument:
    text: str
    estimated_tokens: int


class DocumentProcessor:
    SUPPORTED_TYPES = {"txt"}

    def process(self, raw: str, doc_type: str) -> ProcessedDocument:
        if doc_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported document type: {doc_type}")

        text = self._normalize(raw)

        if not text:
            raise ValueError("Document is empty after processing")

        return ProcessedDocument(
            text=text,
            estimated_tokens=self._estimate_tokens(text),
        )

    def _normalize(self, text: str) -> str:
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)
        # Collapse multiple spaces to single
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _estimate_tokens(self, text: str) -> int:
        # Rough estimate: ~4 chars per token for English, ~3 for Malay
        return max(1, len(text) // 4)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_processor.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add app/processor.py tests/test_processor.py
git commit -m "feat: add document processor with TXT support and Unicode normalization"
```

---

### Task 5: Prompt Loader & YAML Templates

**Files:**
- Create: `tests/test_prompt_loader.py`
- Create: `app/prompt_loader.py`
- Create: `app/prompts/summarize_en_v1.yaml`
- Create: `app/prompts/summarize_ms_v1.yaml`
- Create: `app/prompts/summarize_auto_v1.yaml`
- Create: `app/prompts/evaluate_v1.yaml`

**Step 1: Create YAML prompt templates**

```yaml
# app/prompts/summarize_en_v1.yaml
version: "1.0.0"
language: "en"
template: |
  You are an expert document summarization assistant.

  Instructions:
  - Summarize the following document clearly and concisely.
  - Preserve domain-specific terminology without simplification.
  - Maintain any Malay terms that appear as proper nouns, legal references, or established terminology.
  - Target length: {max_length} words.
  - Summary type: {summary_type}
  - Preserve domain terms: {preserve_domain_terms}

  Document:
  {document_text}

  Return ONLY valid JSON in this exact format:
  {{"summary": "your summary here", "detected_language": "en", "code_switching_detected": false}}
```

```yaml
# app/prompts/summarize_ms_v1.yaml
version: "1.0.0"
language: "ms"
template: |
  Anda adalah pembantu AI yang pakar dalam meringkaskan dokumen dalam Bahasa Melayu.

  Arahan:
  - Ringkaskan dokumen berikut dalam Bahasa Melayu yang formal dan jelas.
  - Kekalkan istilah teknikal Inggeris yang lazim digunakan (contoh: "API", "cloud computing").
  - Jangan terjemahkan nama khas, istilah undang-undang, atau rujukan seksyen.
  - Panjang ringkasan: {max_length} perkataan.
  - Jenis ringkasan: {summary_type}
  - Kekalkan istilah domain: {preserve_domain_terms}

  Dokumen:
  {document_text}

  Kembalikan HANYA JSON yang sah dalam format ini:
  {{"summary": "ringkasan anda di sini", "detected_language": "ms", "code_switching_detected": false}}
```

```yaml
# app/prompts/summarize_auto_v1.yaml
version: "1.0.0"
language: "auto"
template: |
  You are an expert multilingual document summarization assistant specializing in English and Bahasa Melayu.

  Instructions:
  - First, detect the primary language of the document (English or Bahasa Melayu).
  - Detect if the document contains code-switching (mixing of English and Malay, also known as Manglish).
  - Summarize the document in its primary language.
  - If code-switching is detected, summarize in the dominant language while preserving key terms from the other language.
  - Preserve domain-specific terminology without simplification.
  - Target length: {max_length} words.
  - Summary type: {summary_type}
  - Preserve domain terms: {preserve_domain_terms}

  Document:
  {document_text}

  Return ONLY valid JSON in this exact format:
  {{"summary": "your summary here", "detected_language": "en or ms", "code_switching_detected": true or false}}
```

```yaml
# app/prompts/evaluate_v1.yaml
version: "1.0.0"
template: |
  You are evaluating a document summary. Score each dimension 1-5.
  Return ONLY valid JSON.

  Source document (first 2000 chars):
  {source_excerpt}

  Summary:
  {summary}

  Target language: {target_language}

  Evaluate:
  1. faithfulness: Are all claims in the summary grounded in the source? (5 = fully grounded, 1 = hallucinated claims)
  2. coherence: Does the summary read as a well-structured, logical piece? (5 = excellent flow, 1 = disjointed)
  3. coverage: Are the key points of the source captured? (5 = comprehensive, 1 = critical omissions)
  4. language_quality: Is the grammar, register, and terminology appropriate? For BM: is it bahasa baku? Are English terms preserved correctly? (5 = native quality, 1 = unnatural)
  5. conciseness: Is the summary appropriately compressed without losing information? (5 = optimal, 1 = redundant or too sparse)

  Return: {{"faithfulness": N, "coherence": N, "coverage": N, "language_quality": N, "conciseness": N, "justification": "one sentence"}}
```

**Step 2: Write the failing test**

```python
# tests/test_prompt_loader.py
import pytest
from app.prompt_loader import PromptLoader


def test_load_summarize_en():
    loader = PromptLoader()
    prompt = loader.load("summarize", language="en")
    assert "summarization assistant" in prompt["template"]
    assert prompt["version"] == "1.0.0"


def test_load_summarize_ms():
    loader = PromptLoader()
    prompt = loader.load("summarize", language="ms")
    assert "Bahasa Melayu" in prompt["template"]


def test_load_summarize_auto():
    loader = PromptLoader()
    prompt = loader.load("summarize", language="auto")
    assert "multilingual" in prompt["template"]


def test_load_evaluate():
    loader = PromptLoader()
    prompt = loader.load("evaluate")
    assert "faithfulness" in prompt["template"]


def test_render_template():
    loader = PromptLoader()
    prompt = loader.load("summarize", language="en")
    rendered = loader.render(prompt["template"], {
        "max_length": 200,
        "summary_type": "brief",
        "preserve_domain_terms": True,
        "document_text": "Some document text here.",
    })
    assert "200" in rendered
    assert "brief" in rendered
    assert "Some document text here." in rendered


def test_load_nonexistent_raises():
    loader = PromptLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent", language="en")
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_prompt_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write minimal implementation**

```python
# app/prompt_loader.py
from pathlib import Path
from typing import Optional

import yaml


class PromptLoader:
    def __init__(self, prompts_dir: Optional[Path] = None):
        self._dir = prompts_dir or Path(__file__).parent / "prompts"

    def load(self, name: str, language: Optional[str] = None) -> dict:
        if language:
            filename = f"{name}_{language}_v1.yaml"
        else:
            filename = f"{name}_v1.yaml"

        path = self._dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        with open(path) as f:
            return yaml.safe_load(f)

    def render(self, template: str, variables: dict) -> str:
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_prompt_loader.py -v`
Expected: 6 passed

**Step 6: Commit**

```bash
git add app/prompt_loader.py app/prompts/ tests/test_prompt_loader.py
git commit -m "feat: add prompt loader with versioned YAML templates for EN/MS/auto"
```

---

### Task 6: Provider Protocol & ILMU Stub

**Files:**
- Create: `tests/test_providers.py`
- Create: `app/providers/base.py`
- Create: `app/providers/ilmu.py`

**Step 1: Write the failing test**

```python
# tests/test_providers.py
import pytest
from app.providers.base import SummarizationProvider
from app.providers.ilmu import ILMUProvider
from app.models import SummarizeConfig


@pytest.mark.asyncio
async def test_ilmu_stub_summarize():
    provider = ILMUProvider()
    config = SummarizeConfig(target_language="ms", summary_type="brief", max_length=100)
    result = await provider.summarize("Teks contoh dalam Bahasa Melayu.", config)
    assert result.summary
    assert result.detected_language == "ms"
    assert result.model_used == "ilmu-stub"


@pytest.mark.asyncio
async def test_ilmu_stub_evaluate():
    provider = ILMUProvider()
    result = await provider.evaluate(
        source="Some source text",
        summary="A summary",
        target_language="en",
    )
    assert result.faithfulness >= 1
    assert result.faithfulness <= 5
    assert result.justification


def test_ilmu_provider_name():
    provider = ILMUProvider()
    assert provider.name == "ilmu-stub"


def test_ilmu_provider_max_tokens():
    provider = ILMUProvider()
    assert provider.max_context_tokens > 0


def test_ilmu_implements_protocol():
    """Verify ILMUProvider satisfies the SummarizationProvider protocol."""
    provider = ILMUProvider()
    assert isinstance(provider, SummarizationProvider)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# app/providers/base.py
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
```

```python
# app/providers/ilmu.py
from app.models import SummarizeConfig, EvaluationScores
from app.providers.base import SumResult


class ILMUProvider:
    """Stub provider for ILMU. Returns realistic mock data."""

    @property
    def name(self) -> str:
        return "ilmu-stub"

    @property
    def max_context_tokens(self) -> int:
        return 32_000

    async def summarize(self, text: str, config: SummarizeConfig) -> SumResult:
        lang = config.target_language if config.target_language != "auto" else "ms"
        return SumResult(
            summary=f"[ILMU STUB] Summary of {len(text)} chars in {lang}. "
                    f"Type: {config.summary_type}, max: {config.max_length} words.",
            detected_language=lang,
            code_switching_detected=False,
            model_used="ilmu-stub",
            input_tokens=len(text) // 4,
            output_tokens=50,
        )

    async def evaluate(self, source: str, summary: str, target_language: str) -> EvaluationScores:
        return EvaluationScores(
            faithfulness=4.0,
            coherence=4.0,
            coverage=4.0,
            language_quality=4.0,
            conciseness=4.0,
            justification="[ILMU STUB] Evaluation placeholder.",
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add app/providers/base.py app/providers/ilmu.py tests/test_providers.py
git commit -m "feat: add provider protocol and ILMU stub"
```

---

### Task 7: Anthropic Provider

**Files:**
- Create: `app/providers/anthropic.py`
- Modify: `tests/test_providers.py` (add Anthropic tests)

**Step 1: Write the failing test**

Append to `tests/test_providers.py`:

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch
from app.providers.anthropic import AnthropicProvider


def _mock_anthropic_response(content_text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock Anthropic API response."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=content_text)]
    mock_resp.usage.input_tokens = input_tokens
    mock_resp.usage.output_tokens = output_tokens
    return mock_resp


@pytest.mark.asyncio
async def test_anthropic_summarize():
    mock_response = _mock_anthropic_response(
        json.dumps({
            "summary": "This is a test summary.",
            "detected_language": "en",
            "code_switching_detected": False,
        })
    )

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        config = SummarizeConfig(target_language="en", summary_type="brief", max_length=100)
        result = await provider.summarize("Test document text.", config)

    assert result.summary == "This is a test summary."
    assert result.detected_language == "en"
    assert result.code_switching_detected is False
    assert result.model_used == "claude-sonnet-4-20250514"
    assert result.input_tokens == 100
    assert result.output_tokens == 50


@pytest.mark.asyncio
async def test_anthropic_summarize_auto_language():
    mock_response = _mock_anthropic_response(
        json.dumps({
            "summary": "Ringkasan dokumen ini.",
            "detected_language": "ms",
            "code_switching_detected": True,
        })
    )

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        config = SummarizeConfig(target_language="auto", summary_type="brief", max_length=100)
        result = await provider.summarize("Dokumen campuran with English.", config)

    assert result.detected_language == "ms"
    assert result.code_switching_detected is True


@pytest.mark.asyncio
async def test_anthropic_evaluate():
    mock_response = _mock_anthropic_response(
        json.dumps({
            "faithfulness": 4.5,
            "coherence": 4.0,
            "coverage": 3.8,
            "language_quality": 4.2,
            "conciseness": 4.0,
            "justification": "Good summary with minor gaps.",
        })
    )

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.evaluate(
            source="Original document text here.",
            summary="A brief summary.",
            target_language="en",
        )

    assert result.faithfulness == 4.5
    assert result.justification == "Good summary with minor gaps."


def test_anthropic_provider_name():
    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")
    assert provider.name == "claude-sonnet-4-20250514"


def test_anthropic_provider_max_tokens():
    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")
    assert provider.max_context_tokens == 200_000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.providers.anthropic'`

**Step 3: Write minimal implementation**

```python
# app/providers/anthropic.py
import json
import time

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers.py -v`
Expected: 10 passed

**Step 5: Commit**

```bash
git add app/providers/anthropic.py tests/test_providers.py
git commit -m "feat: add Anthropic provider with summarize and evaluate"
```

---

### Task 8: Evaluation Module

**Files:**
- Create: `tests/test_evaluation.py`
- Create: `app/evaluation.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add app/evaluation.py tests/test_evaluation.py
git commit -m "feat: add evaluation module with source truncation"
```

---

### Task 9: FastAPI App & Routes

**Files:**
- Create: `app/main.py`
- Create: `tests/test_main.py`

**Step 1: Write the failing test**

```python
# tests/test_main.py
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.providers.base import SumResult
from app.models import EvaluationScores


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.name = "test-model"
    provider.max_context_tokens = 200_000
    provider.summarize = AsyncMock(return_value=SumResult(
        summary="Test summary output.",
        detected_language="en",
        code_switching_detected=False,
        model_used="test-model",
        input_tokens=100,
        output_tokens=50,
    ))
    provider.evaluate = AsyncMock(return_value=EvaluationScores(
        faithfulness=4.5, coherence=4.0, coverage=3.8,
        language_quality=4.2, conciseness=4.0,
        justification="Good summary.",
    ))
    return provider


@pytest.fixture
def app(mock_provider):
    return create_app(provider=mock_provider)


@pytest.mark.asyncio
async def test_health_endpoint(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_summarize_basic(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "This is a test document about Malaysian technology.",
            "document_type": "txt",
            "config": {
                "target_language": "en",
                "summary_type": "brief",
                "max_length": 100,
            },
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] == "Test summary output."
    assert data["metadata"]["detected_language"] == "en"
    assert data["metadata"]["evaluation"] is None


@pytest.mark.asyncio
async def test_summarize_with_evaluation(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "Test document content.",
            "config": {"evaluate": True},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["evaluation"] is not None
    assert data["metadata"]["evaluation"]["faithfulness"] == 4.5


@pytest.mark.asyncio
async def test_summarize_without_evaluation(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "Test document content.",
            "config": {"evaluate": False},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["evaluation"] is None
    mock_provider.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_empty_document(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "",
        })
    assert resp.status_code == 422 or resp.status_code == 400


@pytest.mark.asyncio
async def test_summarize_default_config(app, mock_provider):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "Just a document with defaults.",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["evaluation"] is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `cannot import name 'create_app' from 'app.main'`

**Step 3: Write minimal implementation**

```python
# app/main.py
import logging
import time

from fastapi import FastAPI, HTTPException

from app.evaluation import evaluate_summary
from app.models import (
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
    SummaryMetadata,
)
from app.processor import DocumentProcessor
from app.providers.anthropic import AnthropicProvider
from app.providers.ilmu import ILMUProvider
from app.config import Settings


logger = logging.getLogger(__name__)


def create_app(provider=None) -> FastAPI:
    app = FastAPI(
        title="Multilingual Document Summarization API",
        version="0.1.0",
    )
    processor = DocumentProcessor()

    if provider is None:
        settings = Settings()
        if settings.PROVIDER == "ilmu":
            provider = ILMUProvider()
        else:
            provider = AnthropicProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.DEFAULT_MODEL,
            )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            provider=provider.name,
            model=provider.name,
        )

    @app.post("/v1/summarize", response_model=SummarizeResponse)
    async def summarize(request: SummarizeRequest):
        start = time.time()

        try:
            doc = processor.process(request.document, request.document_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = await provider.summarize(doc.text, request.config)

        latency_ms = int((time.time() - start) * 1000)

        evaluation = None
        if request.config.evaluate:
            evaluation = await evaluate_summary(
                provider=provider,
                source=request.document,
                summary=result.summary,
                target_language=result.detected_language,
            )

        return SummarizeResponse(
            summary=result.summary,
            metadata=SummaryMetadata(
                detected_language=result.detected_language,
                code_switching_detected=result.code_switching_detected,
                model_used=result.model_used,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=latency_ms,
                evaluation=evaluation,
            ),
        )

    return app


app = create_app()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: add FastAPI app with summarize and health endpoints"
```

---

### Task 10: Dockerfile & Final Wiring

**Files:**
- Create: `Dockerfile`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Run full test suite**

Run: `pytest -v`
Expected: All tests pass (approx 30 tests)

**Step 3: Test the app manually**

Run: `ANTHROPIC_API_KEY=sk-test PROVIDER=ilmu uvicorn app.main:app --reload`

Then in another terminal:
```bash
curl http://localhost:8000/v1/health

curl -X POST http://localhost:8000/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "document": "Malaysia telah melancarkan model AI pertama negara yang dikenali sebagai ILMU.",
    "document_type": "txt",
    "config": {
      "target_language": "ms",
      "summary_type": "brief",
      "max_length": 100
    }
  }'
```

Expected: 200 OK with stub responses.

**Step 4: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for containerized deployment"
```

---

### Task 11: Run Full Suite & Final Commit

**Step 1: Run all tests one final time**

Run: `pytest -v --tb=short`
Expected: All pass

**Step 2: Run linter**

Run: `ruff check app/ tests/`
Expected: No errors (or fix any that appear)

**Step 3: Final commit if any fixes**

```bash
git add -A
git commit -m "chore: lint fixes and final cleanup"
```
