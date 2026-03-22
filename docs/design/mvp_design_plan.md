# Multilingual Document Summarization Service — MVP Design Plan

**Role:** Backend Platform Engineer — YTL AI Labs
**Author:** [Your Name]
**Date:** March 2026
**Status:** Implementation-ready

---

## 1. Problem Statement

Build a production API that summarizes long documents in English and Bahasa Melayu, handling the code-switching (Manglish) that is pervasive in real Malaysian documents. The system must be deployable, testable, and provider-agnostic — designed so ILMU slots in as the primary model with zero application code changes.

## 2. Design Philosophy

Models improve every 6 months. Infrastructure that wraps models must outlast them.

Three principles guide what we build vs. what we delegate to the model:

| Principle | What it means | Implication |
|-----------|--------------|-------------|
| Thin orchestration | The model is the engine. We build everything around it. | No custom NLP pipelines for tasks LLMs already handle (code-switch detection, language normalization). |
| Durable surfaces | API contracts, evaluation harnesses, and observability outlast any model. | Invest engineering time here, not in map-reduce chains that bigger context windows will erase. |
| Provider-agnostic core | Swapping ILMU v1 → v2, or adding Anthropic as fallback, is a config change + eval run. | Abstract all LLM interaction behind a SummarizationProvider protocol. |

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Service                       │
│                                                         │
│  POST /v1/summarize                                     │
│  POST /v1/summarize/batch  (future)                     │
│  GET  /v1/summarize/jobs/{id}  (future)                 │
│  GET  /v1/health                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────┐  │
│  │ Document  │──▶│ Language  │──▶│  Provider Router   │  │
│  │ Processor │   │ Detector  │   │                    │  │
│  │           │   │           │   │  ILMU (primary)    │  │
│  │ PDF       │   │ fastText  │   │  Anthropic (fallback)│
│  │ DOCX      │   │ single    │   │  OpenAI (fallback) │  │
│  │ TXT/HTML  │   │ pass      │   └────────────────────┘  │
│  └──────────┘   └──────────┘            │                │
│                                         ▼                │
│                              ┌────────────────────┐      │
│                              │  Evaluation Engine  │      │
│                              │  LLM-as-Judge       │      │
│                              │  Latency + Tokens   │      │
│                              │  Quality Drift      │      │
│                              └────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## 4. API Contract

### POST /v1/summarize

```json
// Request
{
  "document": "<base64-encoded or raw text>",
  "document_type": "pdf" | "docx" | "txt" | "html",
  "config": {
    "target_language": "en" | "ms" | "auto",
    "summary_type": "brief" | "detailed" | "executive",
    "max_length": 500,
    "preserve_domain_terms": true
  }
}

// Response
{
  "summary": "...",
  "metadata": {
    "detected_language": "ms",
    "language_confidence": 0.92,
    "code_switching_detected": true,
    "model_used": "ilmu-1.0",
    "input_tokens": 4200,
    "output_tokens": 380,
    "latency_ms": 1240,
    "evaluation": {
      "faithfulness": 4.2,
      "coherence": 4.5,
      "coverage": 3.8
    }
  }
}
```

### GET /v1/health

Returns service status, model availability, and last evaluation scores.

### Design decisions:

- **target_language:** `"auto"` returns the summary in the document's primary language — the model decides, not a pipeline.
- **preserve_domain_terms:** `true` tells the prompt to keep English technical/legal terms in Malay summaries (e.g., "API", "cloud computing", legal citations).
- **Evaluation scores** are returned per-request in MVP. In production, this becomes configurable (sample rate) to manage cost.

## 5. Component Design

### 5.1 Document Processor

**Scope:** Extract clean text from raw document bytes. This is the unglamorous work that won't be replaced by better models — PDFs will still be messy in 2030.

| Format | Library | Edge Cases Handled |
|--------|---------|-------------------|
| PDF | PyMuPDF (fitz) for digital; Tesseract OCR for scanned | Mixed scanned/digital pages; mojibake from Windows-1252 encoding in Malaysian government PDFs |
| DOCX | python-docx | Tracked changes stripped; comments preserved as context |
| HTML | trafilatura | Boilerplate removal; bilingual site content extraction |
| TXT | chardet + ftfy | Encoding detection and repair; Unicode NFC normalization |

**Key implementation details:**
- `ftfy` handles the encoding repair that Malaysian government PDFs frequently need (Windows-1252 / ISO-8859 misinterpretation → mojibake).
- Unicode normalization to NFC form ensures Malay diacritics are handled consistently.
- Document structure (headings, sections) is preserved as metadata and passed to the LLM prompt — section structure is signal, not noise.

```python
# Core interface
class DocumentProcessor:
    async def extract(self, raw: bytes, doc_type: str) -> ProcessedDocument:
        """Returns cleaned text + structural metadata."""
        extractor = self._get_extractor(doc_type)
        raw_text = await extractor.extract(raw)
        cleaned = self._normalize(raw_text)  # NFC, ftfy, whitespace
        return ProcessedDocument(text=cleaned, metadata=extractor.metadata)
```

### 5.2 Language Detector

**Scope:** Lightweight language classification for routing. One pass, not three.

**Why single-pass:** Modern multilingual LLMs (especially ILMU) natively handle code-switched input. We don't need a multi-layered detection pipeline — we need just enough information to (a) route to the right provider and (b) inform the summarization prompt.

```python
class LanguageDetector:
    def __init__(self):
        self.model = fasttext.load_model("lid.176.bin")

    def detect(self, text: str) -> LanguageResult:
        predictions = self.model.predict(text, k=3)
        primary = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]

        # Simple code-switch heuristic: if top-2 are en/ms and close in score
        is_code_switched = (
            {"en", "ms"}.issubset({p.replace("__label__", "") for p in predictions[0][:2]})
            and predictions[1][1] > 0.2
        )

        return LanguageResult(
            primary_language=primary,
            confidence=confidence,
            code_switching_detected=is_code_switched,
        )
```

**Failure mode:** If confidence < 0.6, treat as code-switched and route to a model that handles mixed input natively (ILMU). The confidence score is always returned in the API response so consumers can decide how to react.

### 5.3 Provider Abstraction

**Scope:** This is the core architectural investment. Every LLM interaction goes through this interface. Swapping models is a config change, not a code change.

```python
from typing import Protocol

class SummarizationProvider(Protocol):
    """All LLM providers implement this interface."""

    async def summarize(self, text: str, config: SumConfig, context: DocMetadata) -> SumResult:
        ...

    async def translate(self, text: str, target_lang: str) -> str:
        ...

    async def evaluate(self, source: str, summary: str) -> EvalResult:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def max_context_tokens(self) -> int:
        ...


class ILMUProvider(SummarizationProvider):
    """Primary provider for all BM and code-switched content."""
    ...

class AnthropicProvider(SummarizationProvider):
    """Fallback provider. Used for English-heavy content or when ILMU is unavailable."""
    ...
```

**Model routing logic:**

```python
class ProviderRouter:
    def select(self, lang: LanguageResult, config: SumConfig) -> SummarizationProvider:
        # ILMU is always preferred for BM or code-switched content
        if lang.primary_language == "ms" or lang.code_switching_detected:
            return self.ilmu if self.ilmu.is_available() else self.fallback

        # For English-only, ILMU is still preferred (sovereign infra) but fallback is acceptable
        return self.ilmu if self.ilmu.is_available() else self.fallback
```

### 5.4 Prompt Engineering

Prompts are versioned configuration, not inline strings. Each prompt template is a separate file with a semantic version and metadata.

```yaml
# prompts/summarize_bm_v1.yaml
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

  Konteks dokumen:
  - Jenis: {document_type}
  - Bahasa utama: {detected_language}
  - Campuran bahasa dikesan: {code_switching}

  Dokumen:
  {document_text}
```

```yaml
# prompts/summarize_en_v1.yaml
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

  Document context:
  - Type: {document_type}
  - Primary language: {detected_language}
  - Code-switching detected: {code_switching}

  Document:
  {document_text}
```

**Why separate BM and EN templates:** Effective summarization instructions differ between languages. BM has formal register conventions (bahasa baku vs. bahasa pasar), honorific preservation requirements, and different expectations for how technical terms are handled. A single bilingual prompt underperforms two language-specific ones.

### 5.5 Chunking (Fallback Path Only)

Most documents fit within modern context windows (ILMU and Claude both handle 100K+ tokens). Chunking exists as a fallback, not as the primary path.

```python
async def summarize_document(text: str, provider: SummarizationProvider, config: SumConfig, context: DocMetadata) -> SumResult:
    if token_count(text) <= provider.max_context_tokens * 0.8:
        # Happy path: single call. Most documents land here.
        return await provider.summarize(text, config, context)

    # Fallback: simple two-pass for very long documents
    chunks = split_on_sections(text, context.structure, max_tokens=provider.max_context_tokens * 0.6)
    section_summaries = await asyncio.gather(*[
        provider.summarize(chunk, config._replace(summary_type="detailed"), context)
        for chunk in chunks
    ])
    combined = "\n\n".join(s.text for s in section_summaries)
    return await provider.summarize(combined, config._replace(summary_type=config.summary_type), context)
```

**Why 0.8 threshold:** Reserve 20% of the context window for the system prompt, instructions, and output buffer. Sending a document that exactly fills the context window causes truncation or degraded output quality.

## 6. Evaluation Framework

This is the moat. Anyone can call an LLM API. Knowing whether the output is good — and detecting when quality degrades — is platform value.

### 6.1 LLM-as-Judge (Per-Request)

Every summarization request is evaluated by a second LLM call (configurable sample rate in production to manage cost).

```python
EVAL_PROMPT = """
You are evaluating a document summary. Score each dimension 1-5. Return ONLY valid JSON.

Source document (first 2000 chars): {source_excerpt}
Summary: {summary}
Target language: {target_language}

Evaluate:
1. faithfulness: Are all claims in the summary grounded in the source? (5 = fully grounded, 1 = hallucinated claims)
2. coherence: Does the summary read as a well-structured, logical piece? (5 = excellent flow, 1 = disjointed)
3. coverage: Are the key points of the source captured? (5 = comprehensive, 1 = critical omissions)
4. language_quality: Is the grammar, register, and terminology appropriate? For BM: is it bahasa baku? Are English terms preserved correctly? (5 = native quality, 1 = unnatural)
5. conciseness: Is the summary appropriately compressed without losing information? (5 = optimal, 1 = redundant or too sparse)

Return: {"faithfulness": N, "coherence": N, "coverage": N, "language_quality": N, "conciseness": N, "justification": "one sentence"}
"""
```

### 6.2 Automated Metrics (Batch Evaluation)

Run nightly on a curated test set of 50+ documents (mix of EN, BM, code-switched, government, legal, enterprise):

| Metric | Purpose | Why it matters for BM |
|--------|---------|----------------------|
| chrF++ | Character n-gram overlap | Better than ROUGE for agglutinative Malay morphology |
| BERTScore (XLM-RoBERTa) | Semantic similarity | Cross-lingual scoring handles mixed-language content |
| LLM-as-Judge (5 dimensions) | Holistic quality | Most reliable for BM; catches nuances metrics miss |
| Compression ratio | Sanity check | Flags suspiciously short/long summaries |

### 6.3 Observability

```json
// Every request emits structured logs
{
  "request_id": "uuid",
  "timestamp": "ISO8601",
  "document_type": "pdf",
  "input_tokens": 4200,
  "output_tokens": 380,
  "detected_language": "ms",
  "code_switching": true,
  "provider": "ilmu-1.0",
  "latency_ms": 1240,
  "eval_scores": { "faithfulness": 4.2, "coherence": 4.5, "..." : "..." },
  "error": null
}
```

**Alerts configured for:**
- P95 latency exceeds 5s (by document size bucket)
- Average faithfulness score drops below 3.5 over rolling 24h window
- Error rate by language exceeds 5% (catches language-specific regressions)
- Token usage anomalies (potential prompt injection or malformed input)

## 7. Deployment Strategy

**Goal:** Working API accessible via public URL within hours, not weeks.

### Option A: Railway / Render (recommended for speed)

```
GitHub repo → Railway auto-deploy → Public HTTPS endpoint
```

- Single Dockerfile with FastAPI + uvicorn
- Environment variables for API keys (ILMU, Anthropic)
- Free tier sufficient for demo; scales if needed
- Deploy command: `railway up` or push to main branch

### Option B: Docker on any VPS

```bash
docker build -t summarizer .
docker run -p 8000:8000 --env-file .env summarizer
```

### Project Structure

```
summarizer/
├── app/
│   ├── main.py              # FastAPI app, routes, middleware
│   ├── models.py            # Pydantic request/response schemas
│   ├── processor.py         # Document extraction & cleaning
│   ├── detector.py          # Language detection (fastText)
│   ├── router.py            # Provider routing logic
│   ├── providers/
│   │   ├── base.py          # SummarizationProvider protocol
│   │   ├── anthropic.py     # Claude implementation
│   │   └── ilmu.py          # ILMU implementation (stub until API access)
│   ├── evaluation.py        # LLM-as-Judge + metrics
│   ├── prompts/
│   │   ├── summarize_bm_v1.yaml
│   │   ├── summarize_en_v1.yaml
│   │   └── evaluate_v1.yaml
│   └── observability.py     # Structured logging + metrics
├── tests/
│   ├── test_processor.py
│   ├── test_detector.py
│   ├── test_providers.py
│   ├── test_evaluation.py
│   └── fixtures/            # Sample EN/BM/mixed documents
├── eval/
│   ├── test_set/            # 50+ curated docs with reference summaries
│   └── run_eval.py          # Batch evaluation script
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 8. What Ships in MVP vs. What Comes Later

| Component | MVP (Week 1) | V2 (Post-hire) |
|-----------|-------------|----------------|
| API | Single doc endpoint, sync | Batch endpoint, async with job polling |
| Document types | PDF, TXT | + DOCX, HTML, scanned PDF (OCR) |
| Providers | Anthropic (working), ILMU (stub) | ILMU as primary once API access granted |
| Chunking | Simple section split for overflow | Hierarchical with context bridging |
| Evaluation | LLM-as-Judge per request | + chrF++, BERTScore batch nightly |
| Observability | Structured JSON logs | Grafana dashboards, alerting |
| Deployment | Railway free tier | YTL AI Cloud (internal infra) |
| Auth | API key in header | OAuth2 / internal service mesh |

## 9. Why This Design Signals Principal-Level Thinking

1. **Knowing what NOT to build.** No custom code-switch classifier, no NLI pipeline, no multi-phase map-reduce. These are engineering effort that better models will erase. The design invests in surfaces that outlast model improvements.

2. **Provider abstraction is the real deliverable.** The `SummarizationProvider` protocol means ILMU v1, v2, Nemo-30B, or any future model slots in without touching application code. This is exactly the platform infrastructure YTL AI Labs needs across all their products.

3. **Evaluation as a first-class citizen.** Most summarization demos skip evaluation entirely. This design includes per-request LLM-as-Judge scoring, structured observability, and a batch evaluation framework. You can't improve what you can't measure.

4. **Prompts as versioned config.** Separate BM/EN templates with semantic versioning means prompt improvements are tracked, A/B tested, and rolled back if quality drops. This is how you operate prompts in production, not as inline strings.

5. **Deployed and testable.** Not a notebook. Not a slide deck. A working API with a public URL that the interviewer can curl and see results.

## 10. Quick Start (for reviewers)

```bash
# Clone and run locally
git clone <repo>
cd summarizer
cp .env.example .env  # Add your API keys
pip install -e .
uvicorn app.main:app --reload

# Test it
curl -X POST http://localhost:8000/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "document": "Malaysia telah melancarkan model AI pertama...",
    "document_type": "txt",
    "config": {
      "target_language": "ms",
      "summary_type": "brief",
      "max_length": 100
    }
  }'
```
