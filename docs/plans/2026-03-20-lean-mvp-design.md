# Lean MVP Design — Multilingual Document Summarization Service

**Date:** 2026-03-20
**Goal:** Interview demo for YTL AI Labs — working API callable live
**Approach:** Preserve architecture from full design doc, cut scope ruthlessly

---

## 0. The Brief

> Build an application that can summarize long documents in both English and Malay.

The interview task requires covering four areas:

### 1. Preprocessing
- TXT document ingestion with Unicode NFC normalization
- Clean whitespace handling for Malaysian text (mixed Latin/Jawi scripts, inconsistent encoding)
- Forward-compatible `document_type` field — designed so PDF/DOCX/HTML extractors slot in without API changes
- Token counting to decide single-pass vs. chunked summarization

### 2. Multilingual Support & Code-Switching
- Bahasa Melayu (BM) and English as first-class languages with separate prompt templates
- Code-switching (Manglish) detection delegated to the LLM — modern multilingual models handle this natively, no custom classifier needed
- `target_language: "auto"` mode lets the model detect the primary language and summarize accordingly
- `preserve_domain_terms: true` keeps English technical terms in BM summaries (API, cloud computing, legal citations) — reflects real Malaysian document conventions

### 3. Summarization Approach
- **Abstractive summarization via LLM** (Claude via Anthropic API) — not extractive. Extractive methods fail on multilingual/code-switched content because sentence boundaries and importance signals differ across languages
- **Provider-agnostic protocol** — `SummarizationProvider` interface means ILMU (Malaysian sovereign LLM), Anthropic, or any future model slots in via config change
- **Configurable summary types:** brief, detailed, executive — each maps to different prompt instructions
- **Chunking as fallback only** — most documents fit modern context windows (100K+ tokens). Two-pass chunking exists for overflow, not as the default path

### 4. Evaluation
- **LLM-as-Judge** — optional per-request evaluation scoring faithfulness, coherence, coverage, language quality, and conciseness (1-5 scale)
- Separate evaluation prompt (`evaluate_v1.yaml`) ensures scoring criteria are versioned and auditable
- Source excerpt truncated to 2000 chars for cost efficiency
- Designed for future extension: batch evaluation with chrF++ (better than ROUGE for agglutinative Malay morphology) and BERTScore (XLM-RoBERTa for cross-lingual semantic similarity)

---

## 1. API Contract

### `POST /v1/summarize`

```json
// Request
{
  "document": "raw text string",
  "document_type": "txt",
  "config": {
    "target_language": "en" | "ms" | "auto",
    "summary_type": "brief" | "detailed" | "executive",
    "max_length": 500,
    "preserve_domain_terms": true,
    "evaluate": false
  }
}

// Response
{
  "summary": "...",
  "metadata": {
    "detected_language": "ms",
    "code_switching_detected": true,
    "model_used": "claude-sonnet-4-20250514",
    "input_tokens": 4200,
    "output_tokens": 380,
    "latency_ms": 1240,
    "evaluation": null
  }
}
```

### `GET /v1/health`

Returns service status and model availability.

**Simplifications from full design:**
- `document_type` always `"txt"` — field exists for forward compatibility
- `evaluate` defaults to `false` — opt-in per request
- No base64 encoding — raw text in body
- Language detection inside the LLM call, not a separate pipeline step

---

## 2. Project Structure

```
app/
├── main.py              # FastAPI app, routes, middleware
├── models.py            # Pydantic request/response schemas
├── processor.py         # Document extraction (TXT only for now)
├── providers/
│   ├── base.py          # SummarizationProvider protocol
│   ├── anthropic.py     # Claude implementation (working)
│   └── ilmu.py          # ILMU stub (returns mock data)
├── evaluation.py        # LLM-as-Judge (optional per request)
├── prompts/
│   ├── summarize_en_v1.yaml
│   ├── summarize_ms_v1.yaml
│   ├── summarize_auto_v1.yaml
│   └── evaluate_v1.yaml
├── prompt_loader.py     # Loads & renders YAML templates
└── config.py            # Settings via pydantic-settings + env vars
tests/
├── test_models.py
├── test_processor.py
├── test_providers.py
└── test_evaluation.py
pyproject.toml
.env.example
Dockerfile
```

---

## 3. Provider Abstraction

Core architectural investment.

```python
class SummarizationProvider(Protocol):
    async def summarize(self, text: str, config: SumConfig) -> SumResult: ...
    async def evaluate(self, source: str, summary: str, target_language: str) -> EvalResult: ...

    @property
    def name(self) -> str: ...

    @property
    def max_context_tokens(self) -> int: ...
```

**AnthropicProvider:**
- Async Anthropic SDK client
- Selects prompt template by `target_language`
- `auto` mode: prompt asks Claude to detect language, detect code-switching, and summarize — all in one structured JSON response
- Returns token counts and latency from API response

**ILMUProvider (stub):**
- Returns realistic mock data
- Interview talking point: "Swap one config line when ILMU API access is granted"

**No router** — `config.py` has a `PROVIDER` setting, `main.py` instantiates at startup.

---

## 4. Prompts (Versioned YAML)

Prompts as versioned config files in `app/prompts/`:

- `summarize_en_v1.yaml` — English summarization instructions
- `summarize_ms_v1.yaml` — BM summarization in bahasa baku, preserves English technical terms
- `summarize_auto_v1.yaml` — Detects language + code-switching, summarizes in document's primary language
- `evaluate_v1.yaml` — LLM-as-Judge scoring prompt

`prompt_loader.py` loads YAML, substitutes variables (`{max_length}`, `{summary_type}`, `{document_text}`). Supports version swapping via config for A/B testing.

All prompts request structured JSON output for clean parsing.

---

## 5. Evaluation (Optional)

When `"evaluate": true`:

1. First LLM call — summarize
2. Second LLM call — evaluate using `evaluate_v1.yaml`

**Dimensions (1-5):** faithfulness, coherence, coverage, language_quality, conciseness + one-sentence justification.

When `evaluate: false` (default): single LLM call, `evaluation: null` in response.

Source excerpt truncated to first 2000 chars for the judge call.

---

## 6. Dependencies

```
fastapi
uvicorn
anthropic
pyyaml
pydantic-settings
httpx
```

Six dependencies. No fastText, no PyMuPDF, no heavy NLP libraries.

**Environment variables:**

```env
ANTHROPIC_API_KEY=sk-...
PROVIDER=anthropic
DEFAULT_MODEL=claude-sonnet-4-20250514
LOG_LEVEL=info
```

---

## 7. What's Cut vs. What's Kept

| Component | Full Design | Lean MVP |
|---|---|---|
| Document types | PDF, DOCX, TXT, HTML | TXT only |
| Language detection | fastText + code-switch heuristic | LLM detects inline |
| Provider routing | Router with availability checks | Single provider via config |
| Prompts | Versioned YAML | Versioned YAML (kept) |
| Chunking | Section-aware two-pass | Simple token-count check |
| Evaluation | Always-on | Optional via flag |
| Observability | Structured logging + alerts | Basic request logging |
| Deployment | Railway auto-deploy | Deferred |
