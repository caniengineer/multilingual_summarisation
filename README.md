# Multilingual Document Summarization API

Production-grade API service for summarizing documents in English and Bahasa Melayu, with native support for code-switched (Manglish) content. Built on a provider-agnostic architecture that decouples summarization logic from any single LLM, enabling rapid model experimentation and zero-downtime provider swaps.

---

## Architecture Overview

```
                    ┌──────────────────────────────────────────────┐
                    │              FastAPI Application              │
                    │                                              │
  Request ──────►  │  Middleware Pipeline                          │
                    │  ┌─────────────┐  ┌───────────────────────┐  │
                    │  │ Request ID  │  │  Request Deadline     │  │
                    │  │ + Logging   │  │  (timeout enforcement)│  │
                    │  └─────────────┘  └───────────────────────┘  │
                    │                                              │
                    │  Resilience Layer                            │
                    │  ┌──────────────┐ ┌──────────┐ ┌─────────┐  │
                    │  │ Concurrency  │►│ Circuit  │►│ Retry + │  │
                    │  │ Limiter      │ │ Breaker  │ │ Backoff │  │
                    │  └──────────────┘ └──────────┘ └─────────┘  │
                    │                                              │
                    │  Processing Pipeline                        │
                    │  ┌──────────┐ ┌───────────┐ ┌───────────┐   │
                    │  │ Extract  │►│ Normalize │►│ Chunk +   │   │
                    │  │ (PDF/TXT)│ │ (ftfy+NFC)│ │ Map-Reduce│   │
                    │  └──────────┘ └───────────┘ └───────────┘   │
                    │                                              │
                    │  Provider Abstraction                        │
                    │  ┌────────────────┐  ┌──────────────────┐   │
                    │  │ AnthropicProv. │  │ ClaudeCodeProv.  │   │
                    │  │ (Messages API) │  │ (CLI subprocess) │   │
                    │  └────────────────┘  └──────────────────┘   │
                    │                                              │
                    │  Observability                               │
                    │  ┌────────────┐  ┌────────────────────────┐  │
                    │  │ structlog  │  │ Prometheus (RED + LLM) │  │
                    │  │ (JSON/req) │  │ /metrics endpoint      │  │
                    │  └────────────┘  └────────────────────────┘  │
                    └──────────────────────────────────────────────┘
```

## Key Features

### Provider-Agnostic LLM Abstraction

A `SummarizationProvider` protocol defines the contract — `summarize`, `reduce`, `evaluate` — with no inheritance coupling. Providers are runtime-checkable, making it trivial to add new backends (e.g., a sovereign Malaysian LLM) without touching application code.

```python
@runtime_checkable
class SummarizationProvider(Protocol):
    async def summarize(self, text: str, config: SummarizeConfig) -> SumResult: ...
    async def reduce(self, section_summaries: list[str], config: SummarizeConfig) -> SumResult: ...
    async def evaluate(self, source: str, summary: str, target_language: str) -> EvaluationScores: ...
```

### Hierarchical Map-Reduce for Long Documents

Documents exceeding the model's context window are chunked using **structural awareness** — Markdown headings, ALL-CAPS headers (from PDF extraction), numbered sections, and Malaysian government section markers (`BAHAGIAN I`, `BAHAGIAN II`). Each chunk is summarized independently with a **context bridge** (previous summary prefix) to maintain narrative continuity, then merged in a reduce pass that handles deduplication and coherence.

### Multilingual & Code-Switching Support

Purpose-built for the Malaysian language landscape:

- **Language-specific prompt templates** — separate YAML templates for English and Bahasa Melayu, each respecting register conventions (bahasa baku for formal BM)
- **Domain term preservation** — English technical terms (API, cloud computing) are preserved in BM summaries; Malay legal references and proper nouns are never force-translated
- **Code-switching detection** — metadata reports whether intra-sentential mixing was detected, enabling downstream consumers to handle mixed-language content appropriately

### Production Resilience

Three layers of protection cascaded in the request path:

| Layer | Pattern | Behavior |
|-------|---------|----------|
| **Concurrency Limiter** | Asyncio semaphore | Rejects immediately with `Retry-After` when at capacity (default: 10 concurrent LLM calls) |
| **Circuit Breaker** | Sliding-window failure detection | Trips open after N failures within a time window; auto-recovers via half-open probe |
| **Retry with Backoff** | Exponential + jitter | Retries only on 429/500/529; capped delay prevents thundering herd |

All thresholds are configurable via environment variables — no code changes needed to tune for different traffic patterns.

### Structured Observability

- **Structured logging** via structlog — JSON output to stderr with request ID correlation across the full request lifecycle (`request_started` → `document_processing_started` → `chunk_summarized` → `reduce_started` → `request_completed`)
- **Prometheus metrics** following RED methodology:
  - HTTP: `http_requests_total`, `http_request_duration_seconds`, `http_errors_total`
  - LLM: `llm_calls_total`, `llm_call_duration_seconds`, `llm_tokens_total` (input/output)
- **Request ID middleware** — generates or propagates `x-request-id` for distributed tracing

### Multi-Layered Evaluation Framework

Quality assessment combines fast automated metrics with semantic LLM judgment:

| Metric | Role | Cost |
|--------|------|------|
| **chrF++** | Regression detection gate — language-agnostic character n-gram F-score | Free |
| **LLM-as-Judge** | Semantic quality across 5 dimensions (faithfulness, coherence, coverage, language quality, conciseness) | 1 API call/sample |

The evaluation pipeline supports **tiered runs** (smoke / regression / benchmark), **baseline comparison** with configurable regression thresholds, and result aggregation split by reference quality (human vs. machine-generated).

### Document Processing Pipeline

Robust text extraction with Malaysian document edge cases handled:

1. **PDF extraction** — PyMuPDF (fitz) per-page text extraction with base64 input
2. **Header/footer stripping** — heuristic detection of repeated lines across pages + regex for page numbers
3. **Encoding repair** — ftfy fixes mojibake from Windows-1252/ISO-8859 misinterpretation (common in Malaysian government PDFs)
4. **Unicode normalization** — NFC form for consistent Malay diacritic handling
5. **Whitespace cleanup** — collapse redundant spacing while preserving paragraph structure

---

## API Reference

### Summarize a Document

```
POST /v1/summarize
```

```json
{
  "document": "Bank Negara Malaysia mengekalkan Kadar Dasar Semalaman...",
  "document_type": "txt",
  "config": {
    "target_language": "auto",
    "summary_type": "brief",
    "max_length": 500,
    "preserve_domain_terms": true,
    "evaluate": false
  }
}
```

**Response:**

```json
{
  "summary": "Bank Negara maintains the OPR at 3.00%...",
  "metadata": {
    "detected_language": "ms",
    "code_switching_detected": false,
    "model_used": "claude-sonnet-4-20250514",
    "input_tokens": 312,
    "output_tokens": 87,
    "latency_ms": 1420,
    "chunks_used": null,
    "evaluation": null
  }
}
```

For PDF documents, send base64-encoded bytes with `"document_type": "pdf"`.

Set `"evaluate": true` to include LLM-as-Judge scores (faithfulness, coherence, coverage, language_quality, conciseness — each 1-5 with justification).

### Health Checks

| Endpoint | Purpose |
|----------|---------|
| `GET /v1/health` | Provider and model info |
| `GET /v1/health/live` | Liveness probe (always `200` if process is up) |
| `GET /v1/health/ready` | Readiness probe — returns `503` when circuit breaker is open |
| `GET /metrics` | Prometheus text format metrics |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- An Anthropic API key

### Setup

```bash
# Install dependencies
make install-dev

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Start development server
make dev

# Run tests
make test
```

### Docker

```bash
make docker-build
make docker-run
```

### Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | API key for Claude |
| `PROVIDER` | `anthropic` | Provider backend (`anthropic` or `claude-code`) |
| `DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Model for API provider |
| `REQUEST_TIMEOUT` | `120` | Per-request deadline (seconds) |
| `LLM_CALL_TIMEOUT` | `60` | Per-LLM-call timeout (seconds) |
| `RETRY_MAX_ATTEMPTS` | `3` | Retry attempts for transient failures |
| `CB_FAILURE_THRESHOLD` | `5` | Failures before circuit breaker trips |
| `CB_RECOVERY_TIMEOUT` | `30` | Seconds before circuit breaker probes recovery |
| `MAX_CONCURRENT_LLM_CALLS` | `10` | Concurrency limiter ceiling |

---

## Running Evaluations

```bash
# Prepare evaluation datasets (downloads from HuggingFace)
uv run python scripts/prepare_data.py

# Run smoke tier (fast validation)
uv run python scripts/evaluate.py --tier smoke

# Run with LLM-as-Judge scoring
uv run python scripts/evaluate.py --tier regression --evaluate

# Save baseline for regression detection
uv run python scripts/evaluate.py --tier regression --save-baseline

# Full benchmark
uv run python scripts/evaluate.py --tier benchmark --evaluate
```

Evaluation data sources:
- **Malay:** mesolitica/mixtral-malaysian-abstractive-summarization, huseinzol05/malay-dataset
- **English:** csebuetnlp/xlsum

---

## Project Structure

```
app/
├── main.py              # FastAPI app factory, route definitions
├── config.py            # Pydantic settings (env-driven)
├── models.py            # Request/response schemas
├── processor.py         # PDF extraction, normalization, header/footer stripping
├── chunker.py           # Structural document chunking (heading-aware)
├── summarizer.py        # Map-reduce orchestration with context bridging
├── evaluation.py        # LLM-as-Judge wrapper
├── prompt_loader.py     # Versioned YAML prompt template engine
├── resilience.py        # Circuit breaker, retry, concurrency limiter
├── middleware.py         # Request ID propagation, deadline enforcement
├── logging_config.py    # structlog JSON pipeline with request ID injection
├── metrics.py           # Prometheus RED + LLM metric definitions
├── providers/
│   ├── base.py          # SummarizationProvider Protocol + SumResult
│   ├── anthropic.py     # Claude Messages API implementation
│   └── claude_code.py   # Claude CLI subprocess implementation
└── prompts/
    ├── summarize_en_v1.yaml
    ├── summarize_ms_v1.yaml
    ├── reduce_v1.yaml
    └── evaluate_v1.yaml

tests/                   # pytest + pytest-asyncio test suite
scripts/
├── evaluate.py          # Tiered evaluation pipeline (chrF++ + LLM-as-Judge)
└── prepare_data.py      # Dataset download and preparation
```

---

## Development

```bash
make lint          # Ruff linter
make format        # Ruff formatter
make fix           # Auto-fix lint issues
make check         # Lint + test in sequence
make test-v        # Verbose test output
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Framework** | FastAPI | Async-native, Pydantic integration, OpenAPI docs |
| **LLM SDK** | anthropic (AsyncAnthropic) | Async client with native token usage reporting |
| **PDF** | PyMuPDF (fitz) | Fast text extraction without Java/Poppler dependency |
| **Encoding** | ftfy | Fixes mojibake common in Malaysian government PDFs |
| **Logging** | structlog | JSON structured logging with context propagation |
| **Metrics** | prometheus-client | Standard Prometheus exposition format |
| **Eval** | sacrebleu (chrF++) | Language-agnostic metric — no tokenizer alignment needed for BM |
| **Package mgmt** | uv | Fast dependency resolution and lockfile support |
| **Linting** | Ruff | Single tool for linting + formatting |
