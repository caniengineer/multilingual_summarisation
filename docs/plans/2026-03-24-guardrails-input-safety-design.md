# Guardrails AI Input Safety Integration Design

**Date:** 2026-03-24
**Status:** Approved

## Problem

User-submitted documents are inserted directly into LLM prompts with no adversarial input detection. A prompt injection in a PDF or text document could manipulate the summarization output.

## Decision

Integrate Guardrails AI as an input safety layer using a FastAPI dependency on `/v1/summarize`. Use the Hub's `DetectPromptInjection` validator plus a custom `MultilingualInjectionDetector` for Malay/Manglish patterns.

## Architecture

```
Request → FastAPI Dependency (run_input_guard)
              │
              ├─ Extract text (PDF decode or raw txt)
              ├─ Sample text (first 2000 + last 2000 + random middle 2000 chars)
              ├─ Run Guardrails Guard:
              │     ├─ Hub: DetectPromptInjection
              │     └─ Custom: MultilingualInjectionDetector
              │
              ├─ If injection detected:
              │     ├─ strict_mode=true  → HTTP 400 reject
              │     └─ strict_mode=false → attach risk flag, proceed (default)
              │
              └─ Continue to route handler
```

Guard is a FastAPI dependency injected into `/v1/summarize` only. `/v1/health` is unaffected.

## Components

### New files
- `app/guardrails.py` — Guard setup, text sampling, FastAPI dependency
- `app/validators/multilingual_injection.py` — Custom Malay/Manglish injection detector

### Modified files
- `app/models.py` — Add `GuardrailsResult` model to response metadata
- `app/main.py` — Inject guard dependency into `/v1/summarize`
- `app/config.py` — Add `GUARDRAILS_STRICT_MODE` and `GUARDRAILS_SAMPLE_SIZE` settings
- `pyproject.toml` — Add `guardrails-ai` dependency

## Response Schema Addition

```json
{
  "metadata": {
    "guardrails": {
      "passed": true,
      "risk_score": 0.05,
      "validators_triggered": []
    }
  }
}
```

## Custom Validator: MultilingualInjectionDetector

Regex-based detector for Malay/Manglish prompt injection patterns:
- Malay instruction injection: "abaikan arahan sebelum ini", "tukar peranan anda"
- Manglish code-switching: "please ignore semua arahan"
- Role injection: "anda sekarang adalah...", "bertindak sebagai..."
- System prompt extraction: "tunjukkan system prompt"

Registered via `@register_validator` using the Guardrails AI validator interface.

## Large Document Handling

Documents can be up to 50MB. The guard samples text rather than scanning the full document:
- First 2000 characters
- Last 2000 characters
- One random 2000-character window from the middle
- Total sample capped at ~6000 characters

Injections are most commonly placed at document boundaries where they land in the LLM context window.

## Configuration

| Setting | Type | Default | Purpose |
|---------|------|---------|---------|
| `GUARDRAILS_STRICT_MODE` | bool | `False` | `True` = hard reject, `False` = flag and proceed |
| `GUARDRAILS_SAMPLE_SIZE` | int | `2000` | Characters per sample window |

## Error Handling

| Scenario | strict_mode=False | strict_mode=True |
|----------|-------------------|------------------|
| No injection | Proceed, `passed=True` | Proceed, `passed=True` |
| Injection detected | Proceed, `passed=False` + details | HTTP 400 reject |
| Guard itself errors | Log warning, proceed, `passed=None` | Log warning, proceed, `passed=None` |

Fail-open pattern: the guard never blocks requests due to its own failures.

## Testing

Class-based test style in `tests/test_guardrails.py`:
1. Clean text passes
2. English injection detected
3. Malay injection detected
4. Manglish injection detected
5. Large document sampling correctness
6. Injection at document tail caught
7. Strict mode rejects with HTTP 400
8. Default mode proceeds with flag
9. Guard failure is fail-open
10. API integration with guardrails metadata in response
