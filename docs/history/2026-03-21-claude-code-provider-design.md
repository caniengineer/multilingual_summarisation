# Claude Code Local Provider Design

**Date:** 2026-03-21
**Status:** Approved

## Summary

Add a `ClaudeCodeProvider` that calls the `claude` CLI locally via `claude -p --output-format json --model <model>` instead of hitting the Anthropic API. Wire up the existing `PROVIDER` config field to select between `anthropic` and `claude-code` at startup.

## Motivation

Use a local Claude Code installation as the LLM backend for the summarisation app, avoiding direct API key management and leveraging Claude Code's built-in authentication.

## Design

### 1. ClaudeCodeProvider (`app/providers/claude_code.py`)

New class implementing the `SummarizationProvider` protocol.

**Constructor:** `__init__(self, model: str)` — no API key needed.

**Properties:**
- `name` → `f"claude-code/{model}"`
- `max_context_tokens` → `200_000`

**Methods:**
- `summarize(text, config)` — loads prompt template via `PromptLoader`, pipes it to `claude -p` via `asyncio.create_subprocess_exec`, parses JSON response
- `evaluate(source, summary, target_language)` — same pattern for evaluation

**CLI invocation:**
```
claude -p --output-format json --model <model>
```
Prompt is piped via stdin. Response is JSON with this structure:
```json
{
  "result": "<text output>",
  "is_error": false,
  "usage": {
    "input_tokens": 2,
    "output_tokens": 12,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

**Token extraction:** `usage.input_tokens` and `usage.output_tokens` from the JSON response.

**Response parsing:** The `result` field contains the LLM text output. Use the existing `_parse_llm_json()` helper to extract structured JSON from the text (same as `AnthropicProvider`).

**Error handling:**
- Non-zero exit code → raise with stderr message
- `is_error: true` in response → raise
- Malformed JSON → raise

### 2. Provider Selection (`app/config.py` + `app/main.py`)

**Config changes:**
- `PROVIDER: str = "anthropic"` — already exists, wire it up
- Add `CLAUDE_CODE_MODEL: str = "claude-sonnet-4-20250514"` — model for Claude Code provider

**main.py changes:**
Simple if/elif in `create_app()`:
- `PROVIDER=anthropic` → `AnthropicProvider(api_key, model)`
- `PROVIDER=claude-code` → `ClaudeCodeProvider(model=settings.CLAUDE_CODE_MODEL)`
- Unknown → raise `ValueError`

When using `claude-code` provider, `ANTHROPIC_API_KEY` is not required.

### 3. Testing

- Protocol conformance test for `ClaudeCodeProvider`
- Unit tests mocking `asyncio.create_subprocess_exec`:
  - Correct CLI args passed
  - Prompt piped via stdin
  - JSON response parsed into `SumResult`
  - Error cases: non-zero exit, malformed JSON, `is_error: true`
- No integration tests requiring local `claude` installation

## Files Changed

| File | Change |
|------|--------|
| `app/providers/claude_code.py` | New — `ClaudeCodeProvider` class |
| `app/config.py` | Add `CLAUDE_CODE_MODEL` field, make `ANTHROPIC_API_KEY` optional |
| `app/main.py` | Wire up `PROVIDER` setting for provider selection |
| `tests/test_providers.py` | Add Claude Code provider tests |
