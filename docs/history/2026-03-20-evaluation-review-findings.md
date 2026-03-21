# Evaluation Plan Review Findings

**Date:** 2026-03-20
**Status:** Pending — apply fixes before implementing evaluation tasks

---

## Design Review — Ready to implement (2 items to fix)

### What's good

- API contract alignment is correct — request/response shapes match the MVP
- `create_app()` + ASGITransport pattern mirrors existing tests
- chrF++ over ROUGE for BM is well-justified
- Dataset exclusion rationale is solid (especially CrossSum — MVP doesn't do cross-lingual)
- Clean separation from app code — script imports but modifies nothing
- `evaluate: true` integration correctly leverages existing LLM-as-Judge
- Data preparation as one-off committed artifact — no HuggingFace access needed at demo time

### Issues

| # | Severity | Issue | Fix |
|---|---|---|---|
| C1 | High | Sample JSON uses `text` field but API expects `document` — mapping not documented in design | Already handled in script code (maps `sample["text"]` to `"document"`), but make explicit in design |
| C4 | Medium | No per-sample error handling — one failure kills the demo | Add try/except per sample, skip and report |
| C3 | Medium | 5 English / 8 CS samples is statistically thin — design claims "meaningful averages" | Add std deviation to output, be honest about limitations |
| C7 | Low | If CS-Sum unavailable, all 8 CS samples are manual — 5 have references, 3 don't | Already addressed in implementation plan |
| C8 | Low | `datasets` (heavy) in `dev` group bloats install for non-eval devs | Consider separate `eval` optional-dependencies group |

### Suggestions

- Add `--dry-run` flag for offline testing without API calls
- Report `avg +/- std` not just `avg` for statistical transparency
- Add a `--provider ilmu` flag to inject stub provider without env vars
- Consider parallel API calls via `asyncio.gather` with semaphore for faster demo runs

---

## Implementation Review — Needs fixes (3 critical before implementation)

### Bugs (will crash at runtime)

| # | Severity | Issue | Fix |
|---|---|---|---|
| BUG-3 | Critical | `app/main.py` has module-level `app = create_app()` which fires on import, crashes without env vars set | Guard the call or remove it entirely and use `uvicorn app.main:create_app --factory` |
| BUG-1 | Critical | `create_app()` without args calls `Settings()` which requires `ANTHROPIC_API_KEY` as mandatory field | Script should construct provider directly and pass to `create_app(provider=...)`, or set env vars before import |
| BUG-4 | Minor | `results/` in `.gitignore` means `git add results/.gitkeep` silently does nothing | Use `git add -f results/.gitkeep` |
| BUG-5 | Minor | `language_match` is always `True` for code-switching samples — metric is meaningless for that category | Acknowledged, acceptable for demo |

### Gaps

| # | Issue | Fix |
|---|---|---|
| GAP-1 | No prerequisite check that MVP app exists before running | Add try/except on `from app.main import create_app` with clear error message |
| GAP-2 | `sys.path.insert` is fragile if app is also pip-installed | Document that script should be run from project root |
| GAP-3 | No error handling for malformed manifest.json or sample files | Add try/except around JSON loading |
| GAP-4 | No validation of sample data structure (missing fields cause KeyError) | Validate required fields on load |
| GAP-5 | XL-Sum config name "malay" not verified — likely correct but should confirm | Verify against HuggingFace dataset card |

### Warnings

| # | Issue | Fix |
|---|---|---|
| WARN-1 | sacrebleu verification only checks import, not `sentence_chrf` function | Change verification to: `python -c "import sacrebleu; print(sacrebleu.sentence_chrf('test', ['test']).score)"` |
| WARN-2 | ILMU stub hardcodes `code_switching_detected=False` — smoke test shows 0/8 CS detection | Document expected stub output in Task 5 so it doesn't look like a failure |
| WARN-3 | No timeout/retry for individual API calls | 60s timeout exists but no retry logic — acceptable for demo |
| WARN-6 | `datasets` library pulls in pyarrow, fsspec — heavy install | Acceptable for dev-only dependency |

### Style

- Error results have different structure than success results — consider a result dataclass
- Compression ratio returns 0.0 for empty source (semantically odd) — consider returning None
- Latency uses integer division (`//`) instead of `int(sum/len)` — minor truncation difference

---

## Required Fixes Before Implementation (Priority Order)

1. **Fix `app/main.py` module-level `create_app()` call** — affects both MVP and evaluation. Either guard with `if __name__ == "__main__"` or remove and use `uvicorn app.main:create_app --factory`. This is the most critical fix because the evaluation script's import triggers it.

2. **Have evaluation script pass provider explicitly** — add a `--provider` flag that constructs the provider directly (`ILMUProvider()` or `AnthropicProvider(...)`) and passes to `create_app(provider=...)`. Avoids Settings() dependency entirely for stub testing.

3. **Add per-sample error handling** — wrap each API call in try/except, record error, continue to next sample. Report error count in terminal summary. A demo that dies mid-run on one flaky LLM response is a bad look.

4. **Fix `git add -f results/.gitkeep`** — minor but will cause a silent commit failure.

5. **Add std deviation to output** — `avg +/- std` is more honest than bare averages with 5-12 samples per category.

---

## MVP Implementation Code Review

**Date:** 2026-03-20
**Scope:** Full MVP implementation (commits `e310533..a18e650`, 10 commits)
**Result:** 38/38 tests passing, lint clean, **ready for demo use**

### What Was Done Well

- **Faithful plan execution.** 10 commits map cleanly to planned tasks. TDD followed consistently.
- **Clean provider abstraction.** `SummarizationProvider` protocol is `@runtime_checkable`, properly typed, keeps Anthropic decoupled.
- **Testability.** `create_app(provider=None)` factory allows dependency injection of mock providers in tests.
- **Good test coverage.** 38 tests covering config, models, processor edge cases (Unicode NFC, empty input), prompt loading, provider mocking, evaluation, and API endpoints.
- **Dockerfile improvement.** Uses `--factory` flag for uvicorn, avoiding module-level `app = create_app()` that would require env vars at import time.

### Plan Deviations

| Deviation | Status |
|-----------|--------|
| ILMU stub provider dropped (user decision) | Acceptable |
| Module-level `app = create_app()` removed | Positive — prevents import-time crashes |
| Protocol tests use `FakeProvider` instead of ILMU | Acceptable — better protocol test |

### Important Issues (Should Fix)

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 1 | No error handling for malformed LLM JSON responses — `json.loads()` raises raw 500 | `app/providers/anthropic.py:39,64` | Wrap in try/except, strip markdown fences, return clear error |
| 2 | No request size limit on `document` field — oversized input fully processed before API rejects | `app/models.py` | Add `Field(max_length=500_000)` |
| 3 | Double truncation in evaluation path — both `evaluation.py` and `anthropic.py` truncate to 2000 chars | `app/evaluation.py:13`, `app/providers/anthropic.py:53` | Remove `[:2000]` from provider, keep in `evaluation.py` |

### Suggestions (Nice to Have)

| # | Issue | Fix |
|---|-------|-----|
| 4 | `EvaluationScores` has no range validation — LLM could return 0 or 10 | Add `Field(ge=1.0, le=5.0)` to score fields |
| 5 | `latency_ms` excludes evaluation time — computed before eval call | Move after evaluation block, or report separate timings |
| 6 | Prompt template rendering uses naive `str.replace` — could collide with `{{`/`}}` | Fine for MVP, note for future |
| 7 | `PROVIDER` config field is vestigial — no effect since ILMU dropped | Remove or re-add branching logic |
| 8 | Missing `conftest.py` — no shared fixtures | Extract `mock_provider` if test suite grows |

### Security Notes

- API keys loaded from env vars, never hardcoded
- `.gitignore` correctly excludes `.env` while allowing `.env.example`
- No secrets committed
- Dockerfile runs as root (fine for demo, add non-root user for production)

### Summary

| Category | Count |
|----------|-------|
| Critical issues | 0 |
| Important issues | 3 |
| Suggestions | 5 |
| Plan deviations | 3 (all acceptable) |

**Most impactful fix for demo reliability:** JSON error handling in Anthropic provider (Issue #1).
