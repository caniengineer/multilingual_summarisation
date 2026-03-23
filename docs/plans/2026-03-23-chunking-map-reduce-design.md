# Hierarchical Chunking & Map-Reduce Summarization Design

*2026-03-23*

---

## Overview

Long documents that exceed the LLM provider's context window currently fail silently or error out. This design adds a chunking module and summarization orchestrator that splits long documents on structural boundaries and uses a map-reduce flow to produce a coherent unified summary.

**Approach:** Chunker module + separate orchestrator (Approach B).

---

## 1. Chunker Module (`app/chunker.py`)

**Responsibility:** Take normalized text + a token budget → return a list of chunks that respect structural boundaries and fit within the budget.

### Structure Detection Heuristics (priority order)

1. **Markdown-style headings** — `#`, `##`, `###` lines
2. **Numbered section headers** — `1.0`, `1.1`, `BAHAGIAN II` (Malaysian government docs)
3. **Double-newline paragraph breaks** — universal fallback

### Chunking Algorithm

1. Split text into **sections** using heading detection
2. If a section fits within the token budget → it's one chunk
3. If a section is too large → sub-split on paragraph breaks (double-newline)
4. If a single paragraph is still too large → split on sentence boundaries (last resort)
5. **Greedy merge:** After splitting, merge adjacent small sections into one chunk to avoid tiny chunks

### Chunk Sizing

- Token budget = `provider.max_context_tokens` minus reserved tokens (system prompt ~500, instructions ~300, output buffer ~4096)
- Uses existing `_estimate_tokens()` for now

### Context Bridging

- Each chunk after the first gets a prefix: `"[Context: {previous chunk's one-line summary}]"`
- The summary comes from the map phase — sequential processing, not parallel
- First chunk has no bridge prefix

---

## 2. Summarization Orchestrator (`app/summarizer.py`)

**Responsibility:** Own the "chunk or not?" decision and the map-reduce flow.

### Flow

```
summarize_document(text, config, provider)
    │
    ├─ text fits in context window?
    │   └─ YES → provider.summarize(text, config) → return result
    │
    └─ NO → map-reduce:
         ├─ MAP: for each chunk sequentially:
         │        provider.summarize(chunk, config)
         │        → collect section summary
         │        → attach as context bridge to next chunk
         │
         └─ REDUCE: concatenate section summaries →
                     provider.reduce(combined, config)
                     → return final unified summary
```

### Key Decisions

- **Sequential map, not parallel** — context bridging needs the previous chunk's summary before processing the next. Trades latency for coherence.
- **Single reduce step** — one pass over concatenated section summaries. If they somehow exceed context (unlikely with 200k), do recursive reduce.
- **Reduce uses a different prompt** — "synthesize these section summaries into a coherent whole", not the same as chunk summarization.

---

## 3. Provider Protocol Changes

Add one method to `SummarizationProvider` in `app/providers/base.py`:

```python
async def reduce(self, section_summaries: list[str], config: SummarizeConfig) -> SumResult: ...
```

New prompt template: `app/prompts/reduce_v1.yaml` — instructs the model to merge section summaries, deduplicate overlapping points, and produce a coherent unified summary.

Both providers (`AnthropicProvider`, `ClaudeCodeProvider`) get a `reduce()` implementation.

---

## 4. Integration with `main.py`

Minimal change:

- **Current:** `main.py → processor.process() → provider.summarize() → response`
- **New:** `main.py → processor.process() → summarizer.summarize_document() → response`

`main.py` doesn't know about chunks. One call changes.

---

## 5. Response Metadata

One new optional field on `SummaryMetadata` in `app/models.py`:

- `chunks_used: Optional[int] = None` — how many chunks the document was split into (`None` = no chunking needed)
- `input_tokens` / `output_tokens` — aggregated across all map + reduce calls

No breaking API changes.

---

## 6. Out of Scope

- fastText language detection (separate feature)
- Per-paragraph language classification
- Parallel chunk processing (future optimization)
- OCR for scanned PDFs
