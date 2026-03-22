# Evaluation & Testing Methodology Design

**Date:** 2026-03-20
**Goal:** Runnable evaluation script that tests the summarization API against real dataset samples and computes quality metrics — demo-able live during interview.
**Approach:** Bundle ~25 curated samples from XL-Sum and code-switching sources, run them through the API in-process, compute chrF++ and optional LLM-as-Judge scores.

---

## 1. Datasets & Sample Counts

| Dataset | Language | Category | Samples | Source |
|---|---|---|---|---|
| XL-Sum (Malay test split) | ms | monolingual | 12 | `csebuetnlp/xlsum` on HuggingFace |
| XL-Sum (English test split) | en | english | 5 | `csebuetnlp/xlsum` on HuggingFace |
| CS-Sum + manual curation | ms/en mixed | codeswitching | 8 | CS-Sum paper dataset + hand-curated Manglish samples |

**Total: ~25 samples.** Enough for meaningful averages, fast enough to run live (~2-3 min).

### Why these datasets

- **XL-Sum Malay:** Primary benchmark for BM summarization. BBC journalist-written reference summaries. Directly tests the monolingual Malay path.
- **XL-Sum English:** Same source structure as BM samples, keeps evaluation consistent. English is table stakes but needs coverage.
- **CS-Sum / manual code-switching:** Tests the most differentiated feature — code-switching detection. Manual samples fill gaps if CS-Sum isn't fully available and better represent real Malaysian mixed-language content.

### What's excluded and why

- **CNN/DailyMail:** English-only, lower priority than XL-Sum EN which matches the BM evaluation source.
- **CrossSum:** Tests cross-lingual summarization (BM article → EN summary), but the MVP's `auto` mode summarizes in the document's primary language, not across languages.
- **MyTextSum:** Only 100 samples, extractive references — less useful for evaluating an abstractive system.
- **MalayMMLU:** Not a summarization dataset. Useful for model pre-qualification but not for testing this API.
- **BERTScore:** Requires downloading XLM-RoBERTa (~1.1GB), adds friction for a demo. chrF++ covers the same need with zero model download.

---

## 2. Data Layout

```
data/evaluation/
├── xlsum_ms/           # 12 Malay articles from XL-Sum test split
│   ├── 001.json
│   └── ...
├── codeswitching/      # 8 code-switched samples
│   ├── 001.json
│   └── ...
├── english/            # 5 English articles from XL-Sum
│   ├── 001.json
│   └── ...
└── manifest.json       # Index listing all samples with metadata
```

### Sample JSON format

```json
{
  "id": "xlsum_ms_001",
  "source": "xlsum",
  "language": "ms",
  "category": "monolingual",
  "has_reference": true,
  "text": "Full article text...",
  "reference_summary": "Gold reference summary from dataset...",
  "config": {
    "target_language": "ms",
    "summary_type": "brief",
    "max_length": 150
  }
}
```

- Code-switching samples use `"target_language": "auto"` so the system detects language and code-switching itself.
- Manual code-switching samples without reference summaries set `"has_reference": false` — chrF++ is skipped, LLM-as-Judge is the only quality metric.

### manifest.json

```json
{
  "version": "1.0.0",
  "categories": {
    "monolingual": { "path": "xlsum_ms", "count": 12 },
    "codeswitching": { "path": "codeswitching", "count": 8 },
    "english": { "path": "english", "count": 5 }
  }
}
```

---

## 3. Evaluation Script

### Entry point

```bash
ANTHROPIC_API_KEY=sk-... python scripts/evaluate.py
```

### Execution flow

1. Load `manifest.json`, resolve all sample file paths
2. Start the FastAPI app in-process using `httpx.AsyncClient` with `ASGITransport` (same pattern as unit tests — no separate server needed)
3. For each sample:
   - POST to `/v1/summarize` with the sample's text and config
   - Capture generated summary, metadata (detected_language, code_switching_detected, latency_ms, tokens)
   - If `has_reference: true`: compute chrF++ between generated summary and reference summary
4. If `--evaluate` flag is passed: set `evaluate: true` in config to include LLM-as-Judge scores (doubles API cost)
5. Aggregate results per category and overall
6. Print results table to terminal
7. Save full results to `results/evaluation_<timestamp>.json`

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--evaluate` | off | Include LLM-as-Judge scores (5 dimensions). Doubles API cost. |
| `--category` | all | Filter: `monolingual`, `codeswitching`, `english`, or `all` |
| `--output` | `results/evaluation_<timestamp>.json` | Output path for full results JSON |

### Key design decision: in-process execution

The script imports `create_app()` directly and uses `httpx.AsyncClient(transport=ASGITransport(app=app))`. This means:
- Zero setup during demo — no need to start a server in another terminal
- Tests the full request/response cycle including Pydantic validation, document processing, and provider calls
- Same pattern already used in `tests/test_main.py`

---

## 4. Metrics

### Per-sample metrics (always computed)

| Metric | How | Purpose |
|---|---|---|
| **chrF++** | `sacrebleu` against reference summary | Primary quality metric. Character n-gram F-score handles BM morphology better than ROUGE. Skipped when `has_reference: false`. |
| **Compression ratio** | `len(summary) / len(source)` | Sanity check that summaries are appropriately shorter. |
| **Language match** | `detected_language == expected` | Did the system correctly identify the document language? |
| **Code-switching detection** | `code_switching_detected == expected` | For CS samples, did the system flag mixed language? |
| **Latency** | `latency_ms` from response | Performance tracking. |
| **Token usage** | `input_tokens + output_tokens` from response | Cost tracking. |

### Per-sample metrics (with `--evaluate`)

| Metric | Source | Scale |
|---|---|---|
| Faithfulness | LLM-as-Judge | 1-5 |
| Coherence | LLM-as-Judge | 1-5 |
| Coverage | LLM-as-Judge | 1-5 |
| Language quality | LLM-as-Judge | 1-5 |
| Conciseness | LLM-as-Judge | 1-5 |

### Terminal output

```
═══ Multilingual Summarization Evaluation ═══

Category: Malay Monolingual (12 samples)
  chrF++:    avg 42.3  min 31.2  max 56.8
  Compress:  avg 0.08  (target < 0.15)
  Lang match: 12/12 (100%)
  Latency:   avg 1240ms

Category: Code-Switching (8 samples)
  chrF++:    avg 38.1  min 28.4  max 49.2
  Compress:  avg 0.10
  CS detect: 7/8 (87.5%)
  Latency:   avg 1380ms

Category: English (5 samples)
  chrF++:    avg 45.6  min 38.9  max 52.1
  Compress:  avg 0.07
  Lang match: 5/5 (100%)
  Latency:   avg 980ms

═══ Overall ═══
  chrF++ avg: 41.8
  Total tokens: 12,400 in / 3,200 out
  Total time: 2m 14s
```

### JSON output

Full per-sample results array plus aggregated stats per category and overall. Enables comparison across runs (e.g., after prompt changes or model swaps).

---

## 5. Data Sourcing

### `scripts/prepare_data.py`

One-off script. Run once, commit the output, never run again.

1. Downloads XL-Sum via HuggingFace `datasets` library
2. Filters for `language == "malay"` and `language == "english"`
3. Selects samples from the test split (seeded random for reproducibility)
4. Writes individual JSON files + `manifest.json` to `data/evaluation/`

Code-switching samples are curated manually (or from CS-Sum if available) and committed directly — not generated by this script.

### Dependencies

`datasets` (HuggingFace) is only needed for `prepare_data.py`. Not a runtime dependency.

---

## 6. New Dependencies

Added to `[project.optional-dependencies] dev` in `pyproject.toml`:

```
sacrebleu>=2.4.0    # chrF++ computation (used by evaluate.py)
datasets>=2.0       # HuggingFace dataset download (used by prepare_data.py only)
```

Neither is needed at runtime. The main app has zero new dependencies.

---

## 7. New Files

```
scripts/
├── prepare_data.py      # One-off: downloads XL-Sum, writes sample JSONs
└── evaluate.py          # Main evaluation script
data/
└── evaluation/
    ├── manifest.json
    ├── xlsum_ms/001.json ... 012.json
    ├── codeswitching/001.json ... 008.json
    └── english/001.json ... 005.json
results/                 # gitignored
└── .gitkeep
```

**No changes to existing app code.** The evaluation script imports `create_app` and models but modifies nothing.

**`.gitignore` addition:** `results/` directory.

---

## 8. What's Cut vs. What's Kept

| Component | Full Research Doc | This Implementation |
|---|---|---|
| Datasets | 8 datasets across 5 categories | XL-Sum (BM+EN) + CS-Sum/manual |
| Metrics | ROUGE, chrF++, BERTScore, LaSE | chrF++ only (+ LLM-as-Judge already built) |
| Execution | 4-tier pipeline (per-request, sampled, benchmark, human) | Single script, run on-demand |
| Faithfulness checking | NLI model + SummaC | LLM-as-Judge faithfulness dimension |
| CS error taxonomy | 3 error types with detection methods | CS detection boolean + LLM-as-Judge coverage |
| Human evaluation | Bilingual panel, quarterly | Not implemented |
