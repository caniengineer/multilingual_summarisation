# Multilingual Document Summarization System

*System Design & Architecture for English-Malay Summarization*

---

## Executive Summary

This document presents a **production-grade system design** for a multilingual document summarization service supporting English and Bahasa Melayu. I approach it as a **platform capability** — exposed as an internal API, consumed by multiple product teams (chatbots, enterprise search, compliance tools), and evolved independently. The design accounts for real-world Malaysian language patterns including code-switching (Manglish), dialectal variations, and domain-specific terminology across government, legal, and enterprise documents.

The architecture follows a **provider-agnostic abstraction pattern** — decoupling summarization logic from any single LLM provider while enabling rapid experimentation and model swapping.

### High-Level API Contract

Designing the interface first (not the model pipeline) ensures we think about downstream consumers, versioning, and backward compatibility from the start.

```
POST /v1/summarize
  └─ Request:  { document, config: { target_language, summary_type, max_length } }
  └─ Response: { summary, metadata: { detected_language, confidence, token_usage } }

POST /v1/summarize/batch
  └─ Accepts array of documents, returns job_id for async polling

GET /v1/summarize/jobs/{job_id}
  └─ Async status + results for batch operations
```

---

## 1. Preprocessing Pipeline

Preprocessing is where most production systems silently fail. Raw documents arrive in unpredictable formats, and the quality of summarization is fundamentally bounded by the quality of text extraction. I would build this as a **pluggable pipeline** with discrete, testable stages.

### 1.1 Document Ingestion & Text Extraction

| Source Format | Extraction Strategy | Edge Cases |
|---|---|---|
| **PDF** | PyMuPDF (fitz) for text-based; Tesseract OCR + pytesseract for scanned docs with Malay language pack | Mixed scanned/digital pages; Jawi script in government docs; multi-column layouts |
| **DOCX/DOC** | python-docx for .docx; LibreOffice headless conversion for legacy .doc | Tracked changes, embedded objects, comments containing context |
| **HTML/Web** | BeautifulSoup + trafilatura for content extraction; boilerplate removal | Navigation cruft, cookie banners, bilingual sites with mixed content |
| **Plain Text** | Encoding detection (chardet/charset-normalizer); Unicode normalization (NFC) | Mixed encodings in legacy Malaysian government systems |

### 1.2 Text Normalization & Cleaning

This stage is critical for Malaysian content, where text quality varies significantly across sources:

1. **Unicode normalization:** NFC form to handle Malay diacritics consistently (e.g., standard vs decomposed forms of characters).
2. **Encoding repair:** Malaysian government PDFs frequently have mojibake from Windows-1252/ISO-8859 misinterpretation. Detect and repair using `ftfy`.
3. **Whitespace & structure:** Normalize line breaks, remove headers/footers from paginated PDFs, reconstruct paragraph boundaries from heuristic line-height analysis.
4. **Metadata extraction:** Preserve document structure (headings, sections, tables) as semantic markers. A legal document's section structure is signal, not noise.

### 1.3 Chunking Strategy for Long Documents

For documents exceeding model context windows, I would implement a **hierarchical chunking strategy** rather than naive fixed-size splitting:

- **Semantic chunking:** Split on section boundaries, headings, and paragraph clusters using document structure. Preserve semantic coherence within chunks.
- **Overlap with context bridging:** Include a summary of the previous chunk as a prefix to maintain narrative continuity. This is more robust than naive token overlap.
- **Adaptive chunk sizing:** Respect the model's effective context window (not just max tokens). Reserve tokens for the system prompt, summarization instructions, and output buffer.

I would initially build the chunking logic as a **shared module within the monorepo** — chunking is a cross-cutting concern that RAG pipelines and search indexing will also need. Once a second consumer adopts it, promote to a standalone library.

---

## 2. Multilingual Support & Code-Switching

Malaysian documents rarely exist in "pure" Bahasa Melayu or English. The reality is a **spectrum of language mixing** that any production system must handle gracefully.

### 2.1 Language Detection Architecture

I would implement a multi-layered detection approach rather than relying on a single classifier:

1. **Document-level detection:** Use fastText's language identification (`lid.176.bin`) as first pass to determine the primary language. This handles the common case efficiently.
2. **Paragraph-level detection:** For documents flagged as mixed, run per-paragraph classification. This catches the common pattern of English-primary documents with Malay legal citations, or Malay government documents with English technical terms.
3. **Code-switch detection:** For paragraphs that score ambiguously between `en` and `ms`, apply a custom classifier trained on Malaysian code-switching corpora (Manglish detection). This is the layer where domain knowledge matters most.

**Failure handling:** When language detection confidence is below a configurable threshold (e.g., <0.6), the system falls back to treating the document as code-switched and routes to a model that handles mixed-language input natively (ILMU). The detected language and confidence score are always returned in the response metadata so consumers can decide how to act.

### 2.2 Code-Switching Patterns in Malaysian Text

| Pattern | Example | Handling Strategy |
|---|---|---|
| **Intra-sentential** | "Meeting tu postpone ke next week" | Treat as single-language unit; do not split. Summarize in target output language. |
| **Inter-sentential** | BM paragraph followed by English paragraph | Detect per-paragraph; summarize each in its detected language, then unify. |
| **Technical loan words** | "AI", "cloud computing", "API" in BM text | Preserve as-is; these are not code-switches but accepted terminology. |
| **Formal BM with English insertions** | Government docs with English legal terms | Maintain English terms in BM summaries; do not force-translate domain terms. |

### 2.3 Multilingual Summarization Strategy

The summarization approach differs based on the language composition:

- **Monolingual documents (>95% single language):** Summarize directly in the source language. Optionally translate the summary via a second LLM call if a different target language is requested.
- **Code-switched documents:** Summarize in the user's requested output language. The LLM prompt explicitly instructs the model to handle mixed-language input and produce coherent output in the target language. Key domain terms are preserved regardless of output language.
- **Summarize-first over translate-first:** I default to summarize-first because translating a full document before summarization (a) introduces compounding errors — mistranslations propagate into the summary, (b) increases latency and cost by translating content that will be discarded, and (c) can strip cultural and domain nuances that a bilingual model would preserve. The tradeoff: summarize-then-translate risks the model hallucinating during the translation of the shorter summary. I mitigate this by using the same bilingual model (ILMU) for both steps and running a lightweight NLI check between the source-language summary and the translated version to catch semantic drift.

> **Why ILMU matters here:** A sovereign Malaysian LLM like ILMU that natively understands Bahasa Melayu, Manglish, and dialectal patterns would be the ideal primary model for this pipeline. It eliminates the need for explicit code-switching detection in many cases because the model inherently understands Malaysian language patterns. I would design the system to use ILMU as the primary model with fallback to other providers for edge cases.

---

## 3. Summarization Approach

### 3.1 Why Abstractive over Extractive

For this use case, I recommend a primarily abstractive approach with extractive elements as a fallback and quality signal:

| Dimension | Extractive | Abstractive (LLM) |
|---|---|---|
| **Fluency** | Choppy; sentences out of context | Natural, coherent prose |
| **Code-switching** | Extracts mixed sentences as-is | Can normalize output language |
| **Faithfulness** | High (copies source) | Risk of hallucination; needs guardrails |
| **Long documents** | Scales with sentence scoring | Requires chunking strategy |
| **Latency** | Fast (no LLM inference) | Slower; depends on model/infra |
| **BM quality** | Only as good as source sentences | ILMU excels at native BM generation |

### 3.2 Architecture: Hierarchical Map-Reduce Summarization

For long documents, I would implement a hierarchical summarization approach:

1. **Map phase:** Each chunk is summarized independently. The prompt includes document-level metadata (title, type, detected language) to maintain context. Each chunk produces a section summary with key entities and claims extracted.
2. **Reduce phase:** Section summaries are aggregated and fed into a final summarization call. This call receives the full set of section summaries plus a "summary of summaries" instruction that emphasizes coherence, deduplication, and narrative flow.
3. **Verification pass:** For high-stakes use cases (legal, compliance), compare the final summary against source chunks using NLI (Natural Language Inference) to flag potential hallucinations.

### 3.3 End-to-End Pipeline (Pseudocode)

```python
async def summarize_document(document: bytes, config: SumConfig) -> SumResult:
    # Stage 1: Extract text based on document format
    text, doc_metadata = await extract_text(document)
    if doc_metadata.extraction_confidence < QUALITY_THRESHOLD:
        raise LowQualityInputError(doc_metadata)  # Don't summarize garbage

    # Stage 2: Normalize and clean
    cleaned = normalize_text(text)  # NFC, ftfy encoding repair, whitespace

    # Stage 3: Detect language composition
    lang_result = detect_language(cleaned)  # doc-level → paragraph-level → code-switch

    # Stage 4: Select model based on language and routing rules
    provider = select_provider(lang_result, config)  # ILMU for BM, fallback otherwise

    # Stage 5: Chunk if needed, then map-reduce summarize
    if exceeds_context_window(cleaned, provider):
        chunks = semantic_chunk(cleaned, doc_metadata.structure)
        section_summaries = await asyncio.gather(*[
            provider.summarize(chunk, config, context=doc_metadata)
            for chunk in chunks
        ])
        summary = await provider.reduce(section_summaries, config)
    else:
        summary = await provider.summarize(cleaned, config)

    # Stage 6: Optional cross-language translation
    if config.target_language != lang_result.primary_language:
        summary = await provider.translate(summary, config.target_language)
        drift_score = await check_semantic_drift(summary, cleaned)
        if drift_score < DRIFT_THRESHOLD:
            summary.warnings.append("Potential semantic drift in translation")

    return SumResult(summary=summary, metadata=build_metadata(lang_result, provider))
```

### 3.4 Model Selection Strategy

I would build a **provider-agnostic model abstraction** that allows swapping models without changing application code:

```python
class SummarizationProvider(Protocol):
    async def summarize(self, text: str, config: SumConfig) -> SumResult: ...
    async def reduce(self, summaries: list[str], config: SumConfig) -> SumResult: ...
    async def translate(self, text: str, target_lang: str) -> SumResult: ...

class ILMUProvider(SummarizationProvider): ...      # Primary for BM content
class AnthropicProvider(SummarizationProvider): ...  # Fallback / English-heavy
class OpenAIProvider(SummarizationProvider): ...     # Alternative fallback
```

**Model routing logic:** Route to the optimal model based on detected language, document length, and latency requirements. ILMU would be the primary model for any content with significant Bahasa Melayu presence, with other providers as fallbacks. This routing is configurable via feature flags, not hardcoded.

### 3.5 Prompt Engineering & Template Management

Prompts are treated as **versioned configuration**, not inline strings:

- **Template registry:** Prompts stored in a versioned config store. Each template has a semantic version, A/B test variant ID, and associated evaluation metrics.
- **Language-specific templates:** Separate prompt templates for BM and English, because effective summarization instructions differ between languages (e.g., BM formal register conventions, honorific preservation).
- **Domain-specific templates:** Legal documents, government reports, and business documents each get specialized prompts that understand domain conventions (e.g., preserving section references in legal summaries).

---

## 4. Evaluation Framework

Evaluation is where most summarization systems are weakest. I would implement a **multi-layered evaluation framework** that combines automated metrics with human evaluation, operating at different cadences.

### 4.1 Automated Metrics (Run on Every Request)

| Metric | What It Measures | Notes for BM/Multilingual |
|---|---|---|
| **ROUGE-L** | Surface-level overlap with reference or source | Less reliable for BM due to agglutinative morphology; use as directional signal only |
| **chrF++** | Character n-gram F-score | Better suited for BM than word-level ROUGE; captures morphological similarity without requiring tokenizer alignment |
| **BERTScore** | Semantic similarity using embeddings | Use multilingual models (XLM-RoBERTa) for cross-lingual scoring |
| **LLM-as-Judge** | Coherence, faithfulness, and relevance scored by a second LLM | Most reliable for BM; use structured rubric with 1–5 scoring per dimension |
| **NLI Faithfulness** | Entailment check: does source entail each summary claim? | Critical for hallucination detection; run per-sentence against source chunks |
| **Compression ratio** | Summary length vs source length | Sanity check; flag summaries that are suspiciously short or long |

### 4.2 LLM-as-Judge: Structured Evaluation

For production evaluation, I would use a structured rubric evaluated by a separate LLM call, scoring five dimensions: **Faithfulness** (claims grounded in source), **Coverage** (key points captured), **Coherence** (logical flow), **Language quality** (grammar, register, code-switch handling for BM), and **Conciseness** (compression without information loss). Each dimension is scored 1–5 with a one-sentence justification, returned as structured JSON. The evaluation prompt is versioned alongside the summarization prompts.

### 4.3 Human Evaluation (Periodic / Pre-Release)

Automated metrics are necessary but insufficient. I would establish:

- **Bilingual evaluation panel:** Native BM speakers who also read English, evaluating on the same rubric dimensions. Critical for catching cultural/linguistic nuances that automated metrics miss.
- **Adversarial test set:** Curated documents designed to stress-test edge cases — heavy code-switching, Jawi text, dialectal content (Kelantanese, Sarawak BM), legal documents with English citations in BM text.
- **A/B evaluation pipeline:** When testing new models or prompt versions, run both variants on the evaluation set and compare scores with statistical significance testing.

### 4.4 Production Observability

- **Latency tracking:** P50/P95/P99 latency per document length bucket and language. Alert on latency regression.
- **Quality drift detection:** Run automated evaluation on a sample of production requests daily. Alert if average faithfulness or coherence scores drop below threshold.
- **Error rate by language:** Monitor failure rates separately for EN, BM, and code-switched documents to catch language-specific regressions.
- **Token usage monitoring:** Track input/output tokens per request for cost management and anomaly detection.

---

*This design reflects how I approach platform engineering: start with the API contract and consumer needs, build modular components that other teams can reuse, invest in observability from day one, and design for the failure modes that production traffic will inevitably surface. The multilingual dimension is not an afterthought but a first-class design constraint — which is exactly what building for Malaysian users demands.*