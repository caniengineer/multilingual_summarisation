# PDF Preprocessing Use Cases

Production use cases for extending the `/v1/summarize` API to accept PDF documents. Each use case describes a real Malaysian government PDF, how a consumer would call the API, and what challenges the preprocessing pipeline must handle.

All fixture PDFs are in `tests/fixtures/pdf/`.

---

## Use Case 1: BNM Annual Report — English Financial Report

**Fixture:** `bnm_annual_report_2021_en.pdf` (8.8 MB)
**Source:** Bank Negara Malaysia Annual Report 2021

### Document Characteristics
- **Language:** English-primary with Malaysian financial/policy terminology
- **Pages:** ~190 pages (large document — will need chunking)
- **Layout:** Mix of single-column prose, two-column sections (e.g., Statutory Requirements page), tables, charts, and infographics
- **Structure:** Clear section hierarchy — Foreword, Our Role (Monetary Stability, Financial Stability, etc.), Managing the Bank, Governance, Our Finances
- **Content types:** Narrative prose, financial statements, statistical tables, feature articles
- **Edge cases:** Two-column layout on some pages; embedded charts/figures that should be skipped during text extraction; cover pages and decorative pages with minimal text

### API Request
```json
POST /v1/summarize
{
  "document": "<base64-encoded PDF bytes>",
  "document_type": "pdf",
  "config": {
    "target_language": "en",
    "summary_type": "executive",
    "max_length": 1000
  }
}
```

### Expected Pipeline Behavior
1. **Extraction:** PyMuPDF extracts text from all pages. Two-column pages must be handled (text flows left-to-right per column, not across columns). Cover/decorative pages produce minimal text — should not pollute the summary.
2. **Normalization:** Standard whitespace cleanup. Headers/footers (page numbers, "Annual Report 2021") should be stripped.
3. **Chunking:** Document exceeds any reasonable context window. Hierarchical chunking by section structure (Contents page provides the map).
4. **Language detection:** Should detect as English (>95%). Malaysian terms like "Bank Negara Malaysia", "ringgit", "Overnight Policy Rate (OPR)" are loan words, not code-switching.
5. **Summary output:** English executive summary covering key themes — monetary policy decisions, GDP growth, financial stability measures.

### What This Tests
- Large PDF text extraction
- Two-column layout handling
- Long document chunking strategy
- Header/footer removal
- Skipping image-only/decorative pages

---

## Use Case 2: Budget Speech 2026 (BM) — Bahasa Melayu with Natural Code-Switching

**Fixture:** `budget_speech_2026_bm.pdf` (33 MB, 362 pages)
**Source:** Ucapan Belanjawan MADANI Keempat 2026, Ministry of Finance Malaysia

### Document Characteristics
- **Language:** Bahasa Melayu primary with natural English code-switching throughout
- **Pages:** 362 pages (very large — heavy graphics)
- **Layout:** Graphical cover pages, full-page images/backgrounds, text overlaid on decorative backgrounds, table of contents in BM
- **Structure:** Organized by "TEKAD" (pledges) — TEKAD SATU through TEKAD EMPAT, each with numbered sub-sections
- **Code-switching patterns:**
  - English financial terms embedded in BM text: "GDP", "fiscal deficit", "subsidy rationalization"
  - Policy brand names kept in English: "MADANI", though this is an acronym
  - Section headings use BM: "PELABURAN SEKTOR STRATEGIK", "PEMBANGUNAN BAKAT TEMPATAN"
- **Edge cases:** Many pages are primarily graphical (infographics, photos with text overlays). The 33MB size is mostly images. Actual extractable text is much smaller than the page count suggests. Arabic Quranic text may appear (as seen in the EN translation).

### API Request
```json
POST /v1/summarize
{
  "document": "<base64-encoded PDF bytes>",
  "document_type": "pdf",
  "config": {
    "target_language": "ms",
    "summary_type": "executive",
    "max_length": 800
  }
}
```

### Expected Pipeline Behavior
1. **Extraction:** PyMuPDF extracts text. Many pages will yield little/no text (graphical pages). The pipeline should not treat low-text pages as errors.
2. **Normalization:** Unicode NFC normalization for Malay diacritics. Whitespace cleanup. Copyright/publisher info pages should be excluded from summarization content.
3. **Language detection:** Should detect as Bahasa Melayu with code-switching flag set to `true`. English financial terms within BM sentences are intra-sentential code-switching.
4. **Summary output:** BM summary preserving English technical/financial terms as-is (do not translate "GDP" to "KDNK" unless the source uses "KDNK"). Should capture the key budget allocations and policy pledges.

### What This Tests
- Handling graphically-heavy PDFs where most pages are images
- Large file size (33MB) — must handle without memory issues
- BM text extraction and normalization
- Code-switching detection (EN terms in BM text)
- Summarizing in BM output language

---

## Use Case 3: Budget Speech 2026 (EN) — English Translation of BM Speech

**Fixture:** `budget_speech_2026_en.pdf` (876 KB)
**Source:** English translation of the Fourth MADANI Budget 2026 speech

### Document Characteristics
- **Language:** English with Malay greetings and Islamic salutations preserved
- **Pages:** Moderate length (compact compared to BM version — no heavy graphics)
- **Layout:** Clean single-column text with numbered paragraphs (1, 2, 3...). Bold section headings. Page numbers at bottom.
- **Structure:** Preamble → numbered paragraphs covering budget themes. Includes Quranic verses in Arabic script with English translation.
- **Code-switching patterns:**
  - Malay greetings preserved: "Bismillahirrahmanirrahim", "Assalamualaikum Warahmatullahi Wabarakatuh", "Salam Sejahtera"
  - Malaysian policy terms: "MADANI", "Dewan Rakyat", "Tan Sri", "Thirteenth Malaysia Plan"
  - Arabic Quranic text with English translation
- **Edge cases:** Arabic script (Quran verses) embedded in otherwise English text. Honorifics ("YAB Dato' Seri") should be preserved, not normalized away.

### API Request — Summarize in English
```json
POST /v1/summarize
{
  "document": "<base64-encoded PDF bytes>",
  "document_type": "pdf",
  "config": {
    "target_language": "en",
    "summary_type": "brief",
    "max_length": 500
  }
}
```

### API Request — Cross-language: English PDF → BM Summary
```json
POST /v1/summarize
{
  "document": "<base64-encoded PDF bytes>",
  "document_type": "pdf",
  "config": {
    "target_language": "ms",
    "summary_type": "brief",
    "max_length": 500
  }
}
```

### Expected Pipeline Behavior
1. **Extraction:** Clean text extraction — this is a text-heavy PDF with simple layout. Should be straightforward.
2. **Normalization:** Preserve Arabic script characters (don't strip as encoding errors). Preserve Malay honorifics and greetings.
3. **Language detection:** Should detect as English. The Malay greetings and Arabic verses are not code-switching — they are ceremonial/religious inclusions in a formally English document.
4. **Cross-language summary:** When `target_language: "ms"` is requested, the pipeline should summarize the English content and produce output in BM. Malaysian policy terms should map to their BM equivalents where standard (e.g., "Thirteenth Malaysia Plan" → "Rancangan Malaysia Ketiga Belas").

### What This Tests
- Clean PDF text extraction (baseline — should be easy)
- Handling Arabic/Unicode script within English text
- Preservation of honorifics and cultural terms
- Cross-language summarization (EN source → BM output)

---

## Use Case 4: DOSM Economic Statistics Review — Bilingual Statistical Report

**Fixture:** `dosm_mesr_2024_en.pdf` (2.5 MB)
**Source:** Malaysian Economic Statistics Review Vol. 11, 2024, Department of Statistics Malaysia

### Document Characteristics
- **Language:** English-primary with bilingual institutional headers
- **Pages:** ~67 pages
- **Layout:** Cover page, table of contents, narrative sections, statistical tables, charts/graphs
- **Structure:** Sections covering GDP overview, agriculture, industry/manufacturing, services, external sector, labour, prices. Includes "Key Economic Indicators" snapshot pages.
- **Bilingual elements:**
  - Institutional name always bilingual: "JABATAN PERANGKAAN MALAYSIA / DEPARTMENT OF STATISTICS, MALAYSIA"
  - "MINISTRY OF ECONOMY / KEMENTERIAN EKONOMI" on headers
  - Table headers may use abbreviated BM terms
- **Edge cases:** Heavy use of statistical tables with numbers, percentages, and economic indicators. Charts/graphs should be skipped. Publisher/copyright page contains contact details that should not be summarized.

### API Request
```json
POST /v1/summarize
{
  "document": "<base64-encoded PDF bytes>",
  "document_type": "pdf",
  "config": {
    "target_language": "en",
    "summary_type": "detailed",
    "max_length": 800
  }
}
```

### Expected Pipeline Behavior
1. **Extraction:** Text extraction from mixed content — narrative prose + statistical tables. Tables should be extracted as structured text (not garbled). Chart labels may be extractable but chart visual data is not.
2. **Normalization:** Strip repeated headers/footers. Publisher info and copyright pages should be excluded.
3. **Language detection:** Should detect as English. Bilingual headers are institutional convention, not code-switching.
4. **Summary output:** Should capture key economic indicators and trends (GDP growth rate, sector performance, trade balance) rather than listing raw numbers.

### What This Tests
- Table extraction from PDFs
- Handling statistical/numerical content
- Meaningful summarization of data-heavy documents (narrative over raw numbers)
- Bilingual header handling

---

## Use Case 5: Cross-Language — BM Budget Speech → English Summary

Uses the same fixture as Use Case 2 (`budget_speech_2026_bm.pdf`) but requests English output.

### API Request
```json
POST /v1/summarize
{
  "document": "<base64-encoded PDF bytes>",
  "document_type": "pdf",
  "config": {
    "target_language": "en",
    "summary_type": "executive",
    "max_length": 800
  }
}
```

### Expected Pipeline Behavior
1. All preprocessing steps same as Use Case 2.
2. **Language detection:** Detects BM with code-switching.
3. **Cross-language summarization:** Summarize BM content, produce English output. Financial terms that were already in English in the source should remain as-is. BM policy terms should be translated where standard English equivalents exist.
4. **Quality signal:** Since we have the official EN translation (Use Case 3), we can compare the cross-language summary against it as a reference for faithfulness — though this is a future evaluation concern, not a preprocessing one.

### What This Tests
- Full pipeline: large graphical BM PDF → English summary
- Cross-language summarization quality
- Consistency check potential (same source document in two languages)

---

## Implementation Notes for Next Session

### Current State
- `DocumentProcessor` only supports `document_type: "txt"` today
- `SummarizeRequest.document_type` is `Literal["txt"]` — needs extending to include `"pdf"`
- No PDF extraction library installed yet

### What Needs Building
1. **PDF text extraction** — add PyMuPDF (`pymupdf`) dependency, extend `DocumentProcessor` to handle `pdf` type
2. **Base64 document handling** — API currently accepts `document` as a string. PDF bytes need to be base64-encoded by the client and decoded by the API
3. **Normalization** — existing `_normalize()` method should work for extracted PDF text, but may need enhancements for header/footer stripping
4. **Tests** — integration tests using the fixture PDFs in `tests/fixtures/pdf/`

### Fixture Files
```
tests/fixtures/pdf/
├── bnm_annual_report_2021_en.pdf    (8.8 MB) — EN financial report, two-column, large
├── budget_speech_2026_bm.pdf        (33 MB)  — BM with EN code-switching, graphically heavy
├── budget_speech_2026_en.pdf        (876 KB) — EN translation, clean text layout
└── dosm_mesr_2024_en.pdf            (2.5 MB) — EN statistical report, tables + narrative
```

### Priority Order for Implementation
1. Use Case 3 (EN budget speech) — simplest PDF, clean layout, validates basic extraction
2. Use Case 4 (DOSM statistics) — moderate complexity, tests table handling
3. Use Case 1 (BNM annual report) — large document, tests chunking
4. Use Case 2 (BM budget speech) — graphically heavy, tests resilience
5. Use Case 5 (cross-language) — depends on Use Case 2 working first
