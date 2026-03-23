# PDF Preprocessing & Text Normalization Pipeline — Implementation Plan (Team Edition)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the `/v1/summarize` API to accept PDF documents with production-grade text extraction and enhanced text normalization for both `txt` and `pdf` document types — targeting real Malaysian government documents as the primary use case.

**Architecture:** The design follows Anthropic's own API pattern for document handling: clients send base64-encoded PDF bytes in the existing `document` string field, discriminated by `document_type: "pdf"`. Inside `DocumentProcessor`, the PDF path decodes base64 → opens via PyMuPDF → extracts per-page text → strips repeated headers/footers → feeds into the shared normalization pipeline (ftfy encoding repair → Unicode NFC → whitespace normalization). The TXT path also gains ftfy. No OCR, no structured table extraction, no chunking in this iteration — those are orthogonal concerns documented as future work.

**Tech Stack:** PyMuPDF (`pymupdf`) for PDF text extraction, `ftfy` for encoding repair, `base64` stdlib for decoding, existing FastAPI + Pydantic stack.

**Key Design Decisions:**
1. **base64 in `document` field** — not multipart upload. Matches Anthropic's content block pattern, keeps the API contract simple, and avoids breaking existing consumers.
2. **ftfy before NFC** — ftfy repairs mojibake *then* NFC normalizes. Order matters: ftfy needs to see the broken bytes to fix them.
3. **Per-page extraction then join** — extract text per page, strip headers/footers per page, *then* join with `\n\n`. This preserves page boundary signal for the header/footer heuristic.
4. **Size limit increase** — base64 is ~33% larger than raw bytes. The 33MB BM budget speech fixture becomes ~44MB base64. Current 500K limit must increase to ~50MB.
5. **Fail loud on empty** — a PDF that yields zero extractable text raises `ValueError`, not a silent empty summary. The API returns 400, not 200 with garbage.

---

### Task 1: Add `pymupdf` and `ftfy` dependencies

**Files:**
- Modify: `pyproject.toml:6-13`

**Step 1: Add dependencies**

In `pyproject.toml`, update the `dependencies` list:

```toml
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "pydantic-settings>=2.0",
    "httpx>=0.27.0",
    "pymupdf>=1.25.0",
    "ftfy>=6.0",
]
```

**Step 2: Install and verify**

Run: `uv sync && uv run python -c "import fitz; import ftfy; print(fitz.version, ftfy.__version__)"`
Expected: Version numbers printed, no errors.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pymupdf and ftfy dependencies for PDF preprocessing"
```

---

### Task 2: Enhance `_normalize()` with ftfy encoding repair

Applies to **both** TXT and PDF paths. This is the foundation — everything extracted from PDFs will flow through this.

**Files:**
- Modify: `app/processor.py:1-4, 29-40`
- Modify: `tests/test_processor.py`

**Step 1: Write the failing tests**

Add to `tests/test_processor.py`:

```python
def test_normalize_repairs_mojibake():
    """ftfy fixes Windows-1252 mojibake common in Malaysian government PDFs."""
    processor = DocumentProcessor()
    # "naïve" misencoded: UTF-8 bytes interpreted as Latin-1 then re-encoded
    mojibake = "na\u00c3\u00afve"
    result = processor.process(mojibake, doc_type="txt")
    assert "\u00c3" not in result.text  # The broken character should be gone


def test_normalize_preserves_malay_diacritics():
    """ftfy + NFC should not damage legitimate Malay Unicode characters."""
    processor = DocumentProcessor()
    malay_text = "Imbangan Antarabangsa"
    result = processor.process(malay_text, doc_type="txt")
    assert "Imbangan" in result.text
    assert "Antarabangsa" in result.text
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_processor.py::test_normalize_repairs_mojibake -v`
Expected: FAIL — `\u00c3` still present because ftfy isn't applied.

**Step 3: Implement**

In `app/processor.py`, add `import ftfy` and insert `text = ftfy.fix_text(text)` as the **first line** of `_normalize()`, before Unicode NFC:

```python
import re
import unicodedata
from dataclasses import dataclass

import ftfy


@dataclass
class ProcessedDocument:
    text: str
    estimated_tokens: int


class DocumentProcessor:
    SUPPORTED_TYPES = {"txt"}

    def process(self, raw: str, doc_type: str) -> ProcessedDocument:
        if doc_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported document type: {doc_type}")

        text = self._normalize(raw)

        if not text:
            raise ValueError("Document is empty after processing")

        return ProcessedDocument(
            text=text,
            estimated_tokens=self._estimate_tokens(text),
        )

    def _normalize(self, text: str) -> str:
        # Fix encoding errors (mojibake, curly quotes, etc.)
        text = ftfy.fix_text(text)
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)
        # Collapse multiple spaces to single
        text = re.sub(r"[ \t]+", " ", text)
        # Clean spaces around newlines
        text = re.sub(r" ?\n ?", "\n", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
```

**Step 4: Run all processor tests**

Run: `uv run pytest tests/test_processor.py -v`
Expected: ALL PASS — existing tests unaffected, new tests pass.

**Step 5: Commit**

```bash
git add app/processor.py tests/test_processor.py
git commit -m "feat: add ftfy encoding repair to text normalization pipeline"
```

---

### Task 3: Build header/footer stripping method

PDF text extraction produces repeated header/footer lines on every page. Build a heuristic to detect and remove them. This method operates on a `list[str]` (one string per page) and is tested in isolation before wiring into the PDF extraction path.

**Files:**
- Modify: `app/processor.py`
- Modify: `tests/test_processor.py`

**Step 1: Write the failing tests**

Add to `tests/test_processor.py`:

```python
def test_strip_headers_footers_removes_repeated():
    """Lines appearing in first/last 3 lines of >50% of pages are headers/footers."""
    processor = DocumentProcessor()
    pages = [
        "Annual Report 2021\nContent about monetary policy.\nPage 1",
        "Annual Report 2021\nContent about fiscal stability.\nPage 2",
        "Annual Report 2021\nContent about GDP growth.\nPage 3",
        "Annual Report 2021\nContent about trade balance.\nPage 4",
    ]
    result = processor._strip_headers_footers(pages)
    joined = "\n".join(result)
    assert "Annual Report 2021" not in joined
    assert "monetary policy" in joined
    assert "fiscal stability" in joined


def test_strip_headers_footers_removes_page_numbers():
    """Standalone page number lines should be stripped."""
    processor = DocumentProcessor()
    pages = [
        "Content one.\n1",
        "Content two.\n2",
        "Content three.\n3",
    ]
    result = processor._strip_headers_footers(pages)
    joined = "\n".join(result)
    assert "Content one" in joined
    lines = [l.strip() for l in joined.split("\n") if l.strip()]
    assert all(not l.isdigit() for l in lines)


def test_strip_headers_footers_preserves_content():
    """Unique content lines must not be removed."""
    processor = DocumentProcessor()
    pages = [
        "Header\nUnique content A.\nFooter",
        "Header\nUnique content B.\nFooter",
        "Header\nUnique content C.\nFooter",
    ]
    result = processor._strip_headers_footers(pages)
    joined = "\n".join(result)
    assert "Unique content A" in joined
    assert "Unique content B" in joined
    assert "Unique content C" in joined


def test_strip_headers_footers_skips_few_pages():
    """With <3 pages, not enough signal — return pages unchanged."""
    processor = DocumentProcessor()
    pages = ["Header\nContent.\nFooter", "Header\nMore content.\nFooter"]
    result = processor._strip_headers_footers(pages)
    assert result == pages
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_processor.py::test_strip_headers_footers_removes_repeated -v`
Expected: FAIL — `AttributeError: 'DocumentProcessor' object has no attribute '_strip_headers_footers'`

**Step 3: Implement**

Add to `DocumentProcessor` in `app/processor.py`:

```python
    def _strip_headers_footers(self, pages: list[str]) -> list[str]:
        """Remove repeated header/footer lines from paginated text.

        Heuristic: lines in the first or last 3 lines of a page that appear
        across >50% of pages are likely headers/footers.
        """
        if len(pages) < 3:
            return pages

        threshold = len(pages) * 0.5

        from collections import Counter
        header_counts: Counter[str] = Counter()
        footer_counts: Counter[str] = Counter()

        for page in pages:
            lines = page.strip().split("\n")
            for line in lines[:3]:
                s = line.strip()
                if s:
                    header_counts[s] += 1
            for line in lines[-3:]:
                s = line.strip()
                if s:
                    footer_counts[s] += 1

        to_remove = set()
        for line, count in header_counts.items():
            if count >= threshold:
                to_remove.add(line)
        for line, count in footer_counts.items():
            if count >= threshold:
                to_remove.add(line)

        page_num_re = re.compile(r"^(?:page\s+)?\d+$|^-\s*\d+\s*-$", re.IGNORECASE)

        cleaned = []
        for page in pages:
            lines = page.split("\n")
            filtered = [
                line for line in lines
                if line.strip() not in to_remove
                and not page_num_re.match(line.strip())
            ]
            cleaned.append("\n".join(filtered))
        return cleaned
```

**Step 4: Run all processor tests**

Run: `uv run pytest tests/test_processor.py -v`
Expected: ALL PASS.

**Step 5: Commit**

```bash
git add app/processor.py tests/test_processor.py
git commit -m "feat: add header/footer stripping heuristic for paginated documents"
```

---

### Task 4: Extend data model — add `"pdf"` document type and increase size limit

**Files:**
- Modify: `app/models.py:13-16`
- Modify: `app/config.py`
- Modify: `tests/test_models.py`

**Step 1: Write the failing tests**

Add to `tests/test_models.py`:

```python
def test_request_accepts_pdf_type():
    req = SummarizeRequest(document="JVBERi0xLjQK", document_type="pdf")
    assert req.document_type == "pdf"


def test_request_rejects_docx_type():
    with pytest.raises(Exception):
        SummarizeRequest(document="data", document_type="docx")


def test_request_accepts_large_pdf_payload():
    """Base64-encoded PDFs can be very large — 33MB PDF = ~44MB base64."""
    large = "A" * 2_000_000
    req = SummarizeRequest(document=large, document_type="pdf")
    assert len(req.document) == 2_000_000
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models.py::test_request_accepts_pdf_type -v`
Expected: FAIL — `"pdf"` not in `Literal["txt"]`.

**Step 3: Update models and config**

In `app/models.py`:

```python
class SummarizeRequest(BaseModel):
    document: str = Field(max_length=50_000_000)  # ~37.5MB raw after base64 decode
    document_type: Literal["txt", "pdf"] = "txt"
    config: SummarizeConfig = SummarizeConfig()
```

In `app/config.py`, add:

```python
    MAX_DOCUMENT_SIZE: int = 50_000_000  # 50MB base64 ceiling
```

**Step 4: Run all model tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: ALL PASS.

**Step 5: Commit**

```bash
git add app/models.py app/config.py tests/test_models.py
git commit -m "feat: extend API to accept pdf document type, increase size limit to 50MB"
```

---

### Task 5: Implement PDF text extraction in `DocumentProcessor`

The core feature. Wire base64 decode → PyMuPDF → header/footer stripping → normalization.

**Files:**
- Modify: `app/processor.py`
- Modify: `tests/test_processor.py`

**Step 1: Write the failing tests**

Add to `tests/test_processor.py`:

```python
import base64
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdf"


def _load_pdf_b64(name: str) -> str:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture {name} not available")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_process_pdf_extracts_text():
    """EN budget speech is clean text — basic extraction smoke test."""
    processor = DocumentProcessor()
    b64 = _load_pdf_b64("budget_speech_2026_en.pdf")
    result = processor.process(b64, doc_type="pdf")
    assert len(result.text) > 1000
    assert result.estimated_tokens > 100
    assert "MADANI" in result.text or "budget" in result.text.lower()


def test_process_pdf_normalization_applied():
    """Extracted PDF text should go through the shared normalization pipeline."""
    processor = DocumentProcessor()
    b64 = _load_pdf_b64("budget_speech_2026_en.pdf")
    result = processor.process(b64, doc_type="pdf")
    assert "\n\n\n" not in result.text
    assert result.text == result.text.strip()


def test_process_pdf_invalid_base64_raises():
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="base64"):
        processor.process("!!!not-base64!!!", doc_type="pdf")


def test_process_pdf_not_a_pdf_raises():
    processor = DocumentProcessor()
    b64 = base64.b64encode(b"this is plain text, not a PDF").decode()
    with pytest.raises(ValueError, match="PDF"):
        processor.process(b64, doc_type="pdf")


def test_process_pdf_empty_pdf_raises():
    """A valid PDF with zero text content should raise ValueError."""
    import fitz
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    b64 = base64.b64encode(pdf_bytes).decode()
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="empty"):
        processor.process(b64, doc_type="pdf")
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_processor.py::test_process_pdf_extracts_text -v`
Expected: FAIL — `ValueError: Unsupported document type: pdf`

**Step 3: Implement PDF extraction**

Update `app/processor.py` — the complete final file:

```python
import base64 as b64_mod
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import fitz  # PyMuPDF
import ftfy


@dataclass
class ProcessedDocument:
    text: str
    estimated_tokens: int


class DocumentProcessor:
    SUPPORTED_TYPES = {"txt", "pdf"}

    def process(self, raw: str, doc_type: str) -> ProcessedDocument:
        if doc_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported document type: {doc_type}")

        if doc_type == "pdf":
            text = self._extract_pdf(raw)
        else:
            text = raw

        text = self._normalize(text)

        if not text:
            raise ValueError("Document is empty after processing")

        return ProcessedDocument(
            text=text,
            estimated_tokens=self._estimate_tokens(text),
        )

    def _extract_pdf(self, b64_data: str) -> str:
        """Decode base64 → open PDF → extract per-page text → strip headers/footers."""
        try:
            pdf_bytes = b64_mod.b64decode(b64_data, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}") from e

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {e}") from e

        try:
            pages = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append(text)
        finally:
            doc.close()

        if not pages:
            raise ValueError("Document is empty after processing")

        pages = self._strip_headers_footers(pages)
        return "\n\n".join(pages)

    def _normalize(self, text: str) -> str:
        text = ftfy.fix_text(text)
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def _strip_headers_footers(self, pages: list[str]) -> list[str]:
        """Remove repeated header/footer lines from paginated text."""
        if len(pages) < 3:
            return pages

        threshold = len(pages) * 0.5
        header_counts: Counter[str] = Counter()
        footer_counts: Counter[str] = Counter()

        for page in pages:
            lines = page.strip().split("\n")
            for line in lines[:3]:
                s = line.strip()
                if s:
                    header_counts[s] += 1
            for line in lines[-3:]:
                s = line.strip()
                if s:
                    footer_counts[s] += 1

        to_remove = set()
        for line, count in header_counts.items():
            if count >= threshold:
                to_remove.add(line)
        for line, count in footer_counts.items():
            if count >= threshold:
                to_remove.add(line)

        page_num_re = re.compile(r"^(?:page\s+)?\d+$|^-\s*\d+\s*-$", re.IGNORECASE)

        cleaned = []
        for page in pages:
            lines = page.split("\n")
            filtered = [
                line for line in lines
                if line.strip() not in to_remove
                and not page_num_re.match(line.strip())
            ]
            cleaned.append("\n".join(filtered))
        return cleaned

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
```

**Step 4: Run all processor tests**

Run: `uv run pytest tests/test_processor.py -v`
Expected: ALL PASS.

**Step 5: Commit**

```bash
git add app/processor.py tests/test_processor.py
git commit -m "feat: implement PDF text extraction via PyMuPDF with header/footer stripping"
```

---

### Task 6: Fix existing test that uses `pdf` as unsupported type

`test_process_unsupported_type_raises` in `tests/test_processor.py:39-42` currently asserts `doc_type="pdf"` raises. Now that PDF is supported, update it.

**Files:**
- Modify: `tests/test_processor.py:39-42`

**Step 1: Update the test**

```python
def test_process_unsupported_type_raises():
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="Unsupported"):
        processor.process("data", doc_type="docx")
```

**Step 2: Run full suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS.

**Step 3: Commit**

```bash
git add tests/test_processor.py
git commit -m "test: update unsupported-type test to use docx instead of pdf"
```

---

### Task 7: API integration tests for PDF endpoint

Verify the full request → processor → provider flow works for PDF documents at the API level.

**Files:**
- Modify: `tests/test_main.py`

**Step 1: Write the tests**

Add to `tests/test_main.py`:

```python
import base64
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdf"


@pytest.mark.asyncio
async def test_summarize_pdf_document(app, mock_provider):
    """Full API flow: PDF base64 → extraction → provider → response."""
    pdf_path = FIXTURE_DIR / "budget_speech_2026_en.pdf"
    if not pdf_path.exists():
        pytest.skip("PDF fixture not available")
    b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": b64,
            "document_type": "pdf",
            "config": {"target_language": "en", "summary_type": "brief"},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] == "Test summary output."
    # Provider received extracted text, not base64
    call_args = mock_provider.summarize.call_args
    text_sent = call_args[0][0]
    assert len(text_sent) > 100
    assert "JVBERi0" not in text_sent


@pytest.mark.asyncio
async def test_summarize_pdf_invalid_base64_returns_400(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": "!!!invalid-base64!!!",
            "document_type": "pdf",
        })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_summarize_pdf_corrupted_returns_400(app):
    b64 = base64.b64encode(b"not a real PDF file").decode()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/summarize", json={
            "document": b64,
            "document_type": "pdf",
        })
    assert resp.status_code == 400
```

**Step 2: Run API tests**

Run: `uv run pytest tests/test_main.py -v`
Expected: ALL PASS.

**Step 3: Commit**

```bash
git add tests/test_main.py
git commit -m "test: add API integration tests for PDF summarization endpoint"
```

---

### Task 8: PDF fixture integration tests — all 4 use cases

Dedicated test file exercising real Malaysian government PDFs. These are slower but catch real-world extraction issues.

**Files:**
- Create: `tests/test_pdf_integration.py`

**Step 1: Write the integration tests**

```python
"""Integration tests for PDF preprocessing with real Malaysian government fixtures.

Fixture files (tests/fixtures/pdf/):
  budget_speech_2026_en.pdf    (876 KB)  — Use Case 3: clean EN text
  dosm_mesr_2024_en.pdf        (2.5 MB)  — Use Case 4: EN statistics, tables
  bnm_annual_report_2021_en.pdf (8.8 MB) — Use Case 1: large EN, two-column
  budget_speech_2026_bm.pdf    (33 MB)   — Use Case 2: BM, graphically heavy
"""
import base64
from pathlib import Path

import pytest

from app.processor import DocumentProcessor

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdf"


def _load_b64(name: str) -> str:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture not found: {name}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.fixture
def processor():
    return DocumentProcessor()


class TestBudgetSpeechEN:
    """Use Case 3: Clean EN text — baseline."""

    def test_extracts_substantial_text(self, processor):
        result = processor.process(_load_b64("budget_speech_2026_en.pdf"), doc_type="pdf")
        assert len(result.text) > 5_000

    def test_contains_key_content(self, processor):
        result = processor.process(_load_b64("budget_speech_2026_en.pdf"), doc_type="pdf")
        assert "MADANI" in result.text

    def test_preserves_cultural_terms(self, processor):
        result = processor.process(_load_b64("budget_speech_2026_en.pdf"), doc_type="pdf")
        has_cultural = any(t in result.text for t in ["Assalamualaikum", "Bismillah", "Dewan", "Dato"])
        assert has_cultural, "Expected Malaysian cultural terms preserved"

    def test_normalization_clean(self, processor):
        result = processor.process(_load_b64("budget_speech_2026_en.pdf"), doc_type="pdf")
        assert "\n\n\n" not in result.text


class TestDOSMStatistics:
    """Use Case 4: EN statistical report with tables."""

    def test_extracts_substantial_text(self, processor):
        result = processor.process(_load_b64("dosm_mesr_2024_en.pdf"), doc_type="pdf")
        assert len(result.text) > 5_000

    def test_contains_economic_content(self, processor):
        result = processor.process(_load_b64("dosm_mesr_2024_en.pdf"), doc_type="pdf")
        text_lower = result.text.lower()
        has_econ = any(t in text_lower for t in ["gdp", "growth", "economy", "statistics"])
        assert has_econ, "Expected economic terminology"


class TestBNMAnnualReport:
    """Use Case 1: Large EN financial report, two-column."""

    def test_extracts_substantial_text(self, processor):
        result = processor.process(_load_b64("bnm_annual_report_2021_en.pdf"), doc_type="pdf")
        assert len(result.text) > 10_000

    def test_contains_financial_content(self, processor):
        result = processor.process(_load_b64("bnm_annual_report_2021_en.pdf"), doc_type="pdf")
        text_lower = result.text.lower()
        assert "bank negara" in text_lower or "monetary" in text_lower

    def test_headers_not_excessively_repeated(self, processor):
        result = processor.process(_load_b64("bnm_annual_report_2021_en.pdf"), doc_type="pdf")
        from collections import Counter
        line_counts = Counter(
            l.strip() for l in result.text.split("\n") if l.strip() and len(l.strip()) > 5
        )
        if line_counts:
            _, top_count = line_counts.most_common(1)[0]
            assert top_count < 15, f"Line repeated {top_count}x — header stripping issue"


class TestBudgetSpeechBM:
    """Use Case 2: BM with code-switching, graphically heavy (33 MB)."""

    def test_extracts_some_text(self, processor):
        result = processor.process(_load_b64("budget_speech_2026_bm.pdf"), doc_type="pdf")
        assert len(result.text) > 500

    def test_contains_malay_content(self, processor):
        result = processor.process(_load_b64("budget_speech_2026_bm.pdf"), doc_type="pdf")
        text_lower = result.text.lower()
        has_bm = any(t in text_lower for t in ["tekad", "belanjawan", "rakyat", "madani"])
        assert has_bm, "Expected BM terms in budget speech"

    def test_completes_without_memory_issues(self, processor):
        result = processor.process(_load_b64("budget_speech_2026_bm.pdf"), doc_type="pdf")
        assert result.estimated_tokens > 0
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/test_pdf_integration.py -v`
Expected: ALL PASS (BNM/BM tests may be slow — expected).

**Step 3: Commit**

```bash
git add tests/test_pdf_integration.py
git commit -m "test: add PDF fixture integration tests for all 4 Malaysian government use cases"
```

---

### Task 9: Full regression check — run entire test suite + lint

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: ALL PASS — zero regressions.

**Step 2: Run linter**

Run: `uv run ruff check app/ tests/`
Expected: No errors.

**Step 3: Run formatter**

Run: `uv run ruff format --check app/ tests/`
Expected: Clean.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: lint and format fixes for PDF preprocessing pipeline"
```

---

## Execution Dependency Graph

```
Task 1 (deps) ─────────────────────────┐
   │                                    │
   ├──> Task 2 (ftfy normalization) ────┤  parallel
   │                                    │
   └──> Task 3 (header/footer strip) ───┤  parallel
                                        │
        Task 4 (model + size limit) ────┤  parallel with 2,3
                                        │
        Task 5 (PDF extraction) ────────┤  depends on 1,2,3,4
                                        │
        Task 6 (fix old test) ──────────┤  depends on 5
                                        │
        Task 7 (API tests) ────────────┤  depends on 5,4
                                        │
        Task 8 (fixture tests) ────────┤  depends on 5
                                        │
        Task 9 (regression check) ─────┘  depends on all
```

**Parallelizable:** Tasks 2, 3, 4 (after Task 1). Tasks 6, 7, 8 (after Task 5).

## File Change Summary

| File | What Changes |
|---|---|
| `pyproject.toml` | Add `pymupdf>=1.25.0`, `ftfy>=6.0` |
| `app/processor.py` | Add `_extract_pdf()`, `_strip_headers_footers()`, ftfy in `_normalize()`, extend `SUPPORTED_TYPES` |
| `app/models.py` | `document_type: Literal["txt", "pdf"]`, `max_length=50_000_000` |
| `app/config.py` | Add `MAX_DOCUMENT_SIZE` setting |
| `Makefile` | Add `--limit-max-request-size 0` to `dev` and `run` targets (already done) |
| `Dockerfile` | Add `--limit-max-request-size 0` to CMD (already done) |
| `tests/test_processor.py` | ftfy tests, header/footer tests, PDF extraction tests, fix unsupported type test |
| `tests/test_models.py` | PDF type, docx rejection, large payload tests |
| `tests/test_main.py` | API-level PDF tests (success + error cases) |
| `tests/test_pdf_integration.py` | **New** — 4 test classes for real fixture PDFs |

## Pre-applied Changes

The following changes have already been made and should NOT be repeated during plan execution:
- `Makefile`: `--limit-max-request-size 0` added to `dev` and `run` targets
- `Dockerfile`: `--limit-max-request-size 0` added to CMD

These remove uvicorn's default 16MB request body limit, allowing the 33MB BM budget speech PDF (~44MB base64) to be sent through the API.

## Explicitly Out of Scope

- **OCR** (Tesseract) — future work for scanned PDFs
- **Multi-column layout detection** — PyMuPDF's default handles many cases adequately
- **Structured table extraction** — camelot/pdfplumber, separate feature
- **Document chunking** — summarization concern, not preprocessing
- **DOCX/HTML support** — excluded per requirements
