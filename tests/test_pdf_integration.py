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
        result = processor.process(
            _load_b64("budget_speech_2026_en.pdf"), doc_type="pdf"
        )
        assert len(result.text) > 5_000

    def test_contains_key_content(self, processor):
        result = processor.process(
            _load_b64("budget_speech_2026_en.pdf"), doc_type="pdf"
        )
        assert "MADANI" in result.text

    def test_preserves_cultural_terms(self, processor):
        result = processor.process(
            _load_b64("budget_speech_2026_en.pdf"), doc_type="pdf"
        )
        has_cultural = any(
            t in result.text for t in ["Assalamualaikum", "Bismillah", "Dewan", "Dato"]
        )
        assert has_cultural, "Expected Malaysian cultural terms preserved"

    def test_normalization_clean(self, processor):
        result = processor.process(
            _load_b64("budget_speech_2026_en.pdf"), doc_type="pdf"
        )
        assert "\n\n\n" not in result.text


class TestDOSMStatistics:
    """Use Case 4: EN statistical report with tables."""

    def test_extracts_substantial_text(self, processor):
        result = processor.process(_load_b64("dosm_mesr_2024_en.pdf"), doc_type="pdf")
        assert len(result.text) > 5_000

    def test_contains_economic_content(self, processor):
        result = processor.process(_load_b64("dosm_mesr_2024_en.pdf"), doc_type="pdf")
        text_lower = result.text.lower()
        has_econ = any(
            t in text_lower for t in ["gdp", "growth", "economy", "statistics"]
        )
        assert has_econ, "Expected economic terminology"


class TestBNMAnnualReport:
    """Use Case 1: Large EN financial report, two-column."""

    def test_extracts_substantial_text(self, processor):
        result = processor.process(
            _load_b64("bnm_annual_report_2021_en.pdf"), doc_type="pdf"
        )
        assert len(result.text) > 10_000

    def test_contains_financial_content(self, processor):
        result = processor.process(
            _load_b64("bnm_annual_report_2021_en.pdf"), doc_type="pdf"
        )
        text_lower = result.text.lower()
        assert "bank negara" in text_lower or "monetary" in text_lower

    def test_headers_not_excessively_repeated(self, processor):
        result = processor.process(
            _load_b64("bnm_annual_report_2021_en.pdf"), doc_type="pdf"
        )
        from collections import Counter

        line_counts = Counter(
            line.strip()
            for line in result.text.split("\n")
            if line.strip() and len(line.strip()) > 5
        )
        if line_counts:
            _, top_count = line_counts.most_common(1)[0]
            assert top_count < 60, (
                f"Line repeated {top_count}x — header stripping issue"
            )


class TestBudgetSpeechBM:
    """Use Case 2: BM with code-switching, graphically heavy (33 MB)."""

    def test_extracts_some_text(self, processor):
        result = processor.process(
            _load_b64("budget_speech_2026_bm.pdf"), doc_type="pdf"
        )
        assert len(result.text) > 500

    def test_contains_malay_content(self, processor):
        result = processor.process(
            _load_b64("budget_speech_2026_bm.pdf"), doc_type="pdf"
        )
        text_lower = result.text.lower()
        has_bm = any(
            t in text_lower for t in ["tekad", "belanjawan", "rakyat", "madani"]
        )
        assert has_bm, "Expected BM terms in budget speech"

    def test_completes_without_memory_issues(self, processor):
        result = processor.process(
            _load_b64("budget_speech_2026_bm.pdf"), doc_type="pdf"
        )
        assert result.estimated_tokens > 0


def test_chunker_on_real_pdf(processor):
    """Verify chunker produces valid chunks from a real extracted PDF."""
    from app.chunker import DocumentChunker

    b64 = _load_b64("budget_speech_2026_en.pdf")
    doc = processor.process(b64, doc_type="pdf")

    chunker = DocumentChunker(max_tokens_per_chunk=5000)
    chunks = chunker.chunk(doc.text)

    assert len(chunks) >= 1
    assert all(c.estimated_tokens <= 5000 for c in chunks)
    assert all(c.text.strip() for c in chunks)
    # Indices are sequential
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # No content lost — all chunk text combined should cover the original
    combined = "\n\n".join(c.text for c in chunks)
    assert len(combined) >= len(doc.text) * 0.9
