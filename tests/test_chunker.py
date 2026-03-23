from app.chunker import DocumentChunker


class TestSplitSections:
    """Test structure detection and section splitting."""

    def test_split_on_markdown_headings(self):
        text = "# Introduction\nFirst section content.\n\n# Methods\nSecond section content."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 2
        assert "Introduction" in sections[0]
        assert "Methods" in sections[1]

    def test_split_on_allcaps_headers(self):
        """PDF-extracted text with ALL-CAPS section titles (e.g. Malaysian budget speech)."""
        text = "PREAMBLE\nOpening remarks.\n\nFISCAL POLICY AND PUBLIC FINANCE\nBudget details.\n\nDEVELOPMENT EXPENDITURE\nCapital spending."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 3
        assert "PREAMBLE" in sections[0]
        assert "FISCAL POLICY" in sections[1]

    def test_split_on_numbered_sections(self):
        text = "1.0 Pengenalan\nKandungan pertama.\n\n2.0 Kaedah\nKandungan kedua."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 2
        assert "Pengenalan" in sections[0]
        assert "Kaedah" in sections[1]

    def test_split_on_bahagian_headers(self):
        """Malaysian government docs use BAHAGIAN (Part) headers."""
        text = "BAHAGIAN I\nDasar fiskal.\n\nBAHAGIAN II\nDasar monetari."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 2

    def test_fallback_to_paragraph_breaks(self):
        """No headings -- split on double-newline paragraphs."""
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 3

    def test_preamble_before_first_heading_preserved(self):
        """Text before the first heading should not be dropped."""
        text = "Preamble text here.\n\n# Introduction\nContent.\n\n# Methods\nMore."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 3
        assert "Preamble" in sections[0]
        assert "Introduction" in sections[1]
        assert "Methods" in sections[2]

    def test_single_block_no_split(self):
        text = "Just one block of text with no structure."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 1
        assert sections[0] == text
