from app.chunker import DocumentChunker, Chunk


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


class TestChunk:
    """Test the main chunk() method — token-aware splitting and merging."""

    def test_short_text_returns_single_chunk(self):
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        chunks = chunker.chunk("Short text that fits easily.")
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].text == "Short text that fits easily."

    def test_sections_within_budget_merged(self):
        """Adjacent small sections should be merged into one chunk."""
        sections = "# A\nSmall.\n\n# B\nAlso small.\n\n# C\nTiny."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        chunks = chunker.chunk(sections)
        assert len(chunks) == 1  # All fit in one chunk

    def test_large_sections_split_into_multiple_chunks(self):
        """Sections exceeding budget should produce multiple chunks."""
        # ~100 tokens per section, budget of 120 tokens — forces splitting
        section_a = "# Section A\n" + ("Word " * 100)
        section_b = "# Section B\n" + ("Word " * 100)
        chunker = DocumentChunker(max_tokens_per_chunk=120)
        chunks = chunker.chunk(section_a + "\n\n" + section_b)
        assert len(chunks) >= 2
        assert all(c.estimated_tokens <= 120 for c in chunks)

    def test_oversized_paragraph_split_on_sentences(self):
        """A single paragraph too large for budget gets sentence-split."""
        long_para = ". ".join(["This is sentence number " + str(i) for i in range(200)])
        chunker = DocumentChunker(max_tokens_per_chunk=200)
        chunks = chunker.chunk(long_para)
        assert len(chunks) >= 2
        assert all(c.estimated_tokens <= 200 for c in chunks)

    def test_chunk_indices_sequential(self):
        section_a = "# A\n" + ("Word " * 100)
        section_b = "# B\n" + ("Word " * 100)
        chunker = DocumentChunker(max_tokens_per_chunk=120)
        chunks = chunker.chunk(section_a + "\n\n" + section_b)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_token_estimates_populated(self):
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        chunks = chunker.chunk("Some text content here.")
        assert chunks[0].estimated_tokens > 0

    def test_oversized_sentence_split_on_words(self):
        """A single long sentence with no periods triggers word-level split."""
        long_sentence = " ".join(["word"] * 2000)
        chunker = DocumentChunker(max_tokens_per_chunk=200)
        chunks = chunker.chunk(long_sentence)
        assert len(chunks) >= 2
        assert all(c.estimated_tokens <= 200 for c in chunks)
