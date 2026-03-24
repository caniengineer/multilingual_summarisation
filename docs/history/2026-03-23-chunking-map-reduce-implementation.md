# Hierarchical Chunking & Map-Reduce Summarization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a chunking module and summarization orchestrator so long documents are automatically split, summarized per-chunk (map), and merged into a unified summary (reduce).

**Architecture:** New `app/chunker.py` (pure chunking logic) + new `app/summarizer.py` (map-reduce orchestrator). Provider protocol gets a `reduce()` method. `main.py` calls the orchestrator instead of the provider directly.

**Tech Stack:** Python, FastAPI, pytest, PyMuPDF (existing), Anthropic SDK (existing)

**Design doc:** `docs/plans/2026-03-23-chunking-map-reduce-design.md`

---

### Task 1: DocumentChunker — Structure Detection & Splitting

**Files:**
- Create: `app/chunker.py`
- Test: `tests/test_chunker.py`

**Step 1: Write the failing tests**

```python
# tests/test_chunker.py
import pytest
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
        """No headings — split on double-newline paragraphs."""
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 3

    def test_single_block_no_split(self):
        text = "Just one block of text with no structure."
        chunker = DocumentChunker(max_tokens_per_chunk=5000)
        sections = chunker._split_sections(text)
        assert len(sections) == 1
        assert sections[0] == text
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunker.py::TestSplitSections -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.chunker'`

**Step 3: Write minimal implementation**

```python
# app/chunker.py
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    estimated_tokens: int


class DocumentChunker:
    # Heading patterns ordered by priority
    _HEADING_PATTERNS = [
        re.compile(r"^#{1,6}\s+.+", re.MULTILINE),                          # Markdown headings
        re.compile(r"^[A-Z][A-Z\s:&]{5,}$", re.MULTILINE),                  # ALL-CAPS section headers (PDF-extracted)
        re.compile(r"^\d+\.\d*\s+\S.+", re.MULTILINE),                      # Numbered sections (1.0, 1.1)
        re.compile(r"^BAHAGIAN\s+[IVXLCDM]+\b.*", re.MULTILINE | re.IGNORECASE),  # Malaysian government
    ]

    def __init__(self, max_tokens_per_chunk: int = 50_000):
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _split_sections(self, text: str) -> list[str]:
        """Split text into sections using structural headings, falling back to paragraphs."""
        for pattern in self._HEADING_PATTERNS:
            matches = list(pattern.finditer(text))
            if len(matches) >= 2:
                sections = []
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    section = text[start:end].strip()
                    if section:
                        sections.append(section)
                return sections

        # Fallback: split on double-newline paragraph breaks
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        return paragraphs if paragraphs else [text]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunker.py::TestSplitSections -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/chunker.py tests/test_chunker.py
git commit -m "feat: add DocumentChunker with structure detection for headings and paragraphs"
```

---

### Task 2: DocumentChunker — Token-Aware Chunking with Greedy Merge

**Files:**
- Modify: `app/chunker.py`
- Modify: `tests/test_chunker.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_chunker.py

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunker.py::TestChunk -v`
Expected: FAIL — `Chunk` not yet imported, `chunk()` method not defined

**Step 3: Write minimal implementation**

Add to `DocumentChunker` in `app/chunker.py`:

```python
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into token-aware chunks respecting structural boundaries."""
        if self._estimate_tokens(text) <= self.max_tokens_per_chunk:
            return [Chunk(text=text, index=0, estimated_tokens=self._estimate_tokens(text))]

        sections = self._split_sections(text)
        pieces = self._split_oversized(sections)
        merged = self._greedy_merge(pieces)

        return [
            Chunk(text=t, index=i, estimated_tokens=self._estimate_tokens(t))
            for i, t in enumerate(merged)
        ]

    def _split_oversized(self, sections: list[str]) -> list[str]:
        """Sub-split any section that exceeds the token budget."""
        result = []
        for section in sections:
            if self._estimate_tokens(section) <= self.max_tokens_per_chunk:
                result.append(section)
            else:
                # Try paragraph split first
                paragraphs = [p.strip() for p in re.split(r"\n\n+", section) if p.strip()]
                for para in paragraphs:
                    if self._estimate_tokens(para) <= self.max_tokens_per_chunk:
                        result.append(para)
                    else:
                        # Last resort: sentence split
                        result.extend(self._split_sentences(para))
        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Split text on sentence boundaries to fit within token budget."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)
            if current_tokens + sent_tokens > self.max_tokens_per_chunk and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = sent_tokens
            else:
                current.append(sentence)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _greedy_merge(self, pieces: list[str]) -> list[str]:
        """Merge adjacent small pieces into chunks that fit the budget."""
        if not pieces:
            return pieces

        merged = []
        current = pieces[0]
        current_tokens = self._estimate_tokens(current)

        for piece in pieces[1:]:
            piece_tokens = self._estimate_tokens(piece)
            if current_tokens + piece_tokens <= self.max_tokens_per_chunk:
                current = current + "\n\n" + piece
                current_tokens += piece_tokens
            else:
                merged.append(current)
                current = piece
                current_tokens = piece_tokens

        merged.append(current)
        return merged
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunker.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/chunker.py tests/test_chunker.py
git commit -m "feat: add token-aware chunking with greedy merge and sentence fallback"
```

---

### Task 3: Provider Protocol — Add `reduce()` Method

**Files:**
- Modify: `app/providers/base.py`
- Modify: `app/providers/anthropic.py`
- Modify: `app/providers/claude_code.py`
- Create: `app/prompts/reduce_v1.yaml`
- Modify: `tests/test_providers.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_providers.py

def test_protocol_includes_reduce():
    """FakeProvider without reduce() should not satisfy Protocol."""
    class IncompleteProvider:
        async def summarize(self, text, config): ...
        async def evaluate(self, source, summary, target_language): ...
        @property
        def name(self): return "fake"
        @property
        def max_context_tokens(self): return 1000

    # After adding reduce to Protocol, this should fail isinstance check
    # (Protocol runtime checking is structural, so this validates the interface)
    assert not isinstance(IncompleteProvider(), SummarizationProvider)


@pytest.mark.asyncio
async def test_anthropic_reduce():
    mock_response = _mock_anthropic_response(
        json.dumps({
            "summary": "Unified summary of all sections.",
            "detected_language": "en",
            "code_switching_detected": False,
        })
    )

    provider = AnthropicProvider(api_key="sk-test", model="claude-sonnet-4-20250514")

    with patch.object(
        provider._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        config = SummarizeConfig(target_language="en", summary_type="brief", max_length=500)
        result = await provider.reduce(
            section_summaries=["Summary of part 1.", "Summary of part 2."],
            config=config,
        )

    assert result.summary == "Unified summary of all sections."
    assert result.detected_language == "en"


@pytest.mark.asyncio
async def test_claude_code_reduce():
    claude_response = {
        "result": json.dumps({
            "summary": "Combined summary.",
            "detected_language": "en",
            "code_switching_detected": False,
        }),
        "is_error": False,
        "usage": {"input_tokens": 200, "output_tokens": 60},
    }

    provider = ClaudeCodeProvider(model="claude-sonnet-4-20250514")

    with patch(
        "app.providers.claude_code.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=_mock_claude_process(claude_response),
    ):
        config = SummarizeConfig(target_language="en", summary_type="brief", max_length=500)
        result = await provider.reduce(
            section_summaries=["Part 1 summary.", "Part 2 summary."],
            config=config,
        )

    assert result.summary == "Combined summary."
    assert result.input_tokens == 200
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_providers.py::test_protocol_includes_reduce tests/test_providers.py::test_anthropic_reduce tests/test_providers.py::test_claude_code_reduce -v`
Expected: FAIL — `reduce` method not found

**Step 3: Create reduce prompt template**

```yaml
# app/prompts/reduce_v1.yaml
version: "1.0.0"
template: |
  You are an expert multilingual document summarization assistant specializing in English and Bahasa Melayu.

  You have been given section summaries from a long document that was summarized in parts. Your task is to synthesize these into a single coherent summary.

  Instructions:
  - Merge the section summaries into one unified, coherent summary.
  - Remove duplicate information that appears across sections.
  - Maintain logical flow and narrative structure.
  - Preserve domain-specific terminology without simplification.
  - Preserve key terms from both languages if code-switching was detected.
  - Target length: {max_length} words.
  - Summary type: {summary_type}

  Section summaries:
  {section_summaries}

  Return ONLY valid JSON in this exact format:
  {{"summary": "your unified summary here", "detected_language": "en or ms", "code_switching_detected": true or false}}
```

**Step 4: Add `reduce()` to Protocol and both providers**

In `app/providers/base.py`, add to the Protocol:

```python
async def reduce(self, section_summaries: list[str], config: SummarizeConfig) -> SumResult: ...
```

In `app/providers/anthropic.py`, add:

```python
async def reduce(self, section_summaries: list[str], config: SummarizeConfig) -> SumResult:
    prompt_data = self._prompt_loader.load("reduce")
    numbered = "\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(section_summaries)
    )
    rendered = self._prompt_loader.render(prompt_data["template"], {
        "max_length": config.max_length,
        "summary_type": config.summary_type,
        "section_summaries": numbered,
    })

    response = await self._client.messages.create(
        model=self._model,
        max_tokens=4096,
        messages=[{"role": "user", "content": rendered}],
    )

    result = parse_llm_json(response.content[0].text)

    return SumResult(
        summary=result["summary"],
        detected_language=result["detected_language"],
        code_switching_detected=result["code_switching_detected"],
        model_used=self._model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
```

In `app/providers/claude_code.py`, add equivalent `reduce()` using `_call_claude`.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_providers.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add app/providers/base.py app/providers/anthropic.py app/providers/claude_code.py app/prompts/reduce_v1.yaml tests/test_providers.py
git commit -m "feat: add reduce() method to provider protocol for map-reduce summarization"
```

---

### Task 4: Summarization Orchestrator

**Files:**
- Create: `app/summarizer.py`
- Create: `tests/test_summarizer.py`

**Step 1: Write the failing tests**

```python
# tests/test_summarizer.py
import pytest
from unittest.mock import AsyncMock, PropertyMock

from app.chunker import DocumentChunker
from app.models import SummarizeConfig
from app.providers.base import SumResult
from app.summarizer import summarize_document


def _make_provider(max_tokens: int = 200_000):
    """Create a mock provider with configurable context window."""
    provider = AsyncMock()
    type(provider).max_context_tokens = PropertyMock(return_value=max_tokens)
    type(provider).name = PropertyMock(return_value="mock-model")
    return provider


def _make_sum_result(summary: str, input_tokens: int = 100, output_tokens: int = 50):
    return SumResult(
        summary=summary,
        detected_language="en",
        code_switching_detected=False,
        model_used="mock-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class TestSummarizeDocument:

    @pytest.mark.asyncio
    async def test_short_text_no_chunking(self):
        """Text within context window goes directly to provider.summarize()."""
        provider = _make_provider(max_tokens=200_000)
        provider.summarize.return_value = _make_sum_result("Short summary.")
        config = SummarizeConfig()

        result = await summarize_document("Short text.", config, provider)

        provider.summarize.assert_called_once()
        provider.reduce.assert_not_called()
        assert result.summary == "Short summary."
        assert result.chunks_used is None

    @pytest.mark.asyncio
    async def test_long_text_triggers_map_reduce(self):
        """Text exceeding context window triggers chunking + map + reduce."""
        provider = _make_provider(max_tokens=100)  # Very small budget to force chunking
        provider.summarize.return_value = _make_sum_result("Section summary.", input_tokens=50, output_tokens=20)
        provider.reduce.return_value = _make_sum_result("Final unified summary.", input_tokens=80, output_tokens=30)
        config = SummarizeConfig()

        long_text = "First paragraph. " * 50 + "\n\n" + "Second paragraph. " * 50
        result = await summarize_document(long_text, config, provider)

        assert provider.summarize.call_count >= 2  # At least 2 map calls
        provider.reduce.assert_called_once()
        assert result.summary == "Final unified summary."
        assert result.chunks_used >= 2

    @pytest.mark.asyncio
    async def test_tokens_aggregated_across_calls(self):
        """Input/output tokens should be summed across map + reduce calls."""
        provider = _make_provider(max_tokens=100)
        provider.summarize.return_value = _make_sum_result("Chunk.", input_tokens=50, output_tokens=20)
        provider.reduce.return_value = _make_sum_result("Final.", input_tokens=80, output_tokens=30)
        config = SummarizeConfig()

        long_text = "Para one. " * 50 + "\n\n" + "Para two. " * 50
        result = await summarize_document(long_text, config, provider)

        # Tokens = sum of all map calls + reduce call
        assert result.input_tokens > 80  # More than just the reduce call
        assert result.output_tokens > 30

    @pytest.mark.asyncio
    async def test_context_bridge_included_after_first_chunk(self):
        """Chunks after the first should include context from the previous summary."""
        provider = _make_provider(max_tokens=100)
        provider.summarize.return_value = _make_sum_result("Previous context.", input_tokens=50, output_tokens=20)
        provider.reduce.return_value = _make_sum_result("Final.", input_tokens=80, output_tokens=30)
        config = SummarizeConfig()

        long_text = "Para one content. " * 50 + "\n\n" + "Para two content. " * 50
        result = await summarize_document(long_text, config, provider)

        # Second summarize call should have context bridge in the text
        second_call_text = provider.summarize.call_args_list[1][0][0]
        assert "[Context:" in second_call_text
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_summarizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.summarizer'`

**Step 3: Write minimal implementation**

```python
# app/summarizer.py
from dataclasses import dataclass
from typing import Optional

from app.chunker import DocumentChunker
from app.models import SummarizeConfig
from app.providers.base import SummarizationProvider, SumResult


# Reserve tokens for system prompt, instructions, and output buffer
_RESERVED_TOKENS = 5000


@dataclass
class SummarizeResult:
    summary: str
    detected_language: str
    code_switching_detected: bool
    model_used: str
    input_tokens: int
    output_tokens: int
    chunks_used: Optional[int]


async def summarize_document(
    text: str,
    config: SummarizeConfig,
    provider: SummarizationProvider,
) -> SummarizeResult:
    """Summarize a document, chunking and using map-reduce if it exceeds the context window."""
    token_budget = provider.max_context_tokens - _RESERVED_TOKENS
    chunker = DocumentChunker(max_tokens_per_chunk=token_budget)
    estimated_tokens = chunker._estimate_tokens(text)

    if estimated_tokens <= token_budget:
        # Fits in one call — no chunking needed
        result = await provider.summarize(text, config)
        return SummarizeResult(
            summary=result.summary,
            detected_language=result.detected_language,
            code_switching_detected=result.code_switching_detected,
            model_used=result.model_used,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            chunks_used=None,
        )

    # Map-reduce: chunk → summarize each → reduce
    chunks = chunker.chunk(text)
    section_summaries = []
    total_input_tokens = 0
    total_output_tokens = 0
    context_bridge = None

    for chunk in chunks:
        chunk_text = chunk.text
        if context_bridge:
            chunk_text = f"[Context: {context_bridge}]\n\n{chunk_text}"

        result = await provider.summarize(chunk_text, config)
        section_summaries.append(result.summary)
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        # Use this summary as bridge for the next chunk
        context_bridge = result.summary[:400]  # Truncate to ~100 tokens

    # Reduce: merge section summaries
    reduce_result = await provider.reduce(section_summaries, config)
    total_input_tokens += reduce_result.input_tokens
    total_output_tokens += reduce_result.output_tokens

    return SummarizeResult(
        summary=reduce_result.summary,
        detected_language=reduce_result.detected_language,
        code_switching_detected=reduce_result.code_switching_detected,
        model_used=reduce_result.model_used,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        chunks_used=len(chunks),
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_summarizer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/summarizer.py tests/test_summarizer.py
git commit -m "feat: add summarization orchestrator with map-reduce flow and context bridging"
```

---

### Task 5: Update Response Metadata

**Files:**
- Modify: `app/models.py`
- Modify: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# Append to tests/test_models.py

def test_summary_metadata_chunks_used_optional():
    """chunks_used should be optional, defaulting to None."""
    meta = SummaryMetadata(
        detected_language="en",
        code_switching_detected=False,
        model_used="test",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200,
    )
    assert meta.chunks_used is None


def test_summary_metadata_chunks_used_set():
    meta = SummaryMetadata(
        detected_language="en",
        code_switching_detected=False,
        model_used="test",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200,
        chunks_used=5,
    )
    assert meta.chunks_used == 5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py::test_summary_metadata_chunks_used_optional tests/test_models.py::test_summary_metadata_chunks_used_set -v`
Expected: FAIL — `chunks_used` field not found

**Step 3: Add field to SummaryMetadata**

In `app/models.py`, add to `SummaryMetadata`:

```python
chunks_used: Optional[int] = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add chunks_used field to SummaryMetadata"
```

---

### Task 6: Wire Orchestrator into main.py

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_main.py`

**Step 1: Write the failing test**

```python
# Append to tests/test_main.py

def test_summarize_returns_chunks_used_field(client, mock_provider):
    """Response should include chunks_used in metadata."""
    response = client.post("/v1/summarize", json={
        "document": "Test document.",
        "config": {"target_language": "en", "summary_type": "brief", "max_length": 100},
    })
    assert response.status_code == 200
    data = response.json()
    assert "chunks_used" in data["metadata"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py::test_summarize_returns_chunks_used_field -v`
Expected: FAIL — `chunks_used` not in response metadata

**Step 3: Update main.py to use orchestrator**

Replace the direct `provider.summarize()` call in `main.py` with `summarize_document()`:

```python
# In app/main.py, add import:
from app.summarizer import summarize_document

# Replace in the summarize route:
#   result = await provider.summarize(doc.text, request.config)
# With:
        result = await summarize_document(doc.text, request.config, provider)

# Update the response to use result fields:
        return SummarizeResponse(
            summary=result.summary,
            metadata=SummaryMetadata(
                detected_language=result.detected_language,
                code_switching_detected=result.code_switching_detected,
                model_used=result.model_used,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=latency_ms,
                evaluation=evaluation,
                chunks_used=result.chunks_used,
            ),
        )
```

Note: existing tests in `test_main.py` mock `provider.summarize()` which returns `SumResult`. The orchestrator calls `provider.summarize()` internally, so existing mocks should still work for short text. Check that `mock_provider` has `max_context_tokens` set — if not, add `mock_provider.max_context_tokens = 200_000` to the fixture.

**Step 4: Run all tests to verify nothing breaks**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: wire summarization orchestrator into API route"
```

---

### Task 7: Integration Test with Real PDF Fixture

**Files:**
- Modify: `tests/test_pdf_integration.py`

**Step 1: Write the test**

This is an integration test verifying the full pipeline (PDF extraction → chunking → orchestrator) works end-to-end with a real fixture. It doesn't call the LLM — it verifies the chunker produces valid chunks from real document text.

```python
# Append to tests/test_pdf_integration.py

def test_chunker_on_real_pdf():
    """Verify chunker produces valid chunks from a real extracted PDF."""
    from app.chunker import DocumentChunker

    processor = DocumentProcessor()
    b64 = _load_pdf_b64("budget_speech_2026_en.pdf")
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
    # At least 90% of original text should be present (whitespace normalization may differ)
    assert len(combined) >= len(doc.text) * 0.9
```

**Step 2: Run test**

Run: `pytest tests/test_pdf_integration.py::test_chunker_on_real_pdf -v`
Expected: PASS (or SKIP if fixture not available)

**Step 3: Commit**

```bash
git add tests/test_pdf_integration.py
git commit -m "test: add integration test for chunker on real PDF fixture"
```
