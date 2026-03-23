import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    estimated_tokens: int


class DocumentChunker:
    """Split documents into sections using structural headings, with paragraph fallback."""

    # Heading patterns ordered by priority
    _HEADING_PATTERNS = [
        re.compile(r"^#{1,6}\s+.+", re.MULTILINE),  # Markdown headings
        re.compile(
            r"^[A-Z][A-Z\s:&]{5,}$", re.MULTILINE
        ),  # ALL-CAPS section headers (PDF-extracted)
        re.compile(
            r"^\d+\.\d*\s+\S.+", re.MULTILINE
        ),  # Numbered sections (1.0, 1.1)
        re.compile(
            r"^BAHAGIAN\s+[IVXLCDM]+\b.*", re.MULTILINE | re.IGNORECASE
        ),  # Malaysian government
    ]

    def __init__(self, max_tokens_per_chunk: int = 50_000):
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 characters per token."""
        return max(1, len(text) // 4)

    def _split_sections(self, text: str) -> list[str]:
        """Split text into sections using structural headings, falling back to paragraphs."""
        for pattern in self._HEADING_PATTERNS:
            matches = list(pattern.finditer(text))
            if len(matches) >= 2:
                sections = []
                # Capture text before the first heading
                preamble = text[:matches[0].start()].strip()
                if preamble:
                    sections.append(preamble)
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
                        # Try sentence split, then word split as last resort
                        sentence_pieces = self._split_sentences(para)
                        for sp in sentence_pieces:
                            if self._estimate_tokens(sp) <= self.max_tokens_per_chunk:
                                result.append(sp)
                            else:
                                result.extend(self._split_words(sp))
        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Split text on sentence boundaries to fit within token budget."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = []

        for sentence in sentences:
            candidate = " ".join(current + [sentence])
            if self._estimate_tokens(candidate) > self.max_tokens_per_chunk and current:
                chunks.append(" ".join(current))
                current = [sentence]
            else:
                current.append(sentence)

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _split_words(self, text: str) -> list[str]:
        """Last-resort split on word boundaries to fit within token budget."""
        words = text.split()
        chunks = []
        current = []

        for word in words:
            candidate = " ".join(current + [word])
            if self._estimate_tokens(candidate) > self.max_tokens_per_chunk and current:
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

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
            separator_tokens = self._estimate_tokens("\n\n")
            if current_tokens + separator_tokens + piece_tokens <= self.max_tokens_per_chunk:
                current = current + "\n\n" + piece
                current_tokens += separator_tokens + piece_tokens
            else:
                merged.append(current)
                current = piece
                current_tokens = piece_tokens

        merged.append(current)
        return merged
