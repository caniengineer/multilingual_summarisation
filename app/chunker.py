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
