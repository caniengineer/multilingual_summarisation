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
        # Rough estimate: ~4 chars per token for English, ~3 for Malay
        return max(1, len(text) // 4)

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
