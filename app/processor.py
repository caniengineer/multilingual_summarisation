import base64 as b64_mod
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import fitz  # PyMuPDF
import ftfy

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedDocument:
    text: str
    estimated_tokens: int


class DocumentProcessor:
    SUPPORTED_TYPES = {"txt", "pdf"}

    def process(self, raw: str, doc_type: str) -> ProcessedDocument:
        if doc_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported document type: {doc_type}")

        logger.info("document_processing_started", doc_type=doc_type)

        if doc_type == "pdf":
            text = self._extract_pdf(raw)
        else:
            text = raw

        text = self._normalize(text)

        if not text:
            raise ValueError("Document is empty after processing")

        logger.info("document_processing_completed", doc_type=doc_type, estimated_tokens=self._estimate_tokens(text))

        return ProcessedDocument(
            text=text,
            estimated_tokens=self._estimate_tokens(text),
        )

    def _extract_pdf(self, b64_data: str) -> str:
        """Decode base64 -> open PDF -> extract per-page text -> strip headers/footers."""
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

        logger.info("pdf_extraction_completed", page_count=len(pages))
        pages = self._strip_headers_footers(pages)
        return "\n\n".join(pages)

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

    def _strip_headers_footers(self, pages: list[str]) -> list[str]:
        """Remove repeated header/footer lines from paginated text.

        Heuristic: lines in the first or last 3 lines of a page that appear
        across >50% of pages are likely headers/footers.
        """
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
                line
                for line in lines
                if line.strip() not in to_remove and not page_num_re.match(line.strip())
            ]
            cleaned.append("\n".join(filtered))
        return cleaned

    def _estimate_tokens(self, text: str) -> int:
        # Rough estimate: ~4 chars per token for English, ~3 for Malay
        return max(1, len(text) // 4)
