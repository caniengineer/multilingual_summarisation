import re
import unicodedata
from dataclasses import dataclass


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
