import pytest
from app.processor import DocumentProcessor


def test_process_txt_basic():
    processor = DocumentProcessor()
    result = processor.process("Hello world", doc_type="txt")
    assert result.text == "Hello world"


def test_process_txt_unicode_normalization():
    """NFC normalization: decomposed é (e + combining accent) -> composed é"""
    processor = DocumentProcessor()
    decomposed = "caf\u0065\u0301"  # e + combining acute accent
    result = processor.process(decomposed, doc_type="txt")
    assert result.text == "caf\u00e9"  # composed é


def test_process_txt_whitespace_cleanup():
    processor = DocumentProcessor()
    messy = "  Hello   world  \n\n\n  foo  "
    result = processor.process(messy, doc_type="txt")
    assert "  " not in result.text.replace("\n\n", "")
    assert result.text.strip() == result.text


def test_process_txt_empty_raises():
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="empty"):
        processor.process("", doc_type="txt")


def test_process_txt_whitespace_only_raises():
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="empty"):
        processor.process("   \n\n  ", doc_type="txt")


def test_process_unsupported_type_raises():
    processor = DocumentProcessor()
    with pytest.raises(ValueError, match="Unsupported"):
        processor.process("data", doc_type="pdf")


def test_process_returns_token_estimate():
    processor = DocumentProcessor()
    result = processor.process("Hello world this is a test", doc_type="txt")
    assert result.estimated_tokens > 0


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
