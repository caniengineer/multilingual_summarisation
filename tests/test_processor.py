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
