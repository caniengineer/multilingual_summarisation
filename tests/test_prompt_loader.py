import pytest
from app.prompt_loader import PromptLoader


def test_load_summarize_en():
    loader = PromptLoader()
    prompt = loader.load("summarize", language="en")
    assert "summarization assistant" in prompt["template"]
    assert prompt["version"] == "1.0.0"


def test_load_summarize_ms():
    loader = PromptLoader()
    prompt = loader.load("summarize", language="ms")
    assert "Bahasa Melayu" in prompt["template"]


def test_load_summarize_auto():
    loader = PromptLoader()
    prompt = loader.load("summarize", language="auto")
    assert "multilingual" in prompt["template"]


def test_load_evaluate():
    loader = PromptLoader()
    prompt = loader.load("evaluate")
    assert "faithfulness" in prompt["template"]


def test_render_template():
    loader = PromptLoader()
    prompt = loader.load("summarize", language="en")
    rendered = loader.render(
        prompt["template"],
        {
            "max_length": 200,
            "summary_type": "brief",
            "preserve_domain_terms": True,
            "document_text": "Some document text here.",
        },
    )
    assert "200" in rendered
    assert "brief" in rendered
    assert "Some document text here." in rendered


def test_load_nonexistent_raises():
    loader = PromptLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent", language="en")
