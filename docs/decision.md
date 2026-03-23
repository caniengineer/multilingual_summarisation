# Evaluation Metrics Strategy

## What is chrF++

chrF++ is a character n-gram F-score metric for comparing generated text against reference summaries. It is language-agnostic (no tokenizer required), making it suitable for Bahasa Melayu where standard NLP tokenizers are limited.

It reliably detects whether summarization quality has degraded between iterations (prompt changes, model swaps, config tweaks). However, it is not reliable as an absolute quality score because it measures surface-level character overlap, not semantic correctness — valid paraphrases score lower even when meaning is preserved.

## How we use chrF++ with LLM-as-Judge

chrF++ is our fast, free regression gate. LLM-as-Judge (faithfulness, coherence, coverage, language quality, conciseness) is our primary quality assessment — it evaluates semantic quality regardless of surface form.

| Metric | Role | Cost | When |
|--------|------|------|------|
| chrF++ | Regression detection | Free | Every eval run |
| LLM-as-Judge | Semantic quality assessment | 1 API call per sample | Before merges, periodic benchmarks |
