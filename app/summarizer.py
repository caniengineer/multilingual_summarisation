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
    estimated_tokens = chunker.estimate_tokens(text)

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
