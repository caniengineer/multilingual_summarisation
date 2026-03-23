from typing import Literal, Optional
from pydantic import BaseModel, Field


class SummarizeConfig(BaseModel):
    target_language: Literal["en", "ms", "auto"] = "auto"
    summary_type: Literal["brief", "detailed", "executive"] = "brief"
    max_length: int = 500
    preserve_domain_terms: bool = True
    evaluate: bool = False


class SummarizeRequest(BaseModel):
    document: str = Field(max_length=50_000_000)  # ~37.5MB raw after base64 decode
    document_type: Literal["txt", "pdf"] = "txt"
    config: SummarizeConfig = SummarizeConfig()


class EvaluationScores(BaseModel):
    faithfulness: float = Field(ge=1.0, le=5.0)
    coherence: float = Field(ge=1.0, le=5.0)
    coverage: float = Field(ge=1.0, le=5.0)
    language_quality: float = Field(ge=1.0, le=5.0)
    conciseness: float = Field(ge=1.0, le=5.0)
    justification: str


class SummaryMetadata(BaseModel):
    detected_language: str
    code_switching_detected: bool
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    evaluation: Optional[EvaluationScores] = None
    chunks_used: Optional[int] = None


class SummarizeResponse(BaseModel):
    summary: str
    metadata: SummaryMetadata


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
