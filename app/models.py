from typing import Literal, Optional
from pydantic import BaseModel


class SummarizeConfig(BaseModel):
    target_language: Literal["en", "ms", "auto"] = "auto"
    summary_type: Literal["brief", "detailed", "executive"] = "brief"
    max_length: int = 500
    preserve_domain_terms: bool = True
    evaluate: bool = False


class SummarizeRequest(BaseModel):
    document: str
    document_type: Literal["txt"] = "txt"
    config: SummarizeConfig = SummarizeConfig()


class EvaluationScores(BaseModel):
    faithfulness: float
    coherence: float
    coverage: float
    language_quality: float
    conciseness: float
    justification: str


class SummaryMetadata(BaseModel):
    detected_language: str
    code_switching_detected: bool
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    evaluation: Optional[EvaluationScores] = None


class SummarizeResponse(BaseModel):
    summary: str
    metadata: SummaryMetadata


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
