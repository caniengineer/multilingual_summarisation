from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: Optional[str] = None
    PROVIDER: str = "anthropic"
    DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_CODE_MODEL: str = "claude-sonnet-4-20250514"
    LOG_LEVEL: str = "info"
    MAX_DOCUMENT_SIZE: int = 50_000_000  # 50MB base64 ceiling

    # Retry
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BASE_DELAY: float = 1.0
    RETRY_MAX_DELAY: float = 30.0

    # Timeouts
    REQUEST_TIMEOUT: int = 120
    LLM_CALL_TIMEOUT: int = 60

    # Circuit breaker
    CB_FAILURE_THRESHOLD: int = 5
    CB_RECOVERY_TIMEOUT: int = 30
    CB_WINDOW: int = 60

    # Concurrency
    MAX_CONCURRENT_LLM_CALLS: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
