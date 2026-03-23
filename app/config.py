from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: Optional[str] = None
    PROVIDER: str = "anthropic"
    DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_CODE_MODEL: str = "claude-sonnet-4-20250514"
    LOG_LEVEL: str = "info"
    MAX_DOCUMENT_SIZE: int = 50_000_000  # 50MB base64 ceiling

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
