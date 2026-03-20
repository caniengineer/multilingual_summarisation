import os
import pytest
from app.config import Settings


def test_settings_defaults():
    settings = Settings(ANTHROPIC_API_KEY="sk-test-key")
    assert settings.PROVIDER == "anthropic"
    assert settings.DEFAULT_MODEL == "claude-sonnet-4-20250514"
    assert settings.LOG_LEVEL == "info"


def test_settings_requires_api_key():
    # Clear env var if set
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(Exception):
        Settings()


def test_settings_provider_validation():
    settings = Settings(ANTHROPIC_API_KEY="sk-test", PROVIDER="ilmu")
    assert settings.PROVIDER == "ilmu"
