import os
from app.config import Settings


def test_settings_defaults():
    settings = Settings(ANTHROPIC_API_KEY="sk-test-key")
    assert settings.PROVIDER == "anthropic"
    assert settings.DEFAULT_MODEL == "claude-sonnet-4-20250514"
    assert settings.LOG_LEVEL == "info"


def test_settings_api_key_optional():
    # ANTHROPIC_API_KEY is optional at config level (validated at provider creation)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    settings = Settings()
    assert settings.ANTHROPIC_API_KEY is None


def test_settings_provider_validation():
    settings = Settings(ANTHROPIC_API_KEY="sk-test", PROVIDER="ilmu")
    assert settings.PROVIDER == "ilmu"
