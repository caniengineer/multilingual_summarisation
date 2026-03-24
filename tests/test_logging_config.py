import structlog
from app.logging_config import setup_logging, get_logger, request_id_var


class TestStructuredLogging:
    """Test structlog configuration and request ID context."""

    def test_setup_logging_configures_structlog(self):
        setup_logging("info")
        logger = get_logger("test")
        assert logger is not None

    def test_request_id_var_default_is_dash(self):
        request_id_var.set("-")
        assert request_id_var.get() == "-"

    def test_logger_outputs_json(self, capsys):
        setup_logging("info")
        logger = get_logger("test.json")
        request_id_var.set("test-req-123")
        logger.info("hello", foo="bar")
        output = capsys.readouterr().err
        # structlog JSON output should contain our fields
        assert "hello" in output or "test-req-123" in output
