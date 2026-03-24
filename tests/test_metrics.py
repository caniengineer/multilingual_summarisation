from app.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
    TOKENS_USED,
    LLM_CALL_COUNT,
    LLM_CALL_LATENCY,
)


class TestMetricDefinitions:
    """Verify Prometheus metric objects are properly defined."""

    def test_metrics_exist(self):
        assert REQUEST_COUNT is not None
        assert REQUEST_LATENCY is not None
        assert ERROR_COUNT is not None
        assert TOKENS_USED is not None
        assert LLM_CALL_COUNT is not None
        assert LLM_CALL_LATENCY is not None

    def test_request_count_increments(self):
        before = REQUEST_COUNT.labels(method="GET", endpoint="/test", status="200")._value.get()
        REQUEST_COUNT.labels(method="GET", endpoint="/test", status="200").inc()
        after = REQUEST_COUNT.labels(method="GET", endpoint="/test", status="200")._value.get()
        assert after == before + 1
