from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP errors",
    ["method", "endpoint", "error_type"],
)

LLM_CALL_COUNT = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["method", "status"],
)

LLM_CALL_LATENCY = Histogram(
    "llm_call_duration_seconds",
    "LLM API call latency in seconds",
    ["method"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

TOKENS_USED = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["direction"],  # "input" or "output"
)
