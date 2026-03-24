import logging
import sys
from contextvars import ContextVar

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _add_request_id(logger, method_name, event_dict):
    event_dict["request_id"] = request_id_var.get()
    return event_dict


def setup_logging(log_level: str = "info") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_request_id,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to route through structlog
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)