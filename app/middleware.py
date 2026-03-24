import uuid
import time
import asyncio

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import request_id_var, get_logger
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request_id_var.set(request_id)

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
        )

        start = time.time()
        response: Response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)

        response.headers["x-request-id"] = request_id
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration_ms / 1000)

        if response.status_code >= 400:
            ERROR_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                error_type=str(response.status_code),
            ).inc()

        return response


class RequestDeadlineMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout: int = 120):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            response = await asyncio.wait_for(
                call_next(request), timeout=self.timeout
            )
            return response
        except asyncio.TimeoutError:
            logger.error("request_timeout", timeout=self.timeout)
            return Response(
                content='{"detail":"Request timeout"}',
                status_code=504,
                media_type="application/json",
            )
