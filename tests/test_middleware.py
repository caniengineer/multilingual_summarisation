import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from app.middleware import RequestIDMiddleware


@pytest.fixture
def test_app():
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    return app


class TestRequestIDMiddleware:
    """Test request ID middleware generates and propagates request IDs."""

    @pytest.mark.asyncio
    async def test_adds_request_id_header(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/test")
        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        # Should be a UUID
        req_id = resp.headers["x-request-id"]
        assert len(req_id) == 36  # UUID4 format

    @pytest.mark.asyncio
    async def test_preserves_caller_request_id(self, test_app):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/test", headers={"X-Request-ID": "caller-id-123"})
        assert resp.headers["x-request-id"] == "caller-id-123"
