import httpx
import pytest

from ake_backend.demo import seed_demo


@pytest.mark.asyncio
async def test_seed_demo_creates_real_api_upload_and_waits_for_ready():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/v1/auth/login":
            return httpx.Response(200, json={"access_token": "test-token"})
        if request.url.path == "/api/v1/knowledge-bases":
            return httpx.Response(201, json={"id": "base-id"})
        if request.url.path == "/api/v1/knowledge-bases/base-id/documents":
            assert b"%PDF-test" in request.content
            return httpx.Response(202, json={"id": "document-id", "job_id": "job-id"})
        if request.url.path == "/api/v1/documents/document-id/status":
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(404)

    async with httpx.AsyncClient(
        base_url="http://test/api/v1/", transport=httpx.MockTransport(handler)
    ) as client:
        result = await seed_demo(
            client, "demo@example.com", "password", "paper.pdf", b"%PDF-test", poll_seconds=0
        )
    assert result["document_id"] == "document-id"
    assert calls == [
        ("POST", "/api/v1/auth/login"),
        ("POST", "/api/v1/knowledge-bases"),
        ("POST", "/api/v1/knowledge-bases/base-id/documents"),
        ("GET", "/api/v1/documents/document-id/status"),
    ]


@pytest.mark.asyncio
async def test_seed_demo_stops_if_document_processing_fails():
    def handler(request):
        if request.url.path.endswith("/auth/login"):
            return httpx.Response(200, json={"access_token": "test-token"})
        if request.url.path.endswith("/knowledge-bases"):
            return httpx.Response(201, json={"id": "base"})
        if request.url.path.endswith("/documents"):
            return httpx.Response(202, json={"id": "doc", "job_id": "job"})
        return httpx.Response(200, json={"status": "failed"})

    async with httpx.AsyncClient(
        base_url="http://test/api/v1/", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(RuntimeError, match="failed"):
            await seed_demo(
                client, "demo@example.com", "password", "paper.pdf", b"%PDF-test", poll_seconds=0
            )
