import httpx
import pytest

from api.main import app


@pytest.mark.asyncio
async def test_dashboard_returns_json():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "data_until" in data


@pytest.mark.asyncio
async def test_analyze_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/analyze")
    assert resp.status_code == 200
    data = resp.json()
    assert "trend" in data and "sectors" in data


@pytest.mark.asyncio
async def test_unknown_endpoint_404():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/nope")
    assert resp.status_code == 404
