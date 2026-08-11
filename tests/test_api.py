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


@pytest.mark.asyncio
async def test_simulation_endpoint(monkeypatch):
    import api.main as api
    monkeypatch.setattr(
        api, "run_year_simulation",
        lambda *a, **k: {"stats": {"total_return": 0.1}, "curve": [],
                         "trades": [], "rebalances": []})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/simulation")
    assert resp.status_code == 200
    assert "stats" in resp.json()
