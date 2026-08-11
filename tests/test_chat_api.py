"""chat 端点离线测试：monkeypatch 掉网络与 LLM，验证含历史摘要的 chat 链路不抛异常。"""
import sys

import httpx
import pytest


def _offline_import_api():
    """屏蔽 sentence-transformers 强制走 HashEmbedding 降级，避免 import api.main
    时触发 HuggingFace 联网检查（受限网络下会长时间挂起）。"""
    sys.modules["sentence_transformers"] = None
    import api.main as api

    return api


@pytest.mark.asyncio
async def test_chat_endpoint_offline(monkeypatch):
    api = _offline_import_api()
    from ai.mock import MockLLMClient

    # 离线：不触 akshare 网络，不触 DeepSeek
    monkeypatch.setattr(
        api, "run_analysis",
        lambda provider: {"trend": {"state": "低估"}, "sectors": [], "stocks": [],
                          "portfolio": {}, "data_until": "2026-08-11", "warnings": []},
    )
    monkeypatch.setattr(api, "_get_client", lambda: MockLLMClient())

    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"query": "结合我的持仓，现在该买什么？"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "confidence" in data and "disclaimer" in data
    assert "data_until" in data
