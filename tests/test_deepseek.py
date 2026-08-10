import pytest
from ai.deepseek import DeepSeekClient


class FakeResp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_chat_json_parses_content(monkeypatch):
    client = DeepSeekClient(api_key="test-key")
    fake = FakeResp('{"state": "低估机会"}')

    def fake_post(url, headers, json, timeout):
        assert "api.deepseek.com" in url
        return fake

    monkeypatch.setattr(client._session, "post", fake_post)
    out = client.chat_json(
        [{"role": "user", "content": "解读"}],
        {"type": "object", "properties": {"state": {"type": "string"}}},
    )
    assert out == {"state": "低估机会"}


def test_no_api_key_raises():
    from ai.deepseek import DeepSeekClient
    with pytest.raises(ValueError):
        DeepSeekClient(api_key="")
