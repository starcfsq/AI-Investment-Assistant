"""DeepSeek LLM 客户端。"""
import json

import requests

from ai.provider import LLMClient
from core.logging import get_logger

logger = get_logger("ai.deepseek")

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 未配置")
        self.api_key = api_key
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }
        for attempt in range(2):
            resp = self._session.post(API_URL, headers=self._session.headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            try:
                data = json.loads(content)
                _validate(data, schema)
                return data
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("DeepSeek 输出不符合 schema（第 %d 次）: %s", attempt + 1, exc)
        raise ValueError("DeepSeek 输出两次均不符合 schema")


def _validate(data, schema: dict) -> None:
    if schema.get("type") == "object":
        for key, prop in schema.get("properties", {}).items():
            if key not in data:
                raise ValueError(f"缺少字段 {key}")
            if prop.get("type") == "number" and not isinstance(data[key], (int, float)):
                raise ValueError(f"字段 {key} 应为数字")
            if prop.get("type") == "string" and not isinstance(data[key], str):
                raise ValueError(f"字段 {key} 应为字符串")
