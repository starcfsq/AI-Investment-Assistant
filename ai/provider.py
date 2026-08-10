"""LLM Provider 抽象。"""
from abc import ABC, abstractmethod

from core.config import get_env


class LLMClient(ABC):
    @abstractmethod
    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        """发送消息，返回符合 schema 的 JSON 对象。"""


def get_client() -> LLMClient:
    # 演示模式：USE_MOCK_LLM=1 时不访问网络，用 MockLLMClient 预览 AI 解读效果
    if get_env("USE_MOCK_LLM", "0") == "1":
        from ai.mock import MockLLMClient

        return MockLLMClient()
    from ai.deepseek import DeepSeekClient

    return DeepSeekClient(api_key=get_env("DEEPSEEK_API_KEY"))
