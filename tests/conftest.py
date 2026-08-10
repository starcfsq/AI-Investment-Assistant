"""pytest 全局配置。

api/main 在导入时会调用 get_client()，需要 DEEPSEEK_API_KEY，
否则 import 即抛 ValueError。这里在收集测试前用占位值兜底，
保证离线环境也能通过 httpx.ASGITransport 直测。
"""
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test")
