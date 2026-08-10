"""端到端：数据→分析→AI(降级)→账户 全链路不抛异常。"""
import os
import tempfile

import pytest

from core.analyze import run_analysis
from core.config import DB_PATH, load_weights
from core.data import DataProvider
from core.store import Store


def test_full_pipeline_offline_safe(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    store = Store(tempfile.mkdtemp() + "/t.db")
    provider = DataProvider(store)
    result = run_analysis(provider)
    assert isinstance(result, dict)
    assert "trend" in result and "sectors" in result
    assert "stocks" in result and "portfolio" in result


def test_weights_config_parseable():
    w = load_weights()
    for key in ("trend", "sector", "stock"):
        assert key in w
        assert abs(sum(w[key].values()) - 1.0) < 1e-9
